import os, glob, torch
from PIL import Image
from typing import List
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import PeftModel

MODEL_DIR   = "./qwen25vl_lora_merged"
BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, device_map="auto", quantization_config=bnb_cfg,
        torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # LoRA
    model = PeftModel.from_pretrained(base, MODEL_DIR, device_map="auto")
    model.eval()
    return model, processor


def generate_report(img_paths, model, processor, device="cuda"):
    images = [Image.open(p).convert("RGB").resize((448, 448))
              for p in img_paths]
    prompt = (
        "You are a radiologist. Please write a radiology report for the given spine X-ray images in the following structured format:\n\n"
        "Radiography of [Spine Region] ([View(s)]) show::\n"
        "- [Finding 1]\n"
        "- [Finding 2]\n"
        "- [Finding 3]\n"
        "- [Finding 4]\n"
        "- ..."
    )
    msg_imgs = [{"type": "image", "image": img} for img in images]
    messages = [
        {"role": "user", "content": msg_imgs + [{"type": "text", "text": prompt}]},
        {"role": "assistant"} # assistant＝Generation start
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt"
    ).to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True, # Sampling
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # Truncate the generated text to remove the prompt
    text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    answer = text.split("assistant\n")[-1].lstrip()

    return answer


def infer_folder(root: str, model, processor):
    # Folder with ap.jpg, lat.jpg, report.txt
    sub_dirs = sorted(glob.glob(os.path.join(root, "sub*")))
    for sub in sub_dirs:
        ap = os.path.join(sub, "ap.jpg")
        lat = os.path.join(sub, "lat.jpg")
        rpt_path = os.path.join(sub, "report.txt")

        imgs = [ap] if not os.path.exists(lat) else [ap, lat]
        model_answer = generate_report(imgs, model, processor)

        print(f"\n--- {os.path.basename(sub)} ---")
        print("Model Answer:\n", model_answer)

        # With ground truth
        if os.path.exists(rpt_path):
            with open(rpt_path, encoding="utf-8") as f:
                gt = f.read().strip()
            print("\nGround Truth:\n", gt)
        else:
            print("\nNo Ground Truth found.")


if __name__ == "__main__":
    model, proc = load_model()
    # Inference with test subfolders
    infer_folder("Images_anonymized/test", model, proc)