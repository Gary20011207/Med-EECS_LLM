#!/usr/bin/env python
# infer_qwen25_vl_radiology.py
# ----------------------------
# 用微調後 LoRA 權重，對新 X-ray 產生報告

import os, glob, torch
from PIL import Image
from typing import List
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import PeftModel


MODEL_DIR   = "./qwen25vl_lora_radiology"   # ← LoRA 輸出路徑
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

    # 掛上 LoRA
    model = PeftModel.from_pretrained(base, MODEL_DIR, device_map="auto")
    model.eval()
    return model, processor


def generate_report(img_paths, model, processor, device="cuda"):
    # 1. 讀圖
    images = [Image.open(p).convert("RGB").resize((448, 448))
              for p in img_paths]

    # 2. 準備 chat messages
    prompt = (
        "You are a radiologist. Based on the following spine X-ray images "
        "(AP and Lateral views), generate a structured radiology report in this format:\n\n"
        "Radiography of ..."
    )
    msg_imgs = [{"type": "image", "image": img} for img in images]
    messages = [
        {"role": "user", "content": msg_imgs + [{"type": "text", "text": prompt}]},
        {"role": "assistant"}                       # ★ 空 assistant＝生成起點
    ]

    # 3. 轉成 prompt text
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 4. 轉 tensor（注意全部用關鍵字）
    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt"
    ).to(device)

    # 5. 生成
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,           # 建議打開 sampling
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # 6. 從「assistant 標記」之後截取
    text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    answer = text.split("assistant\n")[-1].lstrip()

    return answer


def infer_folder(root: str, model, processor):
    """假設 root/subX/ 內有 ap.jpg, lat.jpg，依序推論"""
    sub_dirs = sorted(glob.glob(os.path.join(root, "sub*")))
    for sub in sub_dirs:
        ap = os.path.join(sub, "ap.jpg")
        lat = os.path.join(sub, "lat.jpg")
        imgs = [ap] if not os.path.exists(lat) else [ap, lat]

        report = generate_report(imgs, model, processor)
        print(f"\n--- {os.path.basename(sub)} ---")
        print(report)


if __name__ == "__main__":
    import argparse, textwrap
    parser = argparse.ArgumentParser(
        description="Generate radiology reports with fine-tuned Qwen-2.5-VL-LoRA",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("img", nargs="*", help=textwrap.dedent("""
        1) 指定 1~2 張影像路徑直接推論；例如：
           python gen_report/generate_report.py Images_anonymized/sub001/ap.jpg Images_anonymized/sub001/lat.jpg
        2) 若不給參數，預設掃描 ./Images_test/sub*/ 資料夾
    """))
    args = parser.parse_args()

    model, proc = load_model()

    if args.img:
        print(generate_report(args.img, model, proc))
    else:
        infer_folder("Images_anonymized/test", model, proc)