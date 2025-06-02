import os, glob, re, torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from transformers import default_data_collator
import numpy as np

# ---------- utils ----------
def clean_report(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ---------- Dataset ----------
class DualViewDataset(Dataset):
    PATCH_HW = (32, 32)

    def __init__(self, root: str, processor):
        self.proc = processor
        self.samples = []
        for sub in sorted(glob.glob(os.path.join(root, "sub*"))):
            ap, lat, rep = (os.path.join(sub, n) for n in ("ap.jpg", "lat.jpg", "report.txt"))
            if not (os.path.exists(ap) and os.path.exists(rep)):
                continue
            imgs = [Image.open(ap).convert("RGB").resize((448, 448))]
            if os.path.exists(lat):
                imgs.append(Image.open(lat).convert("RGB").resize((448, 448)))
            with open(rep, encoding="utf-8") as f:
                rpt = clean_report(f.read())
            if rpt:
                self.samples.append((imgs, rpt))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        imgs, rpt = self.samples[idx]
        prompt = (
            "You are a radiologist. Please write a radiology report for the given spine X-ray images in the following structured format:\n\n"
            "Radiography of [Spine Region] ([View(s)]) show::\n"
            "- [Finding 1]\n"
            "- [Finding 2]\n"
            "- [Finding 3]\n"
            "- [Finding 4]\n"
            "- ..."
        )
        msg_imgs = [{"type": "image", "image": im} for im in imgs]
        messages = [
            {"role": "user", "content": msg_imgs + [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": rpt},
        ]
        msg_user_only = messages[:1]

        full_prompt = self.proc.apply_chat_template(messages, tokenize=False)
        prompt_only = self.proc.apply_chat_template(
            msg_user_only, tokenize=False, add_generation_prompt=True)

        inputs = self.proc(text=full_prompt, images=imgs, return_tensors="pt",
                           padding="max_length", max_length=1024, truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        prompt_ids = self.proc.tokenizer(prompt_only, return_tensors="pt",
                                         max_length=1024, truncation=True)["input_ids"][0]
        labels = inputs["input_ids"].clone()
        labels[: len(prompt_ids)] = -100
        inputs["labels"] = labels

        H, W = self.PATCH_HW
        inputs["image_grid_thw"] = torch.tensor(
            [(1, H, W)] * len(imgs), dtype=torch.long
        )
        return inputs

# ---------- collator ----------
def qwen_collator(features):
    batch = default_data_collator(
        [{k: v for k, v in f.items() if k != "image_grid_thw"} for f in features]
    )
    if len(features) == 1:
        batch["image_grid_thw"] = features[0]["image_grid_thw"]
    else:
        batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])
    return batch

# ---------- main ----------
def main():
    root = "./Images_anonymized/train/"
    ds = DualViewDataset(root, AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct"))
    print(f"總共樣本數：{len(ds)}")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold = 0

    for train_idx, val_idx in kf.split(ds):
        fold += 1
        print(f"\n=========== Fold {fold} ===========")

        train_set = Subset(ds, train_idx)
        val_set = Subset(ds, val_idx)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto",
            quantization_config=bnb, torch_dtype=torch.float16
        )
        proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        base = prepare_model_for_kbit_training(base)
        model = get_peft_model(
            base,
            LoraConfig(
                r=8, lora_alpha=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "vision_proj"
                ],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./qwen25vl_lora_fold{fold}",
                per_device_train_batch_size=1,
                num_train_epochs=50,
                learning_rate=5e-5,
                warmup_steps=5,
                weight_decay=0.01,
                save_strategy="no",
                logging_steps=1,
                report_to="none"

            ),
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=qwen_collator
        )
        trainer.train()
        model.save_pretrained(f"./qwen25vl_lora_fold{fold}")
        proc.tokenizer.save_pretrained(f"./qwen25vl_lora_fold{fold}")

if __name__ == "__main__":
    main()