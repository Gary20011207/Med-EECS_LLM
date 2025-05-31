#!/usr/bin/env python
# finetune_qwen25_vl_radiology.py  (最終版)

import os, glob, re, torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    BitsAndBytesConfig, TrainingArguments, Trainer, default_data_collator
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig


# ---------- utils ----------
def clean_report(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------- Dataset ----------
class DualViewDataset(Dataset):
    PATCH_HW = (32, 32)                       # 448 / 14

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
            "You are a radiologist. Based on the following spine X-ray images "
            "(AP and Lateral views), generate a structured radiology report in this format:\n\n"
            "Radiography of ..."
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

        full_ids = self.proc.tokenizer(full_prompt, return_tensors="pt",
                                       max_length=1024, truncation=True)["input_ids"][0]
        prompt_ids = self.proc.tokenizer(prompt_only, return_tensors="pt",
                                         max_length=1024, truncation=True)["input_ids"][0]

        inputs = self.proc(text=full_prompt, images=imgs, return_tensors="pt",
                           padding="max_length", max_length=1024, truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        labels[: len(prompt_ids)] = -100
        inputs["labels"] = labels

        H, W = self.PATCH_HW
        inputs["image_grid_thw"] = torch.tensor(
            [(1, H, W)] * len(imgs),          # len(imgs)=1 或 2
            dtype=torch.long
        )
        return inputs


# ---------- collator ----------
from transformers import default_data_collator

def qwen_collator(features):
    # 先處理其它鍵
    batch = default_data_collator(
        [{k: v for k, v in f.items() if k != "image_grid_thw"} for f in features]
    )

    if len(features) == 1:
        # 保留 (n_img, 3) tensor，不要多一層
        batch["image_grid_thw"] = features[0]["image_grid_thw"]
    else:
        # 若未來你想 batch>1，就手動 stack → shape (B, n_img, 3)
        batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])

    return batch


# ---------- main ----------
def main():
    root = "./Images_anonymized/train/"

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
            target_modules=
            ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "vision_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
    )

    ds_train = DualViewDataset(root, proc)
    print(f"使用訓練樣本={len(ds_train)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./qwen25vl_lora_radiology",
            per_device_train_batch_size=1,
            num_train_epochs=30,
            learning_rate=5e-5,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none"
        ),
        train_dataset=ds_train,
        data_collator=qwen_collator
    )
    trainer.train()
    model.save_pretrained("./qwen25vl_lora_radiology")
    proc.tokenizer.save_pretrained("./qwen25vl_lora_radiology")


if __name__ == "__main__":
    main()