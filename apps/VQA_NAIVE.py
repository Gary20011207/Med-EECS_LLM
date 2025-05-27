#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal ERAS Assistant – Qwen2.5-VL Integration
---------------------------------------------------
* LLM : Qwen/Qwen2.5-VL-3B-Instruct
* UI  : Gradio chat with optional image upload
"""

import os, io, base64, typing, builtins, torch, gradio as gr
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

assert torch.cuda.is_available(), "需要 CUDA GPU 才能執行 Qwen-VL 模型"

# ------------------- Model & Processor ---------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TEMPERATURE = 0.7
MAX_HISTORY = 10

SYSTEM_PROMPT = (
    "你是「ERAS 醫療專案管理系統」中的智慧個案管理師，專門負責協助病患完成術前、術後的衛教與追蹤。"
    "你的回答應依據 ERAS 指引內容，並以清楚、簡單、友善的語氣引導病患完成待辦事項。"
    "若無法確定答案，請提醒病患聯繫醫療團隊。請勿揭露自己是大型語言模型或內部細節。"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# -------------------- Utilities --------------------
def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def make_chat_view(hist: list):
    return [{"role": m["role"], "content": m.get("display", m["content"])} for m in hist]

# -------------------- Core Chat Logic --------------------
def generate_reply(img: Image.Image, query: str, history: list):
    # —— 1. Display image + query in chat bubble
    if img is not None:
        img_b64 = pil_to_base64(img)
        md_img = f'<img src="data:image/png;base64,{img_b64}" width="200"/>'
        user_display = f"{md_img}\n\n{query}"
        has_image = True
    else:
        user_display = query
        has_image = False

    user_msg = {
        "role": "user",
        "content": [img, query] if has_image else query,
        "display": user_display,
        "has_image": has_image
    }
    history.append(user_msg)

    assistant_msg = {"role": "assistant", "content": "", "display": ""}
    history.append(assistant_msg)

    # —— 2. Yield user message first (讓 Gradio 立即顯示)
    yield make_chat_view(history), history, "", None

    # —— 3. 建構 Qwen 格式的 messages
    if has_image:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": query}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

    # —— 4. 模型生成
    generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=TEMPERATURE)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    assistant_msg["content"] = output_text
    assistant_msg["display"] = output_text

    yield make_chat_view(history), history, "", None

# -------------------- Gradio UI --------------------
def launch_gradio():
    with gr.Blocks(css="footer{visibility:hidden}; .gr-button{min-height:42px;}") as demo:
        gr.Markdown("# 醫學電資整合創意專題(二) ERAS 醫療專案管理系統")
        gr.Markdown("### 團隊成員：傅冠豪、陳冠宇、金哲安、陳孟潔、楊哲瑜、倪昕、張玠")

        chatbot = gr.Chatbot(label="Chat History", show_copy_button=True, type="messages")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_in = gr.Image(label="上傳影像 (可留空)", type="pil", sources=["upload", "clipboard"])
            with gr.Column(scale=2):
                msg_in = gr.Textbox(label="輸入問題", placeholder="請輸入問題，Shift+Enter 換行，Enter 送出", lines=1)

        submit_btn = gr.Button("送出", variant="primary")
        clear_btn = gr.Button("清除對話")
        state = gr.State([])

        def _clear():
            return [], [], "", None

        msg_in.submit(generate_reply, [img_in, msg_in, state], [chatbot, state, msg_in, img_in])
        submit_btn.click(generate_reply, [img_in, msg_in, state], [chatbot, state, msg_in, img_in])
        clear_btn.click(_clear, outputs=[chatbot, state, msg_in, img_in])

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

# ------------------- main ----------------------
if __name__ == "__main__":
    launch_gradio()
