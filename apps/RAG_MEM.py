import os, torch, gradio as gr, threading
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =======================
# Configuration
# =======================
LLM_MODEL   = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS  = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FOLDER  = "./PDFS"
DB_PATH     = "./VectorDB"
GEN_TOKENS  = 256
TEMPERATURE = 0.1

SYSTEM_PROMPT = (
    "你是「ERAS 醫療專案管理系統」中的智慧個案管理師，專門負責協助病患完成術前、術後的衛教與追蹤。"
    "你的回答應依據 ERAS（Enhanced Recovery After Surgery）指引內容，並以清楚、簡單、友善的語氣引導病患完成待辦事項。"
    "若無法確定答案，請提醒病患聯繫醫療團隊。請勿提及自己是大型語言模型或內部開發細節，只需以 ERAS 個管師的身份專業回應。"
)

# =======================
# LLM Initialization
# =======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

MAX_CTX = tokenizer.model_max_length
RESERVED_FOR_PROMPT = MAX_CTX - GEN_TOKENS

# =======================
# Build / Load VectorDB
# =======================
def init_vectorstore(folder_path: str, persist_dir: str, embedding_model: str) -> Chroma:
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_docs = []
        for fn in os.listdir(folder_path):
            if fn.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, fn))
                pages = loader.load_and_split()
                chunks = splitter.split_documents(pages)
                for c in chunks:
                    c.metadata["source_pdf"] = fn
                all_docs.extend(chunks)

        embed = HuggingFaceEmbeddings(model_name=embedding_model)
        db = Chroma.from_documents(
            documents=all_docs,
            embedding=embed,
            persist_directory=persist_dir,
        )
        db.persist()
        return db

    embed = HuggingFaceEmbeddings(model_name=embedding_model)
    db = Chroma(
        embedding_function=embed,
        persist_directory=persist_dir,
    )
    return db

vectordb = init_vectorstore(PDF_FOLDER, DB_PATH, EMBEDDINGS)

# =======================
# Helper
# =======================
def tlen(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def stream_generate(prompt: str):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids

    def _run():
        model.generate(
            input_ids=input_ids,
            max_new_tokens=GEN_TOKENS,
            temperature=TEMPERATURE,
            streamer=streamer,
        )

    threading.Thread(target=_run).start()
    for token in streamer:
        yield token

# =======================
# Memory Builder
# =======================
def build_memory(history_msgs: list, enable: bool, base_tokens: int = 0) -> str:
    if not enable:
        return ""
    memory_blocks = []
    # Reverse scan, latest conversation first
    i = len(history_msgs) - 1
    while i >= 1:
        if history_msgs[i]["role"] == "assistant" and history_msgs[i-1]["role"] == "user":
            block = (
                f"<|im_start|>user\n{history_msgs[i-1]['content']}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{history_msgs[i]['content']}\n<|im_end|>\n"
            )
            block_tok = tlen(block)
            if base_tokens + block_tok > RESERVED_FOR_PROMPT:
                break
            memory_blocks.append(block)
            base_tokens += block_tok
            i -= 2
        else:
            i -= 1
    return "".join(reversed(memory_blocks))

# =======================
# Streaming Reply Logic
# =======================
def generate_reply(query: str, selected_pdf: str, enable_memory: bool, history: list):
    # ---------- Step 0: prepare RAG ----------
    rag_context, sources = "", []
    if selected_pdf not in ["No PDFs"]:
        search_kwargs = {"k": 5}
        if selected_pdf != "All PDFs":
            search_kwargs["filter"] = {"source_pdf": selected_pdf}
        docs = vectordb.similarity_search(query, **search_kwargs)
        if docs:
            current_tok = 0
            chunks = []
            for d in docs:
                chunk = f"[{d.metadata.get('source_pdf','Unknown')}]\n{d.page_content.strip()}"
                ctok  = tlen(chunk)
                if current_tok + ctok > RESERVED_FOR_PROMPT:
                    break
                chunks.append(chunk)
                current_tok += ctok
                sources.append(d.metadata.get("source_pdf", "Unknown"))
            rag_context = "\n\n".join(chunks)

    # ---------- Step 1: System + Memory ----------

    system_msg = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
        f"Context:\n{rag_context}\n<|im_end|>\n"
    ) if rag_context else (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
    )

    memory_context = build_memory(history, enable_memory,
                                  base_tokens=tlen(system_msg) + tlen(query))

    prompt = (
        system_msg
        + memory_context
        + f"<|im_start|>user\n{query}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # ---------- Step 2: Insert user message, lock input ----------
    history.append({"role": "user", "content": query})
    assistant_msg = {"role": "assistant", "content": ""}
    history.append(assistant_msg)
    chatbot_view = history.copy()

    yield chatbot_view, history, ""

    # ---------- Step 3: Streaming tokens ----------
    src_prefix = f"資料來源：{', '.join(sorted(set(sources)))}\n\n" if sources else ""
    partial = src_prefix
    for token in stream_generate(prompt):
        partial += token
        assistant_msg["content"] = partial
        chatbot_view[-1]["content"] = partial
        yield chatbot_view, history, ""

    # ---------- Step 4: Finished, unlock input ----------
    yield chatbot_view, history, ""

# =======================
# Gradio Interface
# =======================
def launch_gradio():
    all_pdfs    = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    pdf_options = ["No PDFs", "All PDFs"] + sorted(all_pdfs)

    with gr.Blocks(css="footer{visibility:hidden}; .gr-button{min-height:42px;}") as demo:
        gr.Markdown("# 醫學電資整合創意專題(二) ERAS 醫療專案管理系統")
        gr.Markdown("### 團隊成員：傅冠豪、陳冠宇、金哲安、陳孟潔、楊哲瑜、倪昕、張玠")

        chatbot         = gr.Chatbot(label="Chat History", show_copy_button=True, type="messages")
        selected_pdf    = gr.Dropdown(choices=pdf_options, value="No PDFs", label="選擇文件範圍")
        memory_checkbox = gr.Checkbox(label="啟用記憶", value=True)

        with gr.Row(equal_height=True):
            msg    = gr.Textbox(
                label="輸入問題",
                placeholder="請輸入問題，Shift + Enter 換行，按下 Enter 送出",
                lines=1,
            )
            submit = gr.Button("送出", variant="primary")

        clear  = gr.Button("清除對話")
        state  = gr.State([]) # Save messages format list

        msg.submit(
            fn=generate_reply,
            inputs=[msg, selected_pdf, memory_checkbox, state],
            outputs=[chatbot, state, msg],
        )

        submit.click(
            fn=generate_reply,
            inputs=[msg, selected_pdf, memory_checkbox, state],
            outputs=[chatbot, state, msg],
        )

        def clear_chat():
            return [], [], ""

        clear.click(
            fn=clear_chat,
            outputs=[chatbot, state, msg],
        )

        gr.Markdown("下載 PDF 文件：")
        for f in sorted(os.listdir(PDF_FOLDER)):
            if f.endswith(".pdf"):
                gr.File(
                    value=os.path.join(PDF_FOLDER, f),
                    file_types=[".pdf"],
                    label=f"📄 {f}",
                    interactive=True,
                )

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()