import os, threading, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextIteratorStreamer,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma   # 官方來源

# ───── Configuration ─────
LLM_MODEL   = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS  = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FOLDER  = "./PDFS"
DB_PATH     = "./VectorDB"

SYSTEM_PROMPT = (
    "你是「ERAS 醫療專案管理系統」中的智慧個案管理師，專門負責協助病患完成術前、術後的衛教與追蹤。"
    "你的回答應依據 ERAS（Enhanced Recovery After Surgery）指引內容，並以清楚、簡單、友善的語氣引導病患完成待辦事項。"
    "若無法確定答案，請提醒病患聯繫醫療團隊。請勿提及自己是大型語言模型或內部開發細節，只需以 ERAS 個管師的身份專業回應。"
)

# =======================
# LLM Initialization
# =======================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
).eval()
MAX_CTX = tokenizer.model_max_length

# =======================
# Build / Load VectorDB
# =======================
def init_vectorstore(folder: str, persist: str, emb_model: str) -> Chroma:
    if not os.path.exists(persist) or not os.listdir(persist):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = []
        for f in os.listdir(folder):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder, f))
                for c in splitter.split_documents(loader.load_and_split()):
                    c.metadata["source_pdf"] = f
                    docs.append(c)
        embed = HuggingFaceEmbeddings(model_name=emb_model)
        return Chroma.from_documents(docs, embed, persist_directory=persist)

    embed = HuggingFaceEmbeddings(model_name=emb_model)
    return Chroma(embedding_function=embed, persist_directory=persist)

vectordb = init_vectorstore(PDF_FOLDER, DB_PATH, EMBEDDINGS)

# =======================
# Helper
# =======================
def tlen(txt: str) -> int:
    return len(tokenizer.encode(txt, add_special_tokens=False))

# =======================
# Memory Builder
# =======================
def build_memory(hist, enable, base_tok=0, reserve=512):
    if not enable: return ""
    blocks, i = [], len(hist)-1
    while i >= 1:
        if hist[i]["role"]=="assistant" and hist[i-1]["role"]=="user":
            blk = (f"<|im_start|>user\n{hist[i-1]['content']}\n<|im_end|>\n"
                   f"<|im_start|>assistant\n{hist[i]['content']}\n<|im_end|>\n")
            bt = tlen(blk)
            if base_tok+bt > reserve: break
            blocks.append(blk); base_tok += bt; i-=2
        else: i-=1
    return "".join(reversed(blocks))

def _build_prompt(query, sel_pdf, enable_mem, hist, max_tokens):
    # RAG
    rag, srcs = "", []
    if sel_pdf!="No PDFs":
        kw = {"k":5}
        if sel_pdf!="All PDFs": kw["filter"]={"source_pdf": sel_pdf}
        docs = vectordb.similarity_search(query, **kw)
        if docs:
            cur, ch, limit = 0, [], MAX_CTX-max_tokens
            for d in docs:
                chunk=f"[{d.metadata.get('source_pdf','?')}]\n{d.page_content.strip()}"
                ct=tlen(chunk)
                if cur+ct>limit: break
                ch.append(chunk); cur+=ct; srcs.append(d.metadata["source_pdf"])
            rag="\n\n".join(ch)

    sys = (f"<|im_start|>system\n{SYSTEM_PROMPT}\nContext:\n{rag}\n<|im_end|>\n"
           if rag else f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n")
    mem = build_memory(hist, enable_mem, base_tok=tlen(sys)+tlen(query),
                       reserve=MAX_CTX-max_tokens)
    prompt = (sys+mem+f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
    return prompt, sorted(set(srcs))

# =========================
# Non-Streaming Reply API
# =========================
def generate_response(query, selected_pdf="All PDFs", enable_memory=True,
                      history=None, max_tokens=256, temperature=0.1):
    history = history or []
    prompt, srcs = _build_prompt(query, selected_pdf, enable_memory,
                                 history, max_tokens)
    out = model.generate(
        tokenizer(prompt, return_tensors="pt").to(model.device).input_ids,
        max_new_tokens=max_tokens, temperature=temperature)
    reply = tokenizer.decode(out[0], skip_special_tokens=True)\
            .split("<|im_start|>assistant\n")[-1].strip()
    new_hist = history+[
        {"role":"user","content":query},
        {"role":"assistant","content":reply},
    ]
    return {"reply":reply, "sources":srcs, "updated_history":new_hist}

# =======================
# Streaming Reply API
# =======================
def stream_response(query, selected_pdf="All PDFs", enable_memory=True,
                    history=None, max_tokens=256, temperature=0.1):
    history = history or []
    prompt, srcs = _build_prompt(query, selected_pdf, enable_memory,
                                 history, max_tokens)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    def _gen():
        model.generate(
            tokenizer(prompt, return_tensors="pt").to(model.device).input_ids,
            max_new_tokens=max_tokens, temperature=temperature, streamer=streamer)
    threading.Thread(target=_gen).start()

    prefix = f"資料來源：{', '.join(srcs)}\n\n" if srcs else ""
    partial = prefix
    for tok in streamer:
        partial += tok
        yield {"reply":partial, "sources":srcs,
               "updated_history": history+[
                   {"role":"user","content":query},
                   {"role":"assistant","content":partial}]}

# Test the API
if __name__ == "__main__":
    pass
    # Q = "手術前一天我可以吃什麼？"
    # print("── Non-Streaming ──")
    # res = generate_response(Q)
    # print("回覆:\n", res["reply"])
    # print("來源:", res["sources"])

    # print("\n── Streaming ──")
    # for chunk in stream_response(Q):
    #     os.system('clear')   # Remove if you don't want clear screen effect
    #     print(chunk["reply"])