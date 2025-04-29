# apps/RAG_NAIVE.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =======================
# Configuration
# =======================
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FOLDER = "./PDFS"
DB_PATH = "./VectorDB"

# =======================
# 環境變數控制
# =======================
RUN_MODE = os.getenv("RAG_MODE", "REAL")  # 預設是 REAL，也可以設成 TEST

# =======================
# 正式版初始化（只有在 REAL 模式才跑）
# =======================
if RUN_MODE == "REAL":
    # Load model
    bnb_config = None
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except ImportError:
        pass  # 沒有bitsandbytes就跳過（只要是CPU就不需要）

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=256,
        temperature=0.1,
    )

    # Initialize vector database
    def init_vectorstore(folder_path, persist_dir, embedding_model):
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            all_docs = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    full_path = os.path.join(folder_path, filename)
                    loader = PyPDFLoader(full_path)
                    pages = loader.load_and_split()
                    chunks = splitter.split_documents(pages)
                    for chunk in chunks:
                        chunk.metadata["source_pdf"] = filename
                    all_docs.extend(chunks)
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            Chroma.from_documents(all_docs, embeddings, persist_directory=persist_dir)

        vectordb = Chroma(
            embedding_function=HuggingFaceEmbeddings(model_name=embedding_model),
            persist_directory=persist_dir
        )
        return vectordb

    vectordb = init_vectorstore(PDF_FOLDER, DB_PATH, EMBEDDINGS)

# =======================
# generate_reply
# =======================
def generate_reply(query, selected_pdf="All PDFs", history=None):
    if history is None:
        history = []

    if RUN_MODE == "TEST":
        fake_reply = f"(測試回覆) 你問的是：「{query}」"
        history.append((query, fake_reply))
        return history, history

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    all_docs = retriever.get_relevant_documents(query)

    docs = all_docs

    if not docs:
        reply = f"找不到相關資訊！"
        history.append((query, reply))
        return history, history

    context_chunks = []
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source_pdf", "Unknown")
        sources.add(source)
        context_chunks.append(f"[{source}]\n{doc.page_content.strip()}")

    context = "\n\n".join(context_chunks)

    prompt = (
        f"<|im_start|>system\n"
        f"You are an intelligent assistant. Use the following extracted content to answer the user's question. "
        f"Do not fabricate. If you are unsure, say so clearly. Be concise.\n"
        f"Context:\n{context}\n<|im_end|>\n"
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = text_gen(prompt)[0]['generated_text']
    reply = output.split("<|im_start|>assistant\n")[-1].strip()
    sources_str = ", ".join(sorted(sources))
    history.append((query, f"資料來源：{sources_str}\n\n回覆：\n{reply}"))
    return history, history

if __name__ == "__main__":
    print("請從 app.py 呼叫 generate_reply，不要單獨執行。")