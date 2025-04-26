import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# =======================
# Configuration
# =======================
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FOLDER = "./PDFS"
DB_PATH = "./VectorDB"

# =======================
# 4-bit Quantized LLM Init
# =======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=256,
    temperature=0.1,
)

# =======================
# Create Vector Database
# =======================
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
# Reply Logic: Support No PDFs / All PDFs / One PDF
# =======================
def generate_reply(query, selected_pdf, history):
    if selected_pdf == "No PDFs":
        prompt = (
            f"<|im_start|>user\n{query}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        output = text_gen(prompt)[0]['generated_text']
        reply = output.split("<|im_start|>assistant\n")[-1].strip()
        history.append((query, f"回覆（模型內部知識）:\n{reply}"))
        return history, history

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    all_docs = retriever.get_relevant_documents(query)

    if selected_pdf != "All PDFs":
        docs = [doc for doc in all_docs if doc.metadata.get("source_pdf") == selected_pdf]
    else:
        docs = all_docs

    if not docs:
        reply = f"在所選的文件中找不到相關資訊！"
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

# =======================
# Gradio Interface
# =======================
def launch_gradio():
    all_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    pdf_options = ["No PDFs", "All PDFs"] + sorted(all_pdfs)

    with gr.Blocks(css="footer {visibility: hidden}; .gr-button {min-height: 42px;}") as demo:
        gr.Markdown("# 醫學電資整合創意專題 Multi-Document RAG Chatbot (Naive Version)")

        chatbot = gr.Chatbot(label="Chat History", show_copy_button=True)

        selected_pdf = gr.Dropdown(choices=pdf_options, value="No PDFs", label="選擇文件範圍")

        with gr.Row(equal_height=True):
            msg = gr.Textbox(label="輸入問題", placeholder="請輸入問題，例如：這份文件內容為何？", lines=2)
            submit = gr.Button("送出", variant="primary")

        clear = gr.Button("清除對話")
        state = gr.State([])

        submit.click(fn=generate_reply, inputs=[msg, selected_pdf, state], outputs=[chatbot, state])
        msg.submit(fn=generate_reply, inputs=[msg, selected_pdf, state], outputs=[chatbot, state])
        clear.click(lambda: ([], []), outputs=[chatbot, state])

        # Download PDF files
        gr.Markdown("下載 PDF 文件：")
        for f in sorted(os.listdir(PDF_FOLDER)):
            if f.endswith(".pdf"):
                file_path = os.path.join(PDF_FOLDER, f)
                gr.File(value=file_path, file_types=[".pdf"], label=f"📄 {f}", interactive=True)

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()