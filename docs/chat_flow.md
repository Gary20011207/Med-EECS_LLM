# ERAS 醫療專案管理系統 - 專案結構與模組功能詳解

## 一、專案目錄結構

```
.
├── app_chat.py                 # Flask Web 應用程式主檔案 (使用者介面與API端點)
├── apps/                       # 後端核心邏輯模組
│   ├── init.py             # 將 'apps' 標示為一個 Python 套件
│   ├── LLM_init.py             # 大型語言模型 (LLM) 和向量資料庫的初始化與管理
│   ├── RAG_build.py            # 從 PDF 文件建立與重建向量資料庫的腳本
│   └── RAG_QA_stream.py        # 實現 RAG (檢索增強生成) 問答與串流回覆邏輯
├── static/                     # (可選) Flask 靜態檔案 (如 CSS, 前端 JavaScript)
│   └── style.css               # (範例，目前 chat_test.html 內嵌 CSS)
├── templates/                  # Flask HTML 前端模板
│   └── chat_test.html          # 提供使用者互動的聊天介面
├── PDFS/                       # 存放 ERAS 指引等來源 PDF 文件的資料夾
├── VectorDB/                   # 存放 ChromaDB 向量資料庫檔案的目錄
├── app_chat.log                # Flask 應用程式運行的日誌檔案
├── requirements.txt            # 專案所需的 Python 套件依賴列表
└── README.md                   # 專案說明文件
```
---

## 二、各 Python 程式檔案功能詳解

### 1. `apps/RAG_build.py` 

* **主要功能**:
    此腳本負責將 `PDFS/` 目錄下的 PDF 文件轉換為知識庫。它可以處理該目錄下的所有 PDF，或根據提供的檔案列表選擇性地處理特定的 PDF。它會讀取 PDF 內容，將文本分割成小塊 (chunks)，生成這些文本塊的向量嵌入 (embeddings)，最後將這些嵌入和文本塊存儲到 `VectorDB/` 目錄下的 Chroma 向量資料庫中。這個向量資料庫是後續 RAG (檢索增強生成) 系統進行資訊檢索的基礎。

* **被呼叫時的用法**:
    * **獨立運行**: 使用者可以直接在終端執行 `python apps/RAG_build.py`。腳本的 `if __name__ == "__main__":` 區塊會演示如何處理所有 PDF 或指定的 PDF 子集，並觸發資料庫的清空與重建（或更新）。這通常在首次建立知識庫、需要更新 PDF 內容或僅針對部分文件重建索引時手動進行。
    * **被 `LLM_init.py` 呼叫**: 在 `LLM_init.py` 的 `initialize_vector_database()` 函數中，如果偵測到向量資料庫不存在或為空，且 `RAG_BUILD_AVAILABLE` 為 `True`，則會自動呼叫本模組的 `reset_and_rebuild_vectordb` 函數（通常不帶 `pdf_filenames` 參數，即處理所有 PDF）來嘗試建立資料庫。

* **主要函數**:
    * `reset_and_rebuild_vectordb(pdf_folder, db_path, emb_model, chunk_size, chunk_overlap, force_reset, pdf_filenames: Optional[List[str]] = None)`:
        * 執行 PDF 處理、嵌入生成和資料庫建立/更新的核心邏輯。
        * 若提供了 `pdf_filenames` 列表，則僅處理列表中指定的、且存在於 `pdf_folder` 中的 PDF 檔案。
        * 若 `pdf_filenames` 為 `None` 或空，則處理 `pdf_folder` 中的所有 PDF 檔案。
        * `force_reset=True` 時，會在處理選定文件前清空整個向量資料庫。若 `force_reset=False` 且資料庫已存在，則選定文件處理後的新數據會被添加到現有資料庫中（注意：ChromaDB 對於重複 ID 的處理方式可能需要考量，通常是更新或忽略）。

* **全域/主要參數設定** (定義於 `RAG_build.py` 模組頂部，並作為 `reset_and_rebuild_vectordb` 函數參數的預設值):
    * `DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"`: 指定用於生成文本嵌入的 Hugging Face Sentence Transformer 模型名稱。**此設定必須與 `LLM_init.py` 中 `EMBEDDINGS_MODEL_NAME` 一致。**
    * `DEFAULT_CHUNK_SIZE: int = 1000`: 文本分割時每個文本塊的目標大小 (字元數)。
    * `DEFAULT_CHUNK_OVERLAP: int = 200`: 文本分割時相鄰文本塊之間的重疊字元數。
    * `DEFAULT_PDF_FOLDER: str = "./PDFS"`: 指定包含來源 PDF 文件的預設資料夾路徑。
    * `DEFAULT_DB_PATH: str = "./VectorDB"`: 指定儲存 Chroma 向量資料庫的預設路徑。

---

### 2. `apps/LLM_init.py`

* **主要功能**:
    此模組是大型語言模型 (LLM) 和向量資料庫的中央管理器。它負責：
    1.  初始化並載入指定的 LLM (例如 Qwen) 及其分詞器，支援4位元量化以節省資源。
    2.  連接到由 `RAG_build.py` 建立的向量資料庫。
    3.  提供獲取 LLM 和向量資料庫實例的函數給其他模組使用。
    4.  實現一個背景監控線程，用於在 LLM 長時間閒置時自動將其從 GPU 移至 CPU，以釋放 GPU 資源。
    5.  提供資源清理函數，在應用程式關閉時安全釋放模型和相關資源。

* **被呼叫時的用法**:
    * **被 `app_chat.py` (Flask應用) 呼叫**:
        * 在 Flask 應用啟動時，`perform_initialization()` 函數會呼叫本模組的 `initialize_llm_model(force_cpu_init=True)` 和 `initialize_vector_database()` 來預先載入模型到 CPU 並準備好向量資料庫。
        * 當處理聊天請求時 (`/api/chat`, `/api/chat/stream`)，會呼叫 `get_llm_model_and_tokenizer()` 來獲取 LLM 實例（此時模型可能會被移至 GPU）和分詞器。
        * `/api/status` 端點會呼叫 `get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)` 以被動模式檢查模型狀態，不影響閒置計時。
        * 在 Flask 應用關閉時，`perform_cleanup()` 函數會呼叫本模組的 `shutdown_llm_resources()`。
    * **被 `RAG_QA_stream.py` 呼叫**:
        * 在執行問答邏輯前，會呼叫 `get_llm_model_and_tokenizer()` 來獲取 LLM 和分詞器。
        * 在進行 RAG 檢索前，會呼叫 `get_vector_db_instance()` 來獲取向量資料庫實例。
        * `check_gpu_status()`、`build_memory()`、`_build_prompt()` 內部為了獲取模型配置或狀態，會以被動模式 (`update_last_used_time=False, ensure_on_gpu=False`) 呼叫 `get_llm_model_and_tokenizer()`。
    * **獨立運行**: `if __name__ == "__main__":` 區塊提供了完整的測試流程，包括初始化、模型在 CPU/GPU 間的移動、閒置轉移測試以及最終的資源釋放。

* **主要函數**:
    * `initialize_llm_model(force_cpu_init)`: 初始化 LLM 和分詞器。
    * `initialize_vector_database()`: 初始化向量資料庫連接，並在需要時嘗試自動重建。
    * `get_llm_model_and_tokenizer(update_last_used_time, ensure_on_gpu)`: 獲取 LLM 和分詞器實例，管理其設備位置和使用時間。
    * `get_vector_db_instance()`: 獲取向量資料庫實例。
    * `shutdown_llm_resources()`: 釋放所有相關資源。
    * `count_tokens(text)`: 計算文本的 token 數量。
    * `_ensure_device_monitor_started()` / `_monitor_model_device_activity()`: 管理閒置模型從 GPU 移至 CPU 的背景線程。

* **全域/主要參數設定**:
    * `LLM_MODEL_NAME: str = "Qwen/Qwen2.5-14B-Instruct-1M"`: 指定要使用的大型語言模型名稱 (來自 Hugging Face Hub)。
    * `EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"`: 指定嵌入模型的名稱，用於查詢向量資料庫。**必須與 `RAG_build.py` 中的 `DEFAULT_EMBEDDING_MODEL` 設定一致。**
    * `DB_PATH: str = "./VectorDB"`: 向量資料庫的儲存路徑。
    * `PDF_FILES_PATH: str = "./PDFS"`: PDF 文件的來源路徑，供 `initialize_vector_database` 在需要自動重建資料庫時傳遞給 `RAG_build.py`。
    * `DEFAULT_INACTIVITY_TIMEOUT: int = 600` (秒): LLM 模型在 GPU 上閒置超過此時間後，會被嘗試移至 CPU (預設10分鐘)。
    * `MONITOR_CHECK_INTERVAL_SECONDS: int = 30` (秒): 背景監控線程檢查模型閒置狀態的頻率。

---

### 3. `apps/RAG_QA_stream.py`

* **主要功能**:
    此模組是問答系統的核心引擎。它整合了 RAG 檢索、歷史對話記憶管理以及與 LLM 的互動，以生成針對使用者問題的回覆。主要功能包括：
    1.  根據使用者查詢，從向量資料庫中檢索相關的上下文資訊 (RAG)。
    2.  管理和構建對話歷史，以便 LLM 能夠理解對話的上下文。
    3.  將系統提示 (System Prompt)、檢索到的上下文、對話歷史和當前使用者問題組合成一個完整的提示詞 (Prompt) 交給 LLM。
    4.  提供兩種回覆模式：
        * `generate_reply()`: 一次性生成完整的答案 (非串流)。
        * `stream_response()`: 以串流方式逐步生成並返回答案片段，適用於即時聊天介面。

* **被呼叫時的用法**:
    * **被 `app_chat.py` (Flask應用) 呼叫**:
        * `/api/chat` 端點 (非串流) 會呼叫本模組的 `generate_reply()` 函數。
        * `/api/chat/stream` 端點 (串流) 會呼叫本模組的 `stream_response()` 函數，並將其返回的生成器用於 Server-Sent Events (SSE)。
    * **獨立運行**: `if __name__ == "__main__":` 區塊提供測試功能，會先初始化 `LLM_init.py` 中的資源，然後執行本模組的非串流和串流問答測試。

* **主要函數**:
    * `build_memory(hist, enable, base_tok, reserve_for_context_and_query)`: 構建歷史對話字串。
    * `_build_prompt(query, sel_pdf, enable_mem, hist, max_new_tokens_for_reply)`: 構建完整的 LLM 提示詞。
    * `check_gpu_status()`: 檢查並打印當前 LLM 的 GPU 狀態 (以被動模式呼叫 `LLM_init` 中的函數)。
    * `generate_reply(...)`: 生成非串流的完整回覆。
    * `stream_response(...)`: 以串流方式生成回覆。

* **全域/主要參數設定**:
    * `SYSTEM_PROMPT: str`: 定義 AI 助手角色的系統級提示詞。
    * `RAG_TOP_K: int = 5`: RAG 檢索時，從向量資料庫返回最相關的 k 個文檔片段。

---

### 4. `app_chat.py`

* **主要功能**:
    這是基於 Flask 的 Web 應用程式入口。它提供了使用者與後端 RAG 問答系統互動的介面。主要職責包括：
    1.  提供一個網頁前端 (`chat_test.html`)，讓使用者可以輸入問題、選擇參考 PDF、管理對話等。
    2.  定義 API 端點 (endpoints) 來處理前端的請求：
        * `/`: 服務主聊天頁面。
        * `/api/status`: 提供系統和 LLM 模型狀態。
        * `/api/chat`: 處理非串流的聊天請求。
        * `/api/chat/stream`: 處理串流的聊天請求 (使用 SSE)。
        * `/api/pdfs`: 獲取 `PDFS/` 目錄下可用的 PDF 文件列表。
        * `/api/chat/history/clear`: 清除指定會話的聊天歷史。
    3.  在應用程式啟動時，調用 `LLM_init.py` 中的函數來初始化 LLM 和向量資料庫。
    4.  在應用程式關閉時，調用 `LLM_init.py` 中的函數來釋放資源。
    5.  管理簡單的會話歷史 (儲存在記憶體中，適用於測試)。
    6.  配置日誌記錄。

* **執行方式**:
    * 可以直接在終端執行 `python app_chat.py` 來啟動 Flask 開發伺服器。
    * 也可以使用 `nohup python -u app_chat.py > /dev/null 2>&1 &` 命令使其在背景常駐運行。

* **主要函數/路由**:
    * `index()`: 渲染 `chat_test.html`。
    * `get_app_status()`: 返回系統狀態 JSON。
    * `handle_chat_request()`: 處理非串流聊天，呼叫 `RAG_QA_stream.generate_reply()`。
    * `handle_stream_chat_request()`: 處理串流聊天，呼叫 `RAG_QA_stream.stream_response()` 並使用 SSE。
    * `get_available_pdfs()`: 返回 PDF 列表。
    * `clear_session_history()`: 清除聊天歷史。
    * `perform_initialization()`: 應用啟動時的資源初始化。
    * `perform_cleanup()`: 應用關閉時的資源清理。

* **全域/主要參數設定** (主要體現在 Flask 設定和一些內部邏輯中):
    * **Flask 相關**:
        * `host='0.0.0.0'`: 使應用可以從外部網路訪問。
        * `port=5001`: 應用運行的埠號。
        * `debug=True/False`: 是否啟用 Flask 的調試模式 (影響自動重載和交互式調試器)。
    * **應用邏輯相關**:
        * `chat_sessions_history = {}`: 一個字典，用於在記憶體中存儲不同 session ID 的對話歷史。在生產環境中，應替換為更持久的儲存方案（如資料庫或 Redis）。
        * 日誌檔案路徑: `'app_chat.log'`。
