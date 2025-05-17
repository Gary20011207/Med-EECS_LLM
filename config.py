# config.py

# --- ModelManager Defaults ---
DEFAULT_LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M" #"microsoft/Phi-4-mini-reasoning" # 原來的預設
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 600  # 閒置超時時間 (秒)
DEFAULT_MONITOR_CHECK_INTERVAL_SECONDS = 60 # 監控執行緒檢查間隔 (秒)
DEFAULT_LOAD_IN_4BIT = True # 是否預設以 4-bit 量化載入
DEFAULT_FORCE_CPU_INIT = False # 是否強制在 CPU 初始化

# --- RAGEngine Defaults ---
DEFAULT_SYSTEM_PROMPT = """您是一個專業的醫療助手，專門協助醫護人員解答關於 ERAS (Enhanced Recovery After Surgery) 手術加速康復計劃的問題。

您的任務是基於提供的 ERAS 指引文件，為醫護人員提供準確、專業的回答。如果問題超出您的知識範圍或提供的文件內容，請誠實告知您無法回答該問題，並建議咨詢專業醫療人員。

請保持回答簡潔明了，並在適當時引用相關指引。"""
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_NEW_TOKENS = 1000
DEFAULT_MIN_TEMPERATURE = 0.0 # 允許 0 代表 greedy search
DEFAULT_MAX_TEMPERATURE = 2.0
DEFAULT_MIN_MAX_NEW_TOKENS = 50
DEFAULT_MAX_MAX_NEW_TOKENS = 4096 # 根據常見模型調整

# --- DBManager Defaults ---
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-small-zh-v1.5"  # 或者您選擇的其他模型
DEFAULT_VECTOR_STORE_PATH = "./VectorDB"         # 預設的向量資料庫儲存路徑
DEFAULT_DOCUMENTS_PATH = "./PDFS"  # 預設PDF路徑
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_K_SEARCH_RESULTS = 5                            # 搜尋時預設返回的結果數量

# --- Log Level ---
DEFAULT_LOG_LEVEL = "INFO" # 例如: "DEBUG", "INFO", "WARNING", "ERROR"

# --- Web App (If any) ---
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 5001
DEFAULT_WEB_DEBUG = False