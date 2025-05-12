# config.py - 統一配置管理

# 模型配置
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUANTIZATION_LEVEL = "4bit"  # 可選：4bit, 8bit, none

# 資料庫配置
PDF_FOLDER = "./PDFS"
DB_PATH = "./VectorDB"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RAG_TOP_K = 5
INITIAL_DB_FORCE_RESET = False

# LLM 資源管理配置
DEFAULT_INACTIVITY_TIMEOUT = 600  # 秒，10分鐘
MONITOR_CHECK_INTERVAL_SECONDS = 30  # 秒
ENABLE_AUTO_CPU_OFFLOAD = True  # 是否啟用自動 GPU 卸載

# Flask 應用配置
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = False
FLASK_USE_RELOADER = False
REQUEST_LIMIT = "20 per minute"  # API 請求限制
HTML_TEMPLATE_FILE = "chat_test_v2.html"

# 生成配置
DEFAULT_TEMPERATURE = 0.1  # 預設溫度參數
DEFAULT_MAX_NEW_TOKENS = 1000  # 預設最大新生成 token 數
MIN_TEMPERATURE = 0.1  # 最小溫度值
MAX_TEMPERATURE = 2.0  # 最大溫度值
MIN_MAX_NEW_TOKENS = 50  # 最小生成 token 數
MAX_MAX_NEW_TOKENS = 4000  # 最大生成 token 數

# 系統提示詞
SYSTEM_PROMPT = """您是一個專業的醫療助手，專門協助醫護人員解答關於 ERAS (Enhanced Recovery After Surgery) 手術加速康復計劃的問題。

您的任務是基於提供的 ERAS 指引文件，為醫護人員提供準確、專業的回答。如果問題超出您的知識範圍或提供的文件內容，請誠實告知您無法回答該問題，並建議咨詢專業醫療人員。

請保持回答簡潔明了，並在適當時引用相關指引。"""

# 日誌配置
LOG_FILE = "app_chat.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL