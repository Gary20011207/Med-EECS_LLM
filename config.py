# config.py - 統一配置管理（支援動態模型切換）
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 預設配置檔案路徑
MODEL_CONFIGS_FILE = "./Eval/model_config.json"

# 預設回退配置
DEFAULT_CONFIG = {
    "config_name": "Default",
    "llm_model_name": "Qwen/Qwen2.5-14B-Instruct-1M",
    "embeddings_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "rag_top_k": 5,
    "temperature": 0.1,
    "max_new_tokens": 1000,
    "system_prompt": """您是一個專業的醫療助手，專門協助醫護人員解答關於 ERAS (Enhanced Recovery After Surgery) 手術加速康復計劃的問題。

您的任務是基於提供的 ERAS 指引文件，為醫護人員提供準確、專業的回答。如果問題超出您的知識範圍或提供的文件內容，請誠實告知您無法回答該問題，並建議咨詢專業醫療人員。

請保持回答簡潔明了，並在適當時引用相關指引。"""
}

class ConfigManager:
    """動態配置管理器"""
    
    def __init__(self, initial_config_name: Optional[str] = None):
        self.configs = {}
        self.current_config = None
        self.load_configs(initial_config_name)
    
    def load_configs(self, initial_config_name: Optional[str] = None):
        """載入配置檔案
        
        Args:
            initial_config_name: 指定的初始配置名稱，如果未指定則使用第一個
        """
        try:
            if os.path.exists(MODEL_CONFIGS_FILE):
                with open(MODEL_CONFIGS_FILE, 'r', encoding='utf-8') as f:
                    configs_list = json.load(f)
                    
                # 轉換為字典格式，方便查找
                self.configs = {config["config_name"]: config for config in configs_list}
                
                # 選擇初始配置
                if initial_config_name and initial_config_name in self.configs:
                    self.current_config = self.configs[initial_config_name]
                    logger.info(f"載入指定配置: {initial_config_name}")
                elif self.configs:
                    # 如果沒有指定配置名稱，使用第一個配置作為預設
                    self.current_config = list(self.configs.values())[0]
                    logger.info(f"載入預設配置: {self.current_config['config_name']}")
                else:
                    self.current_config = DEFAULT_CONFIG
                    logger.warning("配置檔案為空，使用內建預設配置")
            else:
                logger.warning(f"配置檔案不存在: {MODEL_CONFIGS_FILE}，使用預設配置")
                self.current_config = DEFAULT_CONFIG
                
        except Exception as e:
            logger.error(f"載入配置檔案時出錯: {e}")
            self.current_config = DEFAULT_CONFIG
    
    def switch_config(self, config_name: str) -> bool:
        """切換配置"""
        if config_name in self.configs:
            self.current_config = self.configs[config_name]
            logger.info(f"切換到配置: {config_name}")
            return True
        else:
            logger.error(f"找不到配置: {config_name}")
            return False
    
    def get_available_configs(self) -> list:
        """獲取可用配置列表"""
        return list(self.configs.keys())
    
    def get_current_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.current_config.copy()
    
    def get_config_value(self, key: str, default=None):
        """獲取特定配置值"""
        return self.current_config.get(key, default)
    
    def reload_configs(self):
        """重新載入配置檔案"""
        old_config_name = self.current_config.get("config_name", "Unknown")
        self.load_configs()
        logger.info(f"配置已重新載入，當前配置: {self.current_config.get('config_name', 'Unknown')}")
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """保存新配置"""
        self.configs[config_name] = config_data
        
        # 更新檔案
        configs_list = list(self.configs.values())
        os.makedirs(os.path.dirname(MODEL_CONFIGS_FILE), exist_ok=True)
        with open(MODEL_CONFIGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(configs_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置 {config_name} 已保存")

# 創建全域配置管理器
config_manager = ConfigManager()

# 模型配置 - 從配置管理器獲取
LLM_MODEL_NAME = config_manager.get_config_value("llm_model_name")
EMBEDDINGS_MODEL_NAME = config_manager.get_config_value("embeddings_model_name")
QUANTIZATION_LEVEL = "4bit"  # 可選：4bit, 8bit, none (暫不在JSON配置中)

# 資料庫配置 - 從配置管理器獲取
PDF_FOLDER = "./PDFS"  # 保持原有設定
DB_PATH = "./VectorDB"   # 保持原有設定
CHUNK_SIZE = config_manager.get_config_value("chunk_size", 1000)
CHUNK_OVERLAP = config_manager.get_config_value("chunk_overlap", 200)
RAG_TOP_K = config_manager.get_config_value("rag_top_k", 5)
INITIAL_DB_FORCE_RESET = False  # 保持原有設定

# LLM 資源管理配置 - 保持原有設定
DEFAULT_INACTIVITY_TIMEOUT = 600  # 秒，10分鐘
MONITOR_CHECK_INTERVAL_SECONDS = 30  # 秒
ENABLE_AUTO_CPU_OFFLOAD = True  # 是否啟用自動 GPU 卸載

# Flask 應用配置 - 保持原有設定
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = False
FLASK_USE_RELOADER = False
REQUEST_LIMIT = "20 per minute"  # API 請求限制
HTML_TEMPLATE_FILE = "chat_test_v2.html"

# 生成配置 - 從配置管理器獲取
DEFAULT_TEMPERATURE = config_manager.get_config_value("temperature", 0.1)
DEFAULT_MAX_NEW_TOKENS = config_manager.get_config_value("max_new_tokens", 1000)
MIN_TEMPERATURE = 0.1  # 最小溫度值
MAX_TEMPERATURE = 2.0  # 最大溫度值
MIN_MAX_NEW_TOKENS = 50  # 最小生成 token 數
MAX_MAX_NEW_TOKENS = 4000  # 最大生成 token 數

# 系統提示詞 - 從配置管理器獲取
SYSTEM_PROMPT = config_manager.get_config_value("system_prompt", DEFAULT_CONFIG["system_prompt"])

# 日誌配置 - 保持原有設定
LOG_FILE = "app_chat.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# 便利函數
def switch_model_config(config_name: str) -> bool:
    """切換模型配置"""
    return config_manager.switch_config(config_name)

def get_available_model_configs() -> list:
    """獲取可用的模型配置"""
    return config_manager.get_available_configs()

def get_current_model_config() -> Dict[str, Any]:
    """獲取當前模型配置"""
    return config_manager.get_current_config()

def reload_model_configs():
    """重新載入配置"""
    config_manager.reload_configs()
    
    # 更新當前變數
    global LLM_MODEL_NAME, EMBEDDINGS_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
    global RAG_TOP_K, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS, SYSTEM_PROMPT
    
    LLM_MODEL_NAME = config_manager.get_config_value("llm_model_name")
    EMBEDDINGS_MODEL_NAME = config_manager.get_config_value("embeddings_model_name")
    CHUNK_SIZE = config_manager.get_config_value("chunk_size", 1000)
    CHUNK_OVERLAP = config_manager.get_config_value("chunk_overlap", 200)
    RAG_TOP_K = config_manager.get_config_value("rag_top_k", 5)
    DEFAULT_TEMPERATURE = config_manager.get_config_value("temperature", 0.1)
    DEFAULT_MAX_NEW_TOKENS = config_manager.get_config_value("max_new_tokens", 1000)
    SYSTEM_PROMPT = config_manager.get_config_value("system_prompt", DEFAULT_CONFIG["system_prompt"])

def save_new_model_config(config_name: str, config_data: Dict[str, Any]):
    """保存新的模型配置"""
    config_manager.save_config(config_name, config_data)