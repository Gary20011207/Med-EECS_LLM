# utils/logging_setup.py
import logging
import sys
import os

# 嘗試從 config 導入日誌配置，如果失敗則使用預設值
try:
    # 假設 config.py 在專案根目錄
    # 這需要將 eras_medical_system 的父目錄添加到 sys.path，或者在運行 app.py 時 eras_medical_system 是當前工作目錄
    # 或者 config.py 與 utils 在同一級別 (這與提供的結構不符)
    # 為了簡化，我們假設 app.py 會處理路徑問題，或者 config 在 PYTHONPATH 中
    from config import LOG_FILE, LOG_LEVEL
except ImportError:
    print("警告: 無法從 config.py 導入日誌配置。將使用預設日誌設定。")
    LOG_FILE = "app_chat.log"  # 預設日誌檔案名稱
    LOG_LEVEL = "INFO"       # 預設日誌級別

def setup_logging():
    """
    設定應用程式的日誌記錄。
    """
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    selected_log_level = log_level_map.get(LOG_LEVEL.upper(), logging.INFO)
    
    # 確保日誌目錄存在 (如果 LOG_FILE 包含路徑)
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"錯誤: 無法創建日誌目錄 {log_dir}: {e}")
            # 如果無法創建目錄，則將日誌輸出到控制台
            logging.basicConfig(
                level=selected_log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            logging.error(f"日誌目錄創建失敗，日誌將僅輸出到控制台。")
            return

    # 設定日誌記錄器
    logging.basicConfig(
        level=selected_log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(pathname)s:%(lineno)d)",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'), # 輸出到檔案
            logging.StreamHandler(sys.stdout)                 # 輸出到控制台
        ]
    )
    
    # 設定 requests 和 urllib3 的日誌級別為 WARNING，以減少冗餘日誌
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING) # 如果使用 httpx
    logging.getLogger("huggingface_hub").setLevel(logging.INFO) # huggingface_hub 的 INFO 級別通常包含有用的下載信息

    logger = logging.getLogger(__name__)
    logger.info(f"日誌系統已設定。日誌級別: {LOG_LEVEL}, 日誌檔案: {LOG_FILE}")

if __name__ == '__main__':
    # 測試日誌設定
    setup_logging()
    logging.debug("這是一條調試日誌。")
    logging.info("這是一條資訊日誌。")
    logging.warning("這是一條警告日誌。")
    logging.error("這是一條錯誤日誌。")
    logging.critical("這是一條嚴重錯誤日誌。")
