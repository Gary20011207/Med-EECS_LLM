# tests/test_model.py
import os
import sys
import time
import logging

# 添加專案根目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 先定義 logger
logger = logging.getLogger(__name__)

# 導入配置 (比照正式使用情境)
try:
    from config import (
        LLM_MODEL_NAME,
        DEFAULT_INACTIVITY_TIMEOUT,
        MONITOR_CHECK_INTERVAL_SECONDS,
        LOG_FILE,
        LOG_LEVEL
    )
    logger.info("成功導入 config.py 中的配置")
except ImportError as e:
    logger.warning(f"無法導入 config.py: {e}，將使用預設配置")
    # 如果無法導入 config，設定預設值
    LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
    DEFAULT_INACTIVITY_TIMEOUT = 600
    MONITOR_CHECK_INTERVAL_SECONDS = 30
    LOG_FILE = "test_model.log"
    LOG_LEVEL = "INFO"

# 導入模組
from core.model_manager import ModelManager

if __name__ == "__main__":
    # 設定日誌 (使用 config 中的設定)
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # 設定檔案日誌和控制台日誌
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info(" ModelManager 腳本獨立運行測試 (使用正式配置)")
    logger.info("="*60)
    logger.info(f"使用的配置:")
    logger.info(f"  LLM 模型: {LLM_MODEL_NAME}")
    logger.info(f"  預設閒置超時: {DEFAULT_INACTIVITY_TIMEOUT} 秒")
    logger.info(f"  監控檢查間隔: {MONITOR_CHECK_INTERVAL_SECONDS} 秒")
    logger.info(f"  日誌檔案: {LOG_FILE}")
    logger.info(f"  日誌級別: {LOG_LEVEL}")
    
    # 創建 ModelManager 實例
    model_manager = ModelManager()
    
    # 設定測試模式下的較短超時時間
    TEST_INACTIVITY_TIMEOUT_SECONDS = 15
    model_manager.inactivity_timeout = TEST_INACTIVITY_TIMEOUT_SECONDS
    logger.info(f"測試模式：設定閒置超時為 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒")
    
    try:
        logger.info("\n" + "="*50)
        logger.info("[步驟 1] 初始化LLM模型到CPU...")
        logger.info("="*50)
        model_manager.initialize(force_cpu_init=True)
        
        status = model_manager.get_status()
        logger.info("模型狀態:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "="*50)
        logger.info("[步驟 2] 模擬對話啟動 (模型移至GPU)...")
        logger.info("="*50)
        model, tokenizer, _ = model_manager.get_model_and_tokenizer(
            ensure_on_gpu=True, 
            update_last_used_time=True
        )
        
        if model and tokenizer:
            current_device = next(model.parameters()).device
            logger.info(f"模型目前設備: {current_device}")
            
            if current_device.type == "cuda":
                logger.info(f"✓ 模型已移至GPU。預計閒置 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒後自動移回CPU")
                
                # 執行簡單的生成測試
                logger.info("執行測試生成...")
                prompt = "你好"
                inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
                
                import torch
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10)
                    
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"測試生成結果: {generated_text}")
                
                # 測試 token 計算功能
                logger.info("測試 token 計算功能...")
                test_text = "這是一個測試文本"
                token_count = model_manager.count_tokens(test_text)
                logger.info(f"文本 '{test_text}' 的 token 數: {token_count}")
                
                # 等待觀察自動 CPU 轉移
                wait_duration = MONITOR_CHECK_INTERVAL_SECONDS + TEST_INACTIVITY_TIMEOUT_SECONDS + 5
                logger.info(f"\n" + "="*50)
                logger.info(f"[步驟 3] 等待 {wait_duration} 秒觀察模型是否自動移回CPU...")
                logger.info("="*50)
                logger.info("監控線程應該會在模型閒置時自動將其移回CPU...")
                
                # 分段等待，每10秒報告一次
                for i in range(0, wait_duration, 10):
                    remaining = wait_duration - i
                    logger.info(f"等待中... 還剩 {remaining} 秒")
                    time.sleep(min(10, remaining))
                
                # 檢查模型是否已移回 CPU
                model_after_wait, _, _ = model_manager.get_model_and_tokenizer(
                    update_last_used_time=False, 
                    ensure_on_gpu=False
                )
                
                if model_after_wait:
                    final_device = next(model_after_wait.parameters()).device
                    logger.info(f"等待後，模型目前設備: {final_device}")
                    if final_device.type == "cpu":
                        logger.info("✓ 模型已成功自動移回CPU")
                    else:
                        logger.warning(f"⚠ 模型未按預期移回CPU (仍在 {final_device})")
                        logger.warning("可能的原因:")
                        logger.warning("  1. 監控線程尚未執行")
                        logger.warning("  2. 閒置時間未達到閾值")
                        logger.warning("  3. 設備移動過程中發生錯誤")
                else:
                    logger.error("✗ 等待後，模型實例為 None")
            else:
                logger.info(f"\n[步驟 3] 模型在設備 {current_device}，跳過 GPU 相關測試")
                if current_device.type == "cpu":
                    logger.info("✓ 模型正確在 CPU 上")
                else:
                    logger.warning(f"⚠ 模型在未知設備: {current_device}")
        else:
            logger.error("✗ [步驟 2] 獲取模型和分詞器失敗")
    
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}", exc_info=True)
    
    finally:
        logger.info(f"\n" + "="*50)
        logger.info("[步驟 4] 釋放所有資源...")
        logger.info("="*50)
        model_manager.shutdown()
        logger.info("資源釋放完成")
        logger.info("="*60)
        logger.info(" ModelManager 腳本測試完畢")
        logger.info("="*60)