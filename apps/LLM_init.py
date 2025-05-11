# apps/LLM_init.py
import os
import time
import threading
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
import traceback
from typing import Optional, Any 
import sys 

# --- 基本設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 組態設定 ---
LLM_MODEL_NAME: str = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2" 

DB_PATH: str = "./VectorDB"
PDF_FILES_PATH: str = "./PDFS" 

DEFAULT_INACTIVITY_TIMEOUT: int = 600 
MONITOR_CHECK_INTERVAL_SECONDS: int = 30 

# --- 全域變數 ---
_llm_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_model_max_context_length: Optional[int] = None
_embedding_function: Optional[HuggingFaceEmbeddings] = None
_vector_db: Optional[Chroma] = None 
_last_model_use_time: float = 0.0
_model_management_lock: threading.RLock = threading.RLock()
_device_monitor_thread: Optional[threading.Thread] = None
_current_inactivity_timeout: int = DEFAULT_INACTIVITY_TIMEOUT


# =======================
# 設備管理 (Device Management)
# =======================
def _ensure_device_monitor_started():
    global _device_monitor_thread
    with _model_management_lock:
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
            return

        def _monitor_model_device_activity():
            global _llm_model 
            logger.info("模型設備活動監控線程已啟動。")
            while True:
                try:
                    time.sleep(MONITOR_CHECK_INTERVAL_SECONDS)
                    with _model_management_lock:
                        if _llm_model is None:
                            continue
                        
                        model_params = None
                        try:
                            model_params = list(_llm_model.parameters())
                        except Exception as e_get_params:
                            logger.error(f"監控線程：調用 _llm_model.parameters() 時出錯: {e_get_params}", exc_info=True)
                            continue 

                        if not model_params:
                            logger.warning("監控線程：_llm_model 沒有參數。跳過設備檢查。")
                            continue
                            
                        current_model_device_type = model_params[0].device.type

                        if current_model_device_type == "cuda":
                            current_time = time.time()
                            if (current_time - _last_model_use_time > _current_inactivity_timeout):
                                logger.info(f"模型閒置超過 {_current_inactivity_timeout} 秒，準備移至CPU...")
                                try:
                                    _llm_model = _llm_model.to("cpu") 
                                    torch.cuda.empty_cache()
                                    logger.info("模型已成功移至CPU，GPU記憶體已釋放。")
                                except Exception as e_to_cpu:
                                    logger.error(f"監控線程：將模型移至CPU時發生錯誤: {e_to_cpu}", exc_info=True)
                except Exception as e_thread_loop:
                    logger.error(f"模型設備監控線程主循環發生未預期錯誤: {e_thread_loop}", exc_info=True)
                    time.sleep(MONITOR_CHECK_INTERVAL_SECONDS * 2) 
        
        _device_monitor_thread = threading.Thread(target=_monitor_model_device_activity, daemon=True)
        _device_monitor_thread.start()
        logger.info(f"設備監控線程已建立並啟動 (檢查間隔: {MONITOR_CHECK_INTERVAL_SECONDS}s, 閒置超時: {_current_inactivity_timeout}s)。")

# ===========================
# 初始化函數 (Initialization Functions)
# ===========================
def initialize_llm_model(force_cpu_init=False):
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time
    with _model_management_lock:
        if _llm_model is not None and _tokenizer is not None:
            # logger.info("LLM 模型和分詞器已載入。") # 減少重複日誌
            return _llm_model, _tokenizer, _model_max_context_length
        
        logger.info(f"開始初始化 LLM 模型: {LLM_MODEL_NAME} (強制CPU: {force_cpu_init})...")
        # 如果 force_cpu_init 為 True，則目標設備為 "cpu"
        # 否則，如果 CUDA 可用，則為 "auto" (讓 transformers 決定，通常是 GPU)
        # 如果 CUDA 不可用，則也為 "cpu"
        target_device = "cpu" if force_cpu_init or not torch.cuda.is_available() else "auto"
        logger.info(f"LLM 初始化目標設備: {target_device} (CUDA可用: {torch.cuda.is_available()})")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, quantization_config=quantization_config,
                device_map=target_device, trust_remote_code=True
            ).eval()
            
            _model_max_context_length = getattr(_tokenizer, 'model_max_length', 2048) 
            _last_model_use_time = time.time() 
            loaded_device = next(_llm_model.parameters()).device
            logger.info(f"LLM 模型 ({LLM_MODEL_NAME}) 初始化完成。實際載入於: {loaded_device}, 上下文長度: {_model_max_context_length}")
        except Exception as e:
            logger.critical(f"初始化LLM模型 {LLM_MODEL_NAME} 失敗: {e}", exc_info=True)
            _llm_model, _tokenizer = None, None
            raise
        _ensure_device_monitor_started()
        return _llm_model, _tokenizer, _model_max_context_length

# ===========================
# 存取函數 (Access Functions)
# ===========================
def get_llm_model_and_tokenizer(
    update_last_used_time: bool = True, 
    ensure_on_gpu: bool = True         
):
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time
    with _model_management_lock:
        if _llm_model is None or _tokenizer is None:
            logger.info("get_llm_model_and_tokenizer: 模型或分詞器未初始化，將進行首次CPU初始化...")
            # **核心修改**: 首次隱式初始化時，總是強制到 CPU
            initialize_llm_model(force_cpu_init=True) 
            if _llm_model is None: # 確保初始化成功
                 logger.error("get_llm_model_and_tokenizer: 初始化後模型仍為 None。")
                 raise RuntimeError("LLM 模型初始化失敗。")
        
        # 在模型已初始化到 CPU 後，再根據 ensure_on_gpu 決定是否移至 GPU
        if ensure_on_gpu and torch.cuda.is_available():
            try:
                model_device_type = next(_llm_model.parameters()).device.type
                if model_device_type == "cpu":
                    logger.info("get_llm_model_and_tokenizer: (ensure_on_gpu=True) CUDA可用，模型在CPU，移至GPU...")
                    _llm_model = _llm_model.to("cuda")
                    logger.info("模型已成功移至GPU。")
            except Exception as e:
                logger.error(f"get_llm_model_and_tokenizer: 嘗試將模型移至GPU時出錯: {e}", exc_info=True)
        
        if update_last_used_time:
            _last_model_use_time = time.time()
            
        return _llm_model, _tokenizer, _model_max_context_length

def get_vector_db(use_deprecated_name=False) -> Optional[Chroma]: 
    """
    獲取向量資料庫實例。此函數現在只負責連接到預期已存在的資料庫。
    """
    global _vector_db, _embedding_function
    with _model_management_lock:
        if _vector_db is not None:
            return _vector_db
        
        logger.info(f"嘗試連接到向量資料庫: {DB_PATH}")
        if not os.path.exists(DB_PATH) or not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
            logger.error(f"向量資料庫路徑 '{DB_PATH}' 不存在、不是目錄或為空。請先運行 RAG_build.py 建立資料庫。")
            return None 

        try:
            if _embedding_function is None:
                logger.info(f"初始化嵌入函數: {EMBEDDINGS_MODEL_NAME}")
                _embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
            
            _vector_db = Chroma(
                embedding_function=_embedding_function,
                persist_directory=DB_PATH
            )
            db_count = _vector_db._collection.count() if _vector_db and hasattr(_vector_db, '_collection') and _vector_db._collection else 'N/A'
            logger.info(f"成功連接到向量資料庫: {DB_PATH} (記錄數: {db_count})")
            return _vector_db
        except Exception as e:
            logger.error(f"連接到向量資料庫 {DB_PATH} 失敗: {e}", exc_info=True)
            _vector_db = None 
            return None

def reconnect_vector_db():
    global _vector_db
    with _model_management_lock:
        if _vector_db is not None:
            logger.info(f"準備重置向量資料庫連接。")
            _vector_db = None 
        logger.info("向量資料庫連接已標記為重置。下次獲取時將嘗試重新連接。")

# ===========================
# 清理函數 (Cleanup Function)
# ===========================
def shutdown_llm_resources():
    global _llm_model, _tokenizer, _vector_db, _embedding_function, _device_monitor_thread
    global _model_max_context_length, _last_model_use_time
    with _model_management_lock:
        logger.info("開始釋放 LLM 及相關資源...")
        if _llm_model is not None:
            try:
                if next(_llm_model.parameters()).device.type != "cpu":
                    logger.info("將 LLM 模型移至 CPU...")
                    _llm_model = _llm_model.to("cpu")
            except Exception as e: logger.warning(f"關閉前將模型移至CPU失敗: {e}")
            del _llm_model; _llm_model = None; logger.info("LLM 模型已卸載。")
        
        if _tokenizer is not None: del _tokenizer; _tokenizer = None; logger.info("分詞器已卸載。")
        if _vector_db is not None: _vector_db = None; logger.info("向量資料庫實例引用已釋放。")
        if _embedding_function is not None: _embedding_function = None; logger.info("嵌入函數實例引用已釋放。")
            
        _model_max_context_length = None
        _last_model_use_time = 0
            
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
             logger.info("設備監控線程將隨主程序退出。")
        _device_monitor_thread = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache(); logger.info("GPU 快取已清理。")
        logger.info("LLM 相關資源釋放完畢。")

# ===========================
# 輔助函數 (Utility Function)
# ===========================
def count_tokens(text_to_count: str) -> int:
    try:
        with _model_management_lock:
            if _tokenizer is None:
                logger.warning("count_tokens: 分詞器尚未初始化，嘗試被動獲取...")
                _, temp_tokenizer, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
                if temp_tokenizer is None: 
                    logger.error("count_tokens: 仍無法獲取分詞器。")
                    return 0
                tokenizer_instance = temp_tokenizer
            else:
                tokenizer_instance = _tokenizer
        return len(tokenizer_instance.encode(text_to_count, add_special_tokens=False))
    except Exception as e:
        logger.error(f"count_tokens: 計算 token 時出錯: {e}", exc_info=True)
        return -1

# ===========================
# 主執行區塊 (Standalone Run)
# ===========================
if __name__ == "__main__":
    logger.info("="*30 + " LLM_init.py 腳本獨立運行測試 (啟動修正版) " + "="*30)
    
    try:
        # 確保 Python 路徑正確以導入 RAG_build
        if os.path.basename(os.getcwd()) == 'apps': 
            parent_dir = os.path.dirname(os.getcwd())
            if parent_dir not in sys.path: sys.path.insert(0, parent_dir) 
        elif os.path.basename(os.getcwd()) != os.path.basename(os.path.dirname(os.path.abspath(__file__))):
            script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if script_parent_dir not in sys.path: sys.path.insert(0, script_parent_dir)

        from apps.RAG_build import reset_and_rebuild_vectordb as rag_builder_func
        
        logger.info("\n[步驟 0] 執行 RAG_build.py 以確保向量資料庫存在...")
        db_instance_from_rag = rag_builder_func(
            pdf_folder=PDF_FILES_PATH, db_path=DB_PATH,
            emb_model=EMBEDDINGS_MODEL_NAME, force_reset=True, pdf_filenames=None )
        if db_instance_from_rag: logger.info(f"RAG_build.py 成功執行，DB '{DB_PATH}' 已建立/更新。")
        else: logger.error("RAG_build.py 執行失敗。")
    except ImportError as e_imp:
        logger.error(f"無法導入 apps.RAG_build: {e_imp}。跳過資料庫自動建立。")
    except Exception as e_rag_build:
        logger.error(f"執行 RAG_build 時發生錯誤: {e_rag_build}", exc_info=True)

    TEST_INACTIVITY_TIMEOUT_SECONDS = 15
    _current_inactivity_timeout = TEST_INACTIVITY_TIMEOUT_SECONDS
    logger.info(f"測試模式：模型閒置超時設定為 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒。")

    try:
        logger.info("\n[步驟 1] 初始化LLM模型到CPU...")
        initialize_llm_model(force_cpu_init=True) # 明確強制CPU初始化
        logger.info(f"LLM 已初始化到 CPU。設備: {next(_llm_model.parameters()).device if _llm_model else 'N/A'}")

        logger.info("\n[步驟 2] 獲取向量資料庫實例...")
        db = get_vector_db() 
        if db: logger.info(f"向量資料庫獲取成功。記錄數: {db._collection.count() if hasattr(db, '_collection') else 'N/A'}")
        else: logger.error(f"獲取向量資料庫實例失敗。路徑 '{DB_PATH}' 可能不存在。")

        logger.info("\n[步驟 3] 模擬對話啟動 (模型移至GPU)...")
        model, tokenizer, _ = get_llm_model_and_tokenizer(ensure_on_gpu=True, update_last_used_time=True)
        if model and tokenizer:
            current_device = next(model.parameters()).device
            logger.info(f"模型目前設備: {current_device}")
        
            if torch.cuda.is_available() and current_device.type == "cuda":
                logger.info(f"模型已移至GPU。預計閒置 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒後自動移回CPU。")
                prompt = "你好"
                inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
                with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=10)
                logger.info(f"測試生成: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

                wait_duration = MONITOR_CHECK_INTERVAL_SECONDS + TEST_INACTIVITY_TIMEOUT_SECONDS + 5
                logger.info(f"\n[步驟 4] 等待 {wait_duration} 秒觀察模型是否自動移回CPU...")
                time.sleep(wait_duration)
                
                model_after_wait, _, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
                if model_after_wait:
                    final_device = next(model_after_wait.parameters()).device
                    logger.info(f"等待後，模型目前設備: {final_device}")
                    if final_device.type == "cpu": logger.info("模型已成功自動移回CPU。")
                    else: logger.warning(f"模型未按預期移回CPU (仍在 {final_device})。")
                else: logger.error("等待後，模型實例為 None。")
            else: logger.info("\n[步驟 4] CUDA不可用或模型不在GPU，跳過閒置轉移CPU的等待測試。")
        else: logger.error("[步驟 3] 獲取模型和分詞器失敗。")
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}", exc_info=True)
    finally:
        logger.info("\n[步驟 5] 釋放所有資源...")
        shutdown_llm_resources()
        logger.info("="*30 + " LLM_init.py 腳本測試完畢 (啟動修正版) " + "="*30)
