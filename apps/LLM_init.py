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
import traceback # 引入 traceback

# 嘗試導入 RAG_build 中的函數
try:
    from .RAG_build import reset_and_rebuild_vectordb # 使用相對導入
    RAG_BUILD_AVAILABLE = True
except ImportError:
    RAG_BUILD_AVAILABLE = False
    def reset_and_rebuild_vectordb(*args, **kwargs):
        logging.error("RAG_build.py 或其 reset_and_rebuild_vectordb 函數無法導入。")
        raise NotImplementedError("reset_and_rebuild_vectordb is not available.")

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- 組態設定 ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./VectorDB"
PDF_FILES_PATH = "./PDFS" # RAG_build.py 會使用此路徑
DEFAULT_INACTIVITY_TIMEOUT = 600  # 預設閒置時間（秒），10分鐘
MONITOR_CHECK_INTERVAL_SECONDS = 30 # 監控線程檢查間隔

# --- 全域變數 ---
_llm_model = None
_tokenizer = None
_model_max_context_length = None
_embedding_function = None
_vector_db = None
_last_model_use_time = 0
_model_management_lock = threading.RLock() # 使用可重入鎖
_device_monitor_thread = None
_current_inactivity_timeout = DEFAULT_INACTIVITY_TIMEOUT


# =======================
# 設備管理 (Device Management)
# =======================
def _ensure_device_monitor_started():
    global _device_monitor_thread # 允許修改全域 _device_monitor_thread
    with _model_management_lock:
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
            return

        def _monitor_model_device_activity():
            # 此內部函數需要訪問和修改全域 _llm_model
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
                            # 嘗試獲取模型參數列表，以檢查模型是否有效且有參數
                            model_params = list(_llm_model.parameters())
                        except Exception as e_get_params:
                            logger.error(f"監控線程：調用 _llm_model.parameters() 時出錯: {e_get_params}", exc_info=True)
                            continue 

                        if not model_params:
                            logger.warning("監控線程：_llm_model 沒有參數。跳過設備檢查。")
                            continue
                            
                        # 安全地獲取第一個參數的設備類型
                        current_model_device_type = model_params[0].device.type
                        # logger.debug(f"監控線程：模型首參數設備類型: {current_model_device_type}")

                        if current_model_device_type == "cuda":
                            current_time = time.time()
                            if (current_time - _last_model_use_time > _current_inactivity_timeout):
                                logger.info(f"模型閒置超過 {_current_inactivity_timeout} 秒，準備移至CPU...")
                                try:
                                    # 因為 _llm_model 是全域變數，且 .to() 對於量化模型返回新物件，
                                    # 所以這裡的賦值必須更新全域的 _llm_model 參考。
                                    _llm_model = _llm_model.to("cpu") 
                                    torch.cuda.empty_cache()
                                    logger.info("模型已成功移至CPU，GPU記憶體已釋放。")
                                except Exception as e_to_cpu:
                                    logger.error(f"監控線程：將模型移至CPU時發生錯誤: {e_to_cpu}", exc_info=True)
                except Exception as e_thread_loop:
                    logger.error(f"模型設備監控線程主循環發生未預期錯誤: {e_thread_loop}", exc_info=True)
                    time.sleep(MONITOR_CHECK_INTERVAL_SECONDS * 2) # 若循環本身出錯，則等待更長時間
        
        _device_monitor_thread = threading.Thread(target=_monitor_model_device_activity, daemon=True)
        _device_monitor_thread.start()
        logger.info(f"設備監控線程已建立並啟動 (檢查間隔: {MONITOR_CHECK_INTERVAL_SECONDS}s, 閒置超時: {_current_inactivity_timeout}s)。")

# ===========================
# 初始化函數 (Initialization Functions)
# ===========================
def initialize_llm_model(force_cpu_init=False):
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time # 聲明修改全域變數
    with _model_management_lock:
        if _llm_model is not None and _tokenizer is not None:
            logger.info("LLM 模型和分詞器已載入。")
            return _llm_model, _tokenizer, _model_max_context_length
        
        logger.info(f"開始初始化 LLM 模型: {LLM_MODEL_NAME}...")
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
            _llm_model, _tokenizer = None, None # 確保失敗時重置
            raise
        _ensure_device_monitor_started() # 在成功初始化後啟動監控
        return _llm_model, _tokenizer, _model_max_context_length

def initialize_vector_database():
    global _embedding_function, _vector_db # 聲明修改全域變數
    with _model_management_lock:
        if _vector_db is not None:
            logger.info("向量資料庫已載入。")
            return _vector_db
        
        logger.info(f"開始連接向量資料庫: {DB_PATH}")
        if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH): # 檢查路徑是否存在且非空
            logger.warning(f"向量資料庫路徑 {DB_PATH} 不存在或為空。")
            if RAG_BUILD_AVAILABLE:
                logger.info(f"嘗試使用 RAG_build.py 重建資料庫...")
                try:
                    reset_and_rebuild_vectordb(
                        pdf_folder=PDF_FILES_PATH, db_path=DB_PATH,
                        emb_model=EMBEDDINGS_MODEL_NAME, force_reset=True
                    )
                    logger.info(f"向量資料庫已在 {DB_PATH} 重建。")
                except Exception as e:
                    logger.error(f"使用 reset_and_rebuild_vectordb 重建向量資料庫失敗: {e}", exc_info=True)
                    raise FileNotFoundError(f"無法建立或找到向量資料庫於 {DB_PATH}。") from e
            else: # RAG_BUILD 不可用
                logger.error("RAG_build.py 功能不可用，無法自動重建。請先手動運行 RAG_build.py。")
                raise FileNotFoundError(f"向量資料庫 {DB_PATH} 不存在或為空，且無法自動重建。")
        try:
            _embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
            _vector_db = Chroma(embedding_function=_embedding_function, persist_directory=DB_PATH)
            logger.info(f"成功連接到向量資料庫: {DB_PATH} (記錄數: {_vector_db._collection.count() if _vector_db and hasattr(_vector_db, '_collection') and _vector_db._collection else 'N/A'})")
        except Exception as e:
            logger.error(f"連接到向量資料庫 {DB_PATH} 失敗: {e}", exc_info=True)
            _vector_db = None # 確保失敗時重置
            raise
        return _vector_db

# ===========================
# 存取函數 (Access Functions)
# ===========================
def get_llm_model_and_tokenizer(
    update_last_used_time: bool = True, 
    ensure_on_gpu: bool = True         
):
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time # 聲明修改全域變數
    with _model_management_lock:
        if _llm_model is None or _tokenizer is None:
            logger.info("get_llm_model_and_tokenizer: 模型或分詞器未初始化，開始初始化...")
            # 預設嘗試GPU (如果可用)，除非 force_cpu_init=True 在 initialize_llm_model 中被設置
            initialize_llm_model(force_cpu_init=not torch.cuda.is_available()) 
            if _llm_model is None: # 雙重檢查，確保初始化成功
                 logger.error("get_llm_model_and_tokenizer: 初始化後模型仍為 None。")
                 raise RuntimeError("LLM 模型初始化失敗。")
        
        if ensure_on_gpu and torch.cuda.is_available():
            try:
                model_device_type = next(_llm_model.parameters()).device.type
                if model_device_type == "cpu":
                    logger.info("get_llm_model_and_tokenizer: (ensure_on_gpu=True) CUDA可用，模型在CPU，移至GPU...")
                    _llm_model = _llm_model.to("cuda") # 更新全域 _llm_model
                    logger.info("模型已成功移至GPU。")
            except Exception as e: # 更廣泛地捕獲可能的錯誤
                logger.error(f"get_llm_model_and_tokenizer: 嘗試將模型移至GPU時出錯: {e}", exc_info=True)
        
        if update_last_used_time:
            _last_model_use_time = time.time()
            
        return _llm_model, _tokenizer, _model_max_context_length

def get_vector_db_instance():
    global _vector_db # 聲明修改全域變數
    with _model_management_lock:
        if _vector_db is None:
            logger.info("get_vector_db_instance: 向量資料庫未初始化，開始初始化...")
            initialize_vector_database()
            if _vector_db is None: # 雙重檢查
                logger.error("get_vector_db_instance: 初始化後向量資料庫仍為 None。")
                raise RuntimeError("向量資料庫初始化失敗。")
        return _vector_db

# ===========================
# 清理函數 (Cleanup Function)
# ===========================
def shutdown_llm_resources():
    global _llm_model, _tokenizer, _vector_db, _embedding_function, _device_monitor_thread
    global _model_max_context_length, _last_model_use_time # 確保也重設這些
    with _model_management_lock:
        logger.info("開始釋放 LLM 及相關資源...")
        if _llm_model is not None:
            try:
                if next(_llm_model.parameters()).device.type != "cpu":
                    logger.info("將 LLM 模型移至 CPU...")
                    _llm_model = _llm_model.to("cpu") # 更新全域 _llm_model
            except Exception as e: logger.warning(f"關閉前將模型移至CPU失敗: {e}")
            del _llm_model; _llm_model = None; logger.info("LLM 模型已卸載。")
        
        if _tokenizer is not None: del _tokenizer; _tokenizer = None; logger.info("分詞器已卸載。")
        if _vector_db is not None: _vector_db = None; logger.info("向量資料庫實例引用已釋放。")
        if _embedding_function is not None: _embedding_function = None; logger.info("嵌入函數實例引用已釋放。")
            
        _model_max_context_length = None
        _last_model_use_time = 0
            
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
             logger.info("設備監控線程將隨主程序退出。")
        _device_monitor_thread = None # 清除引用

        if torch.cuda.is_available():
            torch.cuda.empty_cache(); logger.info("GPU 快取已清理。")
        logger.info("LLM 相關資源釋放完畢。")

# ===========================
# 輔助函數 (Utility Function)
# ===========================
def count_tokens(text_to_count: str) -> int:
    try:
        # 計算token不應觸發GPU移動或時間更新，也不應在模型未初始化時嘗試初始化
        # 因此，先檢查模型是否已初始化
        with _model_management_lock: # 確保線程安全地訪問 _tokenizer
            if _tokenizer is None:
                 # 如果分詞器未初始化，可以選擇返回錯誤，或者嘗試被動獲取一次
                 # 這裡我們選擇如果未初始化則不進行初始化，直接報錯或返回預設值
                logger.warning("count_tokens: 分詞器尚未初始化，無法計算 token。")
                return 0 # 或 -1 表示錯誤
            tokenizer_instance = _tokenizer
        
        return len(tokenizer_instance.encode(text_to_count, add_special_tokens=False))
    except Exception as e:
        logger.error(f"count_tokens: 計算 token 時出錯: {e}", exc_info=True)
        return -1 # 表示錯誤

# ===========================
# 主執行區塊 (Standalone Run)
# ===========================
if __name__ == "__main__":
    logger.info("="*30 + " LLM_init.py 腳本獨立運行測試 " + "="*30)
    TEST_INACTIVITY_TIMEOUT_SECONDS = 15
    _current_inactivity_timeout = TEST_INACTIVITY_TIMEOUT_SECONDS # 修改全域變數以供測試
    logger.info(f"測試模式：模型閒置超時設定為 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒。")

    try:
        logger.info("\n[步驟 1] 初始化LLM模型到CPU...")
        initialize_llm_model(force_cpu_init=True)
        logger.info(f"LLM 已初始化到 CPU。設備: {next(_llm_model.parameters()).device if _llm_model else 'N/A'}")

        logger.info("\n[步驟 2] 初始化/獲取向量資料庫...")
        db = get_vector_db_instance()
        logger.info(f"向量資料庫獲取成功。記錄數: {db._collection.count() if db and hasattr(db, '_collection') and db._collection else 'N/A'}")

        logger.info("\n[步驟 3] 模擬對話啟動 (模型移至GPU)...")
        model, tokenizer, _ = get_llm_model_and_tokenizer(ensure_on_gpu=True, update_last_used_time=True)
        current_device = next(model.parameters()).device
        logger.info(f"模型目前設備: {current_device}")
        
        if torch.cuda.is_available() and current_device.type == "cuda":
            logger.info(f"模型已移至GPU。預計閒置 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒後自動移回CPU。")
            prompt = "你好"
            inputs = tokenizer(prompt, return_tensors="pt").to(current_device)
            with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=10)
            logger.info(f"測試生成: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

            # 等待時間應大於 (監控線程檢查間隔 + 設定的閒置超時 + 一點緩衝)
            wait_duration = MONITOR_CHECK_INTERVAL_SECONDS + TEST_INACTIVITY_TIMEOUT_SECONDS + 5
            logger.info(f"\n[步驟 4] 等待 {wait_duration} 秒觀察模型是否自動移回CPU...")
            time.sleep(wait_duration)
            
            # 再次檢查模型設備 (使用被動模式獲取，不觸發GPU移動或時間更新)
            model_after_wait, _, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
            if model_after_wait: # 確保模型對象存在
                final_device = next(model_after_wait.parameters()).device
                logger.info(f"等待後，模型目前設備: {final_device}")
                if final_device.type == "cpu": logger.info("模型已成功自動移回CPU。")
                else: logger.warning(f"模型未按預期移回CPU (仍在 {final_device})。請檢查監控線程日誌和等待時間。")
            else:
                logger.error("等待後，模型實例為 None，無法檢查設備。")
        else:
            logger.info("\n[步驟 4] CUDA不可用或模型不在GPU，跳過閒置轉移CPU的等待測試。")
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}", exc_info=True)
    finally:
        logger.info("\n[步驟 5] 釋放所有資源...")
        shutdown_llm_resources()
        logger.info("="*30 + " LLM_init.py 腳本測試完畢 " + "="*30)