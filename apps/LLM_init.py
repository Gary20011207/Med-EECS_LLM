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

# 嘗試導入 RAG_build 中的函數，用於自動建立向量資料庫
try:
    from RAG_build import reset_and_rebuild_vectordb
    RAG_BUILD_AVAILABLE = True
except ImportError:
    RAG_BUILD_AVAILABLE = False
    def reset_and_rebuild_vectordb(*args, **kwargs): # 建立一個虛擬函數
        logging.error("RAG_build.py 或其 reset_and_rebuild_vectordb 函數無法導入。向量資料庫自動建立功能將不可用。")
        raise NotImplementedError("reset_and_rebuild_vectordb is not available.")

# --- 基本設定 ---
# 設定日誌記錄器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 組態設定 ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M" # LLM 模型名稱
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # 嵌入模型的名稱，與 RAG_build.py 一致
DB_PATH = "./VectorDB" # 向量資料庫的路徑，與 RAG_build.py 一致
PDF_FILES_PATH = "./PDFS" # PDF 文件的路徑，用於 RAG_build.py
DEFAULT_INACTIVITY_TIMEOUT = 600  # 預設閒置時間（秒），10分鐘無活動後將模型移至CPU

# --- 全域變數 ---
_llm_model = None            # LLM 模型實例
_tokenizer = None          # LLM 分詞器實例
_model_max_context_length = None # 模型最大上下文長度
_embedding_function = None # 嵌入函數實例
_vector_db = None          # 向量資料庫實例
_last_model_use_time = 0   # 上次使用模型的時間戳
_model_management_lock = threading.RLock() # 使用可重入鎖以避免同一線程內重複獲取鎖導致的死鎖
_device_monitor_thread = None # 設備監控線程，用於管理模型在CPU/GPU間的移動
_current_inactivity_timeout = DEFAULT_INACTIVITY_TIMEOUT # 當前閒置超時時間，允許在測試中修改

# =======================
# 設備管理 (Device Management)
# =======================
def _ensure_device_monitor_started():
    """
    確保設備監控線程已啟動。
    此線程負責在模型閒置一段時間後將其從GPU移至CPU，以節省GPU資源。
    """
    global _device_monitor_thread
    with _model_management_lock: # 保護對 _device_monitor_thread 的檢查與啟動
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
            return # 監控線程已在運行

        def _monitor_model_device_activity():
            """
            監控模型設備活動的內部函數，在單獨線程中運行。
            """
            global _llm_model, _last_model_use_time, _current_inactivity_timeout
            logger.info("模型設備活動監控線程已啟動。")
            while True:
                time.sleep(30) # 每30秒檢查一次
                
                with _model_management_lock: # 操作共享資源前獲取鎖
                    if _llm_model is None:
                        # logger.debug("監控線程：模型未載入，跳過檢查。")
                        continue # 如果模型未載入，則不進行任何操作

                    # 檢查模型是否在GPU上
                    # model.device 可能會引發異常，如果模型結構複雜或未完全載入
                    try:
                        model_device_type = next(_llm_model.parameters()).device.type
                    except Exception as e:
                        logger.warning(f"監控線程：無法獲取模型設備類型 ({e})，跳過此次檢查。")
                        continue

                    if model_device_type == "cuda":
                        current_time = time.time()
                        # 檢查是否超過閒置時間
                        if (current_time - _last_model_use_time > _current_inactivity_timeout):
                            logger.info(f"模型閒置超過 {_current_inactivity_timeout} 秒，準備移至CPU...")
                            try:
                                _llm_model = _llm_model.to("cpu")
                                torch.cuda.empty_cache() # 清理GPU快取
                                logger.info("模型已成功移至CPU，GPU記憶體已釋放。")
                            except Exception as e:
                                logger.error(f"監控線程：將模型移至CPU時發生錯誤: {e}")
                        # else:
                            # logger.debug(f"監控線程：模型在GPU上，活躍時間: {current_time - _last_model_use_time:.2f}s / {_current_inactivity_timeout}s")
                    # else:
                        # logger.debug(f"監控線程：模型在CPU上 ({model_device_type})，無需處理。")
        
        _device_monitor_thread = threading.Thread(target=_monitor_model_device_activity, daemon=True)
        _device_monitor_thread.start()
        logger.info("設備監控線程已建立並啟動。")

# ===========================
# 初始化函數 (Initialization Functions)
# ===========================
def initialize_llm_model(force_cpu_init=False):
    """
    初始化或獲取LLM模型和分詞器。
     Args:
        force_cpu_init (bool): 是否強制在CPU上初始化模型。
                               若為 False，則會嘗試使用GPU (如果可用)。
                               對於首次下載模型或節省啟動時的GPU資源，建議設為True。
    Returns:
        tuple: (model, tokenizer, max_context_length)
               模型實例，分詞器實例，以及模型的最大上下文長度。
               如果初始化失敗，則可能引發異常。
    """
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time
    
    with _model_management_lock: # 確保線程安全
        if _llm_model is not None and _tokenizer is not None:
            logger.info("LLM 模型和分詞器已載入，直接返回。")
            return _llm_model, _tokenizer, _model_max_context_length
        
        logger.info(f"開始初始化 LLM 模型: {LLM_MODEL_NAME}...")
        
        # 決定模型載入設備
        if force_cpu_init:
            target_device = "cpu"
        elif torch.cuda.is_available():
            target_device = "auto" # "auto" 會讓 transformers 自動選擇 GPU
            logger.info("檢測到可用CUDA，將嘗試在GPU上初始化模型。")
        else:
            target_device = "cpu"
            logger.info("未檢測到可用CUDA或被強制，將在CPU上初始化模型。")

        # 設定量化組態 (BitsAndBytesConfig for 4-bit quantization)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        try:
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                quantization_config=quantization_config, # 套用量化組態
                device_map=target_device, # 指定設備映射 ('auto' 通常表示優先GPU)
                trust_remote_code=True    # 對於某些 HuggingFace Hub 上的模型需要設定
            ).eval() # 設定為評估模式
            
            _model_max_context_length = _tokenizer.model_max_length
            _last_model_use_time = time.time() # 更新最後使用時間
            
            # 確保模型在目標設備上
            loaded_device = next(_llm_model.parameters()).device
            logger.info(f"LLM 模型 ({LLM_MODEL_NAME}) 初始化完成。")
            logger.info(f"模型實際載入於: {loaded_device}, 要求設備: {target_device}, 上下文長度: {_model_max_context_length}")

        except Exception as e:
            logger.error(f"初始化LLM模型 {LLM_MODEL_NAME} 時發生嚴重錯誤: {e}", exc_info=True)
            _llm_model = None # 重設為 None
            _tokenizer = None
            raise # 重新引發錯誤，讓上層處理

        _ensure_device_monitor_started() # 確保監控線程已啟動
            
        return _llm_model, _tokenizer, _model_max_context_length

def initialize_vector_database():
    """
    初始化或獲取向量資料庫 (ChromaDB)。
    如果指定的資料庫路徑不存在或為空，且 RAG_build 功能可用，
    它會嘗試調用 reset_and_rebuild_vectordb 來建立資料庫。

    Returns:
        Chroma: 向量資料庫實例。
                如果初始化失敗，則可能引發異常。
    """
    global _embedding_function, _vector_db
    
    with _model_management_lock: # 保護共享資源
        if _vector_db is not None:
            logger.info("向量資料庫已載入，直接返回。")
            return _vector_db
        
        logger.info(f"開始連接向量資料庫，路徑: {DB_PATH}")
        
        # 檢查資料庫是否存在且非空
        if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
            logger.warning(f"向量資料庫路徑 {DB_PATH} 不存在或為空。")
            if RAG_BUILD_AVAILABLE:
                logger.info(f"嘗試使用 RAG_build.py 中的 reset_and_rebuild_vectordb 函數重建資料庫...")
                try:
                    # 使用 RAG_build.py 中定義的預設 PDF 路徑，或在此處明確指定
                    reset_and_rebuild_vectordb(
                        pdf_folder=PDF_FILES_PATH, # 指定PDF來源
                        db_path=DB_PATH,           # 指定DB儲存位置
                        emb_model=EMBEDDINGS_MODEL_NAME, # 確保嵌入模型一致
                        force_reset=True           # 強制重建
                    )
                    logger.info(f"向量資料庫已成功在 {DB_PATH} 重建。")
                except Exception as e:
                    logger.error(f"使用 reset_and_rebuild_vectordb 重建向量資料庫失敗: {e}", exc_info=True)
                    # 如果重建失敗，依然拋出錯誤，因為後續操作會依賴此資料庫
                    raise FileNotFoundError(f"無法建立或找到向量資料庫於 {DB_PATH}，即使嘗試重建也失敗。") from e
            else:
                logger.error("RAG_build.py 功能不可用，無法自動重建向量資料庫。請先手動運行 RAG_build.py。")
                raise FileNotFoundError(f"向量資料庫路徑 {DB_PATH} 不存在或為空，且無法自動重建。")
        
        try:
            _embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
            _vector_db = Chroma(
                embedding_function=_embedding_function,
                persist_directory=DB_PATH
            )
            logger.info(f"成功連接到向量資料庫: {DB_PATH}。資料庫包含 {_vector_db._collection.count()} 條記錄。")
        except Exception as e:
            logger.error(f"連接到向量資料庫 {DB_PATH} 時發生錯誤: {e}", exc_info=True)
            _vector_db = None # 重設為 None
            raise # 重新引發錯誤

        return _vector_db

# ===========================
# 存取函數 (Access Functions)
# ===========================
def get_llm_model_and_tokenizer():
    """
    獲取LLM模型和分詞器，並確保模型在適當的設備上（優先GPU）。
    如果模型不在GPU上且GPU可用，則會將其移至GPU。
    此函數會更新模型的最後使用時間，供設備監控線程使用。

    Returns:
        tuple: (model, tokenizer, max_context_length)
    """
    global _llm_model, _tokenizer, _model_max_context_length, _last_model_use_time
    
    with _model_management_lock: # 線程安全
        # 1. 如果模型未初始化，則進行初始化
        if _llm_model is None or _tokenizer is None:
            logger.info("get_llm_model_and_tokenizer: 模型或分詞器未初始化，開始初始化程序...")
            # 嘗試在GPU上初始化 (如果可用)，否則在CPU上
            initialize_llm_model(force_cpu_init=not torch.cuda.is_available())
            if _llm_model is None: # 初始化失敗
                 logger.error("get_llm_model_and_tokenizer: 初始化後模型仍為 None，請檢查錯誤日誌。")
                 raise RuntimeError("LLM 模型初始化失敗。")
        
        # 2. 檢查並移動模型到目標設備 (優先GPU)
        try:
            model_device_type = next(_llm_model.parameters()).device.type
        except Exception as e:
            logger.error(f"get_llm_model_and_tokenizer: 無法獲取模型設備類型 ({e})", exc_info=True)
            raise RuntimeError(f"無法確認或操作模型設備: {e}") from e

        if torch.cuda.is_available():
            if model_device_type == "cpu":
                logger.info("get_llm_model_and_tokenizer: CUDA可用，模型目前在CPU，準備移至GPU...")
                try:
                    _llm_model = _llm_model.to("cuda")
                    logger.info("模型已成功移至GPU。")
                except Exception as e:
                    logger.error(f"get_llm_model_and_tokenizer: 將模型移至GPU時發生錯誤: {e}。模型將保留在CPU。", exc_info=True)
                    # 如果移動失敗，保持在CPU上，避免應用程式崩潰
            # else: logger.debug("模型已在GPU上。")
        else: # CUDA 不可用
            if model_device_type == "cuda": # 理論上不應發生，但作為安全檢查
                logger.warning("get_llm_model_and_tokenizer: CUDA突然不可用，但模型在GPU上。準備移至CPU...")
                try:
                    _llm_model = _llm_model.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("模型已因CUDA不可用而移至CPU。")
                except Exception as e:
                    logger.error(f"get_llm_model_and_tokenizer: 將模型從失效的GPU移至CPU時發生錯誤: {e}", exc_info=True)
            # else: logger.debug("CUDA不可用，模型已在CPU上。")
        
        # 3. 更新最後使用時間
        _last_model_use_time = time.time()
        # logger.debug(f"模型最後使用時間已更新: {_last_model_use_time}")
            
        return _llm_model, _tokenizer, _model_max_context_length

def get_vector_db_instance():
    """
    獲取向量資料庫的實例。
    如果尚未初始化，則會調用 initialize_vector_database()。

    Returns:
        Chroma: 向量資料庫實例。
    """
    global _vector_db
    with _model_management_lock: # 雖然 ChromaDB 本身可能有線程安全機制，但對全域變數的訪問最好加鎖
        if _vector_db is None:
            logger.info("get_vector_db_instance: 向量資料庫未初始化，開始初始化...")
            initialize_vector_database() # 此函數內部已有日誌和錯誤處理
            if _vector_db is None: # 初始化失敗
                logger.error("get_vector_db_instance: 初始化後向量資料庫仍為 None。")
                raise RuntimeError("向量資料庫初始化失敗。")
        return _vector_db

# ===========================
# 清理函數 (Cleanup Function)
# ===========================
def shutdown_llm_resources():
    """
    釋放LLM模型、分詞器等相關資源，並將模型移至CPU。
    通常在應用程式關閉時調用。
    """
    global _llm_model, _tokenizer, _vector_db, _embedding_function, _device_monitor_thread
    
    with _model_management_lock:
        logger.info("開始釋放 LLM 及相關資源...")
        
        if _llm_model is not None:
            try:
                if next(_llm_model.parameters()).device.type != "cpu":
                    logger.info("將 LLM 模型移至 CPU...")
                    _llm_model = _llm_model.to("cpu")
            except Exception as e:
                logger.warning(f"關閉前將模型移至CPU失敗: {e}")
            del _llm_model # 刪除模型實例的引用
            _llm_model = None
            logger.info("LLM 模型已卸載。")
        
        if _tokenizer is not None:
            del _tokenizer # 刪除分詞器實例的引用
            _tokenizer = None
            logger.info("分詞器已卸載。")
        
        # 向量資料庫和嵌入函數通常不需要特別的“關閉”操作，
        # 但為了保持一致性，也將它們的引用設為 None
        if _vector_db is not None:
            _vector_db = None # ChromaDB 的持久化是基於文件的，這裡僅釋放內存中的實例引用
            logger.info("向量資料庫實例引用已釋放。")
        
        if _embedding_function is not None:
            _embedding_function = None
            logger.info("嵌入函數實例引用已釋放。")

        _model_max_context_length = None
        _last_model_use_time = 0
            
        # 停止監控線程 (daemon線程會在主線程結束時自動結束，但這裡可以做個標記)
        # 注意：無法從外部安全地“停止”一個正在運行的線程，除非線程自身設計了停止機制。
        # 由於是 daemon 線程，它會隨主程序退出。
        if _device_monitor_thread is not None and _device_monitor_thread.is_alive():
             logger.info("設備監控線程將隨主程序退出。")
        _device_monitor_thread = None # 清除引用

        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 清理GPU快取
            logger.info("GPU 快取已清理。")
            
        logger.info("LLM 相關資源釋放完畢。")

# ===========================
# 輔助函數 (Utility Function)
# ===========================
def count_tokens(text_to_count: str) -> int:
    """
    計算給定文本的 token 數量。
    會自動獲取或初始化分詞器。

    Args:
        text_to_count (str): 需要計算 token 的文本。

    Returns:
        int: 文本的 token 數量。
    """
    try:
        _, tokenizer_instance, _ = get_llm_model_and_tokenizer()
        if tokenizer_instance is None:
            logger.error("count_tokens: 無法獲取分詞器。")
            return -1 # 表示錯誤
        return len(tokenizer_instance.encode(text_to_count, add_special_tokens=False))
    except Exception as e:
        logger.error(f"count_tokens: 計算 token 時出錯: {e}", exc_info=True)
        return -1


# ===========================
# 主執行區塊 (Standalone Run)
# ===========================
if __name__ == "__main__":
    logger.info("="*30 + " LLM_init.py 腳本獨立運行測試 " + "="*30)
    
    # 為了測試，設定一個較短的閒置時間
    TEST_INACTIVITY_TIMEOUT_SECONDS = 15 # 改為15秒方便觀察
    _current_inactivity_timeout = TEST_INACTIVITY_TIMEOUT_SECONDS
    logger.info(f"測試模式：模型閒置超時設定為 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒。")

    main_model = None
    main_tokenizer = None

    try:
        # 1. 應用程式啟動：預先載入LLM至CPU standby (模擬 app.py 啟動)
        logger.info("\n[步驟 1] 模擬應用程式啟動：初始化LLM模型到CPU...")
        # 首次初始化強制CPU，避免下載時佔用GPU
        main_model, main_tokenizer, max_len = initialize_llm_model(force_cpu_init=True)
        if main_model and main_tokenizer:
            logger.info(f"LLM ({LLM_MODEL_NAME}) 已成功初始化到 CPU standby。設備: {next(main_model.parameters()).device}")
        else:
            logger.error("LLM 模型初始化到 CPU 失敗。測試終止。")
            exit()

        # 2. 獲取向量資料庫 (如果是空的，會嘗試調用 reset_and_rebuild_vectordb 函數建立)
        logger.info("\n[步驟 2] 初始化/獲取向量資料庫...")
        db = get_vector_db_instance()
        if db:
            logger.info(f"向量資料庫 ({DB_PATH}) 獲取成功。包含 {db._collection.count()} 條記錄。")
        else:
            logger.error("向量資料庫獲取失敗。測試終止。")
            exit()

        # 3. 模擬對話啟動：調用 get_llm_model (會將模型移到GPU/CUDA)
        logger.info("\n[步驟 3] 模擬對話啟動：調用 get_llm_model_and_tokenizer()...")
        main_model, main_tokenizer, max_len = get_llm_model_and_tokenizer()
        current_device = next(main_model.parameters()).device
        logger.info(f"get_llm_model_and_tokenizer() 調用成功。模型目前設備: {current_device}")
        
        if torch.cuda.is_available() and current_device.type == "cuda":
            logger.info(f"模型已移至GPU。預計在閒置 {TEST_INACTIVITY_TIMEOUT_SECONDS} 秒後自動移回CPU。")
        elif current_device.type == "cpu":
            logger.info("模型仍在CPU (可能CUDA不可用或移動失敗)。閒置移回CPU的邏輯將不會在GPU上觸發。")
        
        # 測試一個簡單的對話 (無RAG)
        logger.info("\n[步驟 3.1] 測試簡單對話生成...")
        prompt = "你好，請用中文簡短介紹一下你自己以及你的功能。"
        logger.info(f"測試提示 (Prompt): '{prompt}'")
        
        # 檢查 tokenizer 和 model 是否有效
        if main_tokenizer is None or main_model is None:
            logger.error("Tokenizer 或 Model 未正確初始化，無法進行生成。")
        else:
            inputs = main_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(current_device)
            logger.info(f"輸入已轉換為張量，設備: {inputs['input_ids'].device}")
            
            # 生成時禁用梯度計算
            with torch.no_grad():
                outputs = main_model.generate(
                    **inputs, 
                    max_new_tokens=100, # 限制新生成的token數量
                    pad_token_id=main_tokenizer.eos_token_id # 確保能正確處理填充
                ) 
            response = main_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"模型回應 (Response): '{response}'")
            logger.info(f"測試 Token 計數 for '{prompt}': {count_tokens(prompt)}")


        # 4. 觀察計時器行為 (自動將閒置的LLM移回CPU)
        if torch.cuda.is_available() and current_device.type == "cuda":
            logger.info(f"\n[步驟 4] 等待 {TEST_INACTIVITY_TIMEOUT_SECONDS + 5} 秒觀察模型是否自動從GPU移回CPU...")
            time.sleep(TEST_INACTIVITY_TIMEOUT_SECONDS + 5) # 等待超過超時時間
            
            # 再次檢查模型設備 (需要再次獲取鎖和模型實例)
            with _model_management_lock:
                if _llm_model: # 檢查模型是否仍然存在
                    final_device = next(_llm_model.parameters()).device
                    logger.info(f"等待後，模型目前設備: {final_device}")
                    if final_device.type == "cpu":
                        logger.info("模型已成功自動移回CPU。")
                    else:
                        logger.warning("模型未按預期移回CPU。請檢查監控線程日誌。")
                else:
                    logger.warning("等待後，模型實例變為 None，可能已被卸載。")
        else:
            logger.info("\n[步驟 4] CUDA不可用或模型不在GPU，跳過閒置轉移CPU的等待測試。")


    except FileNotFoundError as e:
        logger.error(f"初始化錯誤 (FileNotFoundError): {e}")
        logger.error("請確保您的向量資料庫路徑正確，或者 PDFS 文件夾中有文件可供 RAG_build.py 建立資料庫。")
    except NotImplementedError as e:
        logger.error(f"功能未實現錯誤: {e}")
        logger.error("可能是 RAG_build.py 相關功能無法導入，請檢查環境和文件結構。")
    except RuntimeError as e:
        logger.error(f"運行時錯誤: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"測試過程中發生未預期錯誤: {e}", exc_info=True)
    finally:
        # 5. 結束應用 (模擬 app.py 關閉，卸載模型)
        logger.info("\n[步驟 5] 模擬應用程式關閉：釋放所有資源...")
        shutdown_llm_resources()
        logger.info("="*30 + " LLM_init.py 腳本測試完畢 " + "="*30)