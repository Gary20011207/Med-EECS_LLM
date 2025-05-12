# core/model_manager.py
import time
import threading
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
import logging
from typing import Optional, Any, Tuple

# 導入配置
try:
    from config import (
        LLM_MODEL_NAME,
        DEFAULT_INACTIVITY_TIMEOUT,
        MONITOR_CHECK_INTERVAL_SECONDS
    )
except ImportError:
    # 如果無法導入配置，使用預設值
    LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
    DEFAULT_INACTIVITY_TIMEOUT = 600
    MONITOR_CHECK_INTERVAL_SECONDS = 30

logger = logging.getLogger(__name__)

class ModelManager:
    """LLM 模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.model_name = LLM_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.model_max_context_length = None
        self.last_model_use_time = 0.0
        self.inactivity_timeout = DEFAULT_INACTIVITY_TIMEOUT
        self._model_management_lock = threading.RLock()
        self._device_monitor_thread = None
        self._shutdown_flag = threading.Event()
        
        logger.info(f"ModelManager 初始化完成 (模型: {self.model_name})")
    
    def initialize(self, force_cpu_init: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """初始化模型，強制在 CPU 上載入
        
        Args:
            force_cpu_init: 是否強制在 CPU 上初始化
            
        Returns:
            (model, tokenizer, max_context_length) 的元組
        """
        with self._model_management_lock:
            if self.model is not None and self.tokenizer is not None:
                logger.debug("LLM 模型和分詞器已載入")
                return self.model, self.tokenizer, self.model_max_context_length
            
            logger.info(f"開始初始化 LLM 模型: {self.model_name} (強制CPU: {force_cpu_init})...")
            
            # 決定目標設備
            if force_cpu_init or not torch.cuda.is_available():
                target_device = "cpu"
            else:
                target_device = "auto"
            
            logger.info(f"LLM 初始化目標設備: {target_device} (CUDA可用: {torch.cuda.is_available()})")
            
            # 配置量化設定
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            try:
                # 載入分詞器
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                
                # 載入模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    quantization_config=quantization_config,
                    device_map=target_device, 
                    trust_remote_code=True
                ).eval()
                
                # 設定模型屬性
                self.model_max_context_length = getattr(self.tokenizer, 'model_max_length', 2048)
                self.last_model_use_time = time.time()
                
                # 記錄實際載入設備
                loaded_device = next(self.model.parameters()).device
                logger.info(f"LLM 模型 ({self.model_name}) 初始化完成。實際載入於: {loaded_device}, 上下文長度: {self.model_max_context_length}")
                
            except Exception as e:
                logger.critical(f"初始化LLM模型 {self.model_name} 失敗: {e}", exc_info=True)
                self.model, self.tokenizer = None, None
                raise
            
            # 啟動設備監控線程
            self._ensure_device_monitor_started()
            
            return self.model, self.tokenizer, self.model_max_context_length
    
    def get_model_and_tokenizer(self, 
                               update_last_used_time: bool = True, 
                               ensure_on_gpu: bool = True) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """獲取模型和分詞器，根據需要將模型移至 GPU
        
        Args:
            update_last_used_time: 是否更新最後使用時間
            ensure_on_gpu: 是否確保模型在 GPU 上
            
        Returns:
            (model, tokenizer, max_context_length) 的元組
        """
        with self._model_management_lock:
            # 如果模型未初始化，進行首次初始化（強制到 CPU）
            if self.model is None or self.tokenizer is None:
                logger.info("get_model_and_tokenizer: 模型或分詞器未初始化，將進行首次CPU初始化...")
                self.initialize(force_cpu_init=True)
                if self.model is None:
                    logger.error("get_model_and_tokenizer: 初始化後模型仍為 None")
                    raise RuntimeError("LLM 模型初始化失敗")
            
            # 根據需要將模型移至 GPU
            if ensure_on_gpu and torch.cuda.is_available():
                try:
                    model_device_type = next(self.model.parameters()).device.type
                    if model_device_type == "cpu":
                        logger.info("get_model_and_tokenizer: (ensure_on_gpu=True) CUDA可用，模型在CPU，移至GPU...")
                        self.model = self.model.to("cuda")
                        logger.info("模型已成功移至GPU")
                except Exception as e:
                    logger.error(f"get_model_and_tokenizer: 嘗試將模型移至GPU時出錯: {e}", exc_info=True)
            
            # 更新最後使用時間
            if update_last_used_time:
                self.last_model_use_time = time.time()
            
            return self.model, self.tokenizer, self.model_max_context_length
    
    def count_tokens(self, text_to_count: str) -> int:
        """計算文本的 token 數量
        
        Args:
            text_to_count: 要計算 token 的文本
            
        Returns:
            token 數量，錯誤時返回 -1
        """
        try:
            with self._model_management_lock:
                if self.tokenizer is None:
                    logger.warning("count_tokens: 分詞器尚未初始化，嘗試被動獲取...")
                    _, temp_tokenizer, _ = self.get_model_and_tokenizer(
                        update_last_used_time=False, 
                        ensure_on_gpu=False
                    )
                    if temp_tokenizer is None:
                        logger.error("count_tokens: 仍無法獲取分詞器")
                        return 0
                    tokenizer_instance = temp_tokenizer
                else:
                    tokenizer_instance = self.tokenizer
            
            return len(tokenizer_instance.encode(text_to_count, add_special_tokens=False))
        except Exception as e:
            logger.error(f"count_tokens: 計算 token 時出錯: {e}", exc_info=True)
            return -1
    
    def get_status(self) -> dict:
        """獲取模型狀態
        
        Returns:
            包含模型狀態資訊的字典，包括詳細的設備和記憶體信息
        """
        with self._model_management_lock:
            if self.model is None:
                return {    
                    "model_name": self.model_name,
                    "initialized": False,
                    "current_device": None,
                    "device_details": None,
                    "last_used": None,
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_memory": None,
                    "max_context_length": None
                }
            
            try:
                current_device = next(self.model.parameters()).device
                device_type = current_device.type
                
                # 準備設備詳細信息
                device_details = {
                    'type': device_type
                }
                
                # 如果是 GPU，添加索引信息
                if device_type == 'cuda':
                    device_details['index'] = current_device.index if hasattr(current_device, 'index') else 0
                
                # GPU 記憶體信息
                gpu_memory_info = None
                if device_type == 'cuda':
                    try:
                        allocated = round(torch.cuda.memory_allocated(current_device) / (1024 ** 3), 2)
                        reserved = round(torch.cuda.memory_reserved(current_device) / (1024 ** 3), 2)
                        gpu_memory_info = {
                            'allocated': allocated,
                            'reserved': reserved,
                            'unit': 'GB'
                        }
                    except Exception as e:
                        logger.warning(f"get_status: 獲取 GPU 記憶體信息時出錯: {e}")
                
            except Exception as e:
                logger.error(f"get_status: 獲取模型設備時出錯: {e}")
                device_type = "unknown"
                device_details = {'type': 'unknown'}
                gpu_memory_info = None
            
            return {
                "model_name": self.model_name,
                "initialized": True,
                "current_device": device_type,
                "device_details": device_details,
                "last_used": self.last_model_use_time,
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory": gpu_memory_info,
                "max_context_length": self.model_max_context_length
            }

    def shutdown(self):
        """釋放所有相關資源"""
        with self._model_management_lock:
            logger.info("開始釋放 LLM 及相關資源...")
            
            # 設定關閉標誌
            self._shutdown_flag.set()
            
            # 移動模型到 CPU
            if self.model is not None:
                try:
                    if next(self.model.parameters()).device.type != "cpu":
                        logger.info("將 LLM 模型移至 CPU...")
                        self.model = self.model.to("cpu")
                except Exception as e:
                    logger.warning(f"關閉前將模型移至CPU失敗: {e}")
                
                del self.model
                self.model = None
                logger.info("LLM 模型已卸載")
            
            # 釋放分詞器
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.info("分詞器已卸載")
            
            # 重置其他屬性
            self.model_max_context_length = None
            self.last_model_use_time = 0
            
            # 等待監控線程結束
            if self._device_monitor_thread is not None and self._device_monitor_thread.is_alive():
                logger.info("等待設備監控線程結束...")
                self._device_monitor_thread.join(timeout=5)
            
            self._device_monitor_thread = None
            
            # 清理 GPU 快取
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU 快取已清理")
            
            logger.info("LLM 相關資源釋放完畢")
    
    def _ensure_device_monitor_started(self):
        """確保設備監控線程已啟動"""
        if self._device_monitor_thread is not None and self._device_monitor_thread.is_alive():
            return
        
        logger.info(f"設備監控線程已建立並啟動 (檢查間隔: {MONITOR_CHECK_INTERVAL_SECONDS}s, 閒置超時: {self.inactivity_timeout}s)")
        
        self._shutdown_flag.clear()
        self._device_monitor_thread = threading.Thread(
            target=self._monitor_model_device_activity, 
            daemon=True
        )
        self._device_monitor_thread.start()
    
    def _monitor_model_device_activity(self):
        """監控模型活動，在閒置時將模型移至 CPU"""
        logger.info("模型設備活動監控線程已啟動")
        
        while not self._shutdown_flag.is_set():
            try:
                # 等待檢查間隔
                if self._shutdown_flag.wait(MONITOR_CHECK_INTERVAL_SECONDS):
                    break  # 收到關閉信號
                
                with self._model_management_lock:
                    if self.model is None:
                        continue
                    
                    try:
                        model_params = list(self.model.parameters())
                    except Exception as e_get_params:
                        logger.error(f"監控線程：調用 model.parameters() 時出錯: {e_get_params}", exc_info=True)
                        continue
                    
                    if not model_params:
                        logger.warning("監控線程：model 沒有參數。跳過設備檢查")
                        continue
                    
                    current_model_device_type = model_params[0].device.type
                    
                    if current_model_device_type == "cuda":
                        current_time = time.time()
                        if (current_time - self.last_model_use_time > self.inactivity_timeout):
                            logger.info(f"模型閒置超過 {self.inactivity_timeout} 秒，準備移至CPU...")
                            try:
                                self.model = self.model.to("cpu")
                                torch.cuda.empty_cache()
                                logger.info("模型已成功移至CPU，GPU記憶體已釋放")
                            except Exception as e_to_cpu:
                                logger.error(f"監控線程：將模型移至CPU時發生錯誤: {e_to_cpu}", exc_info=True)
                        
            except Exception as e_thread_loop:
                logger.error(f"模型設備監控線程主循環發生未預期錯誤: {e_thread_loop}", exc_info=True)
                # 如果出錯，等待更長時間再重試
                if not self._shutdown_flag.wait(MONITOR_CHECK_INTERVAL_SECONDS * 2):
                    continue