# core/model_manager.py
import time
import threading
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextIteratorStreamer
)
import logging
from typing import Optional, Any, Tuple, Dict, Generator

# 從 config.py 導入預設值
from config import (
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    DEFAULT_MONITOR_CHECK_INTERVAL_SECONDS,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_FORCE_CPU_INIT
)

logger = logging.getLogger(__name__)

class ModelManager:
    """LLM 模型管理器"""

    def __init__(self,
                 model_name: Optional[str] = None,
                 inactivity_timeout: Optional[int] = None,
                 monitor_check_interval: Optional[int] = None,
                 load_in_4bit: Optional[bool] = None):
        """
        初始化 ModelManager。

        Args:
            model_name: 要載入的 LLM 模型名稱 (例如 "Qwen/Qwen2-1.5B-Instruct")。
                        如果為 None，則使用 config.py 中的 DEFAULT_LLM_MODEL_NAME。
            inactivity_timeout: GPU 模型閒置多少秒後移至 CPU。
                                如果為 None，則使用 DEFAULT_INACTIVITY_TIMEOUT_SECONDS。
            monitor_check_interval: 監控執行緒檢查模型活動的間隔秒數。
                                     如果為 None，則使用 DEFAULT_MONITOR_CHECK_INTERVAL_SECONDS。
            load_in_4bit: 是否以 4-bit 量化載入模型以節省記憶體。
                          如果為 None，則使用 DEFAULT_LOAD_IN_4BIT。
        """
        # 使用傳入參數或配置檔案的預設值
        self.model_name: str = model_name or DEFAULT_LLM_MODEL_NAME
        self.inactivity_timeout: int = inactivity_timeout if inactivity_timeout is not None else DEFAULT_INACTIVITY_TIMEOUT_SECONDS
        self.monitor_check_interval: int = monitor_check_interval if monitor_check_interval is not None else DEFAULT_MONITOR_CHECK_INTERVAL_SECONDS
        self.load_in_4bit: bool = load_in_4bit if load_in_4bit is not None else DEFAULT_LOAD_IN_4BIT

        logger.info(f"ModelManager initialized with: model_name='{self.model_name}', "
                    f"inactivity_timeout={self.inactivity_timeout}s, "
                    f"monitor_check_interval={self.monitor_check_interval}s, "
                    f"load_in_4bit={self.load_in_4bit}")

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_max_context_length: Optional[int] = None
        self.last_model_use_time: float = 0.0
        self._model_management_lock = threading.RLock()
        self._device_monitor_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

    def _update_activity_and_ensure_monitor(self):
        """更新最後使用時間並確保裝置監控執行緒運行 (如果模型在GPU上)"""
        with self._model_management_lock:
            self.last_model_use_time = time.time()
            if self.model is not None and next(self.model.parameters()).device.type == 'cuda':
                self._ensure_device_monitor_started()

    def initialize(self, force_cpu_init: Optional[bool] = None) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """
        初始化模型。

        Args:
            force_cpu_init: 是否強制在 CPU 上初始化。若為 None，則使用 config.py 的 DEFAULT_FORCE_CPU_INIT。
                            若為 True/False，則覆寫預設值。
        Returns:
            (model, tokenizer, max_context_length) 的元組
        """
        _force_cpu_init = force_cpu_init if force_cpu_init is not None else DEFAULT_FORCE_CPU_INIT

        with self._model_management_lock:
            if self.model is not None and self.tokenizer is not None:
                # logger.debug("LLM 模型和分詞器已載入")
                self._update_activity_and_ensure_monitor()
                return self.model, self.tokenizer, self.model_max_context_length

            logger.info(f"開始初始化 LLM 模型: {self.model_name} (強制CPU: {_force_cpu_init})")

            target_device = "cpu"
            use_quantization = self.load_in_4bit

            if not _force_cpu_init and torch.cuda.is_available():
                target_device = "auto"
                logger.info("CUDA 可用，嘗試將模型載入 GPU。")
            else:
                logger.info("CUDA 不可用或強制使用 CPU，模型將載入 CPU。")
                if _force_cpu_init:
                    logger.info("由於 force_cpu_init=True，即使CUDA可用也將使用CPU。")
                use_quantization = False # CPU模式下通常不使用BitsAndBytes量化
                if self.load_in_4bit and not _force_cpu_init and not torch.cuda.is_available():
                    logger.warning("請求4-bit量化但CUDA不可用，將在CPU上以全精度載入。")


            quantization_config = None
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, # self.load_in_4bit 應該為 True
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("使用 4-bit 量化設定。")


            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=target_device if target_device != "cpu" else None,
                    torch_dtype=torch.float16
                    trust_remote_code=True
                )
                if target_device == "cpu" or _force_cpu_init: # 確保如果目標是CPU，模型最終在CPU
                    self.model = self.model.to("cpu")

                self.model = self.model.eval()

                self.model_max_context_length = getattr(self.model.config, 'max_position_embeddings', None) or \
                                                getattr(self.tokenizer, 'model_max_length', 2048)

                loaded_device_type = next(self.model.parameters()).device.type
                logger.info(f"LLM 模型 ({self.model_name}) 初始化完成。實際載入於: {loaded_device_type}, 上下文長度: {self.model_max_context_length}")

                self._update_activity_and_ensure_monitor()

            except Exception as e:
                logger.critical(f"初始化LLM模型 {self.model_name} 失敗: {e}", exc_info=True)
                self.model, self.tokenizer, self.model_max_context_length = None, None, None
                raise
            return self.model, self.tokenizer, self.model_max_context_length

    def get_model(self, ensure_on_gpu: bool = True) -> Any:
        """獲取模型，自動處理裝置轉移"""
        # 移除了 check_and_reload_if_needed
        with self._model_management_lock:
            if self.model is None:
                logger.info("get_model: 模型尚未初始化，開始進行初始化。")
                # 如果要確保在GPU，則不強制CPU初始化 (除非預設是強制CPU)
                _force_init_cpu = DEFAULT_FORCE_CPU_INIT if not ensure_on_gpu else False
                self.initialize(force_cpu_init=_force_init_cpu)
                if self.model is None:
                    logger.error("get_model: 初始化後模型仍為 None")
                    raise RuntimeError("LLM 模型初始化失敗")

            model_device_type = next(self.model.parameters()).device.type
            if ensure_on_gpu and torch.cuda.is_available():
                if model_device_type == "cpu":
                    try:
                        logger.info("get_model: 模型目前在 CPU，嘗試移至 GPU...")
                        self.model = self.model.to("cuda")
                        logger.info("get_model: 模型已成功移至 GPU。")
                        model_device_type = "cuda"
                    except Exception as e:
                        logger.error(f"get_model: 嘗試將模型移至 GPU 時出錯: {e}", exc_info=True)
            elif not ensure_on_gpu and model_device_type != "cpu":
                if model_device_type == "cuda":
                    try:
                        logger.info("get_model: 模型目前在 GPU，但要求在 CPU，嘗試移至 CPU...")
                        self.model = self.model.to("cpu")
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        logger.info("get_model: 模型已成功移至 CPU。")
                        model_device_type = "cpu"
                    except Exception as e:
                        logger.error(f"get_model: 嘗試將模型移至 CPU 時出錯: {e}", exc_info=True)

            if model_device_type == 'cuda':
                 self._update_activity_and_ensure_monitor()
            return self.model

    def count_tokens(self, text_to_count: str) -> int:
        """計算文本的 token 數量"""
        with self._model_management_lock:
            if self.tokenizer is None:
                logger.info("count_tokens: 分詞器未初始化，嘗試進行初始化 (CPU優先)。")
                self.initialize(force_cpu_init=True)
                if self.tokenizer is None:
                    logger.error("count_tokens: 初始化後分詞器仍為 None")
                    return int(len(text_to_count) * 1.2)
            try:
                return len(self.tokenizer.encode(text_to_count, add_special_tokens=False))
            except Exception as e:
                estimated_tokens = int(len(text_to_count) * 1.2)
                logger.warning(f"count_tokens: 精確計算 token 失敗 ({e})，使用粗略估算: {estimated_tokens}")
                return estimated_tokens

    def generate_response(self, prompt: str, **generation_kwargs: Any) -> str:
        """使用載入的模型生成回應。"""
        with self._model_management_lock:
            model_instance = self.get_model(ensure_on_gpu=True)
            if model_instance is None or self.tokenizer is None:
                logger.error("generate_response: 模型或分詞器尚未初始化。")
                raise RuntimeError("模型或分詞器尚未初始化，無法生成回應。")
            self._update_activity_and_ensure_monitor()
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(model_instance.device) for k, v in inputs.items()}
                if 'pad_token_id' not in generation_kwargs:
                    generation_kwargs['pad_token_id'] = self.tokenizer.eos_token_id
                if 'eos_token_id' not in generation_kwargs:
                    generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
                if 'max_new_tokens' not in generation_kwargs: # 確保有 max_new_tokens
                    generation_kwargs['max_new_tokens'] = 512

                outputs = model_instance.generate(**inputs, **generation_kwargs)
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                self._update_activity_and_ensure_monitor()
                return response
            except Exception as e:
                logger.error(f"generate_response: 生成回應時發生錯誤: {e}", exc_info=True)
                raise

    def generate_stream_response(self, prompt: str, **generation_kwargs: Any) -> Generator[str, None, None]:
        """使用載入的模型以流式方式生成回應。"""
        with self._model_management_lock:
            model_instance = self.get_model(ensure_on_gpu=True)
            if model_instance is None or self.tokenizer is None:
                logger.error("generate_stream_response: 模型或分詞器尚未初始化。")
                raise RuntimeError("模型或分詞器尚未初始化，無法生成流式回應。")
            self._update_activity_and_ensure_monitor()
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(model_instance.device) for k, v in inputs.items()}
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                if 'pad_token_id' not in generation_kwargs:
                    generation_kwargs['pad_token_id'] = self.tokenizer.eos_token_id
                if 'eos_token_id' not in generation_kwargs:
                    generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                if 'max_new_tokens' not in generation_kwargs: # 確保有 max_new_tokens
                    generation_kwargs['max_new_tokens'] = 512

                gen_kwargs_with_streamer = {**inputs, "streamer": streamer, **generation_kwargs}
                thread = threading.Thread(target=model_instance.generate, kwargs=gen_kwargs_with_streamer)
                thread.start()
                for new_text in streamer:
                    self._update_activity_and_ensure_monitor()
                    yield new_text
                thread.join()
            except Exception as e:
                logger.error(f"generate_stream_response: 生成流式回應時發生錯誤: {e}", exc_info=True)
                raise

    def get_status(self) -> dict:
        """獲取模型狀態"""
        with self._model_management_lock:
            monitor_alive = self._device_monitor_thread.is_alive() if self._device_monitor_thread else False
            if self.model is None:
                return {
                    "model_name": self.model_name, "initialized": False, "current_device": None,
                    "device_details": None, "last_used": None, "gpu_available": torch.cuda.is_available(),
                    "gpu_memory": None, "max_context_length": None,
                    "monitor_thread_alive": monitor_alive,
                    "load_in_4bit_setting": self.load_in_4bit,
                    "inactivity_timeout_setting": self.inactivity_timeout,
                    "monitor_check_interval_setting": self.monitor_check_interval
                }
            try:
                current_device = next(self.model.parameters()).device
                device_type = current_device.type
                device_details = {'type': device_type}
                if device_type == 'cuda':
                    device_details['index'] = current_device.index if hasattr(current_device, 'index') else 0
                gpu_memory_info = None
                if device_type == 'cuda' and torch.cuda.is_available():
                    try:
                        allocated = round(torch.cuda.memory_allocated(current_device) / (1024 ** 3), 2)
                        reserved = round(torch.cuda.memory_reserved(current_device) / (1024 ** 3), 2)
                        total_mem = round(torch.cuda.get_device_properties(current_device).total_memory / (1024**3), 2)
                        gpu_memory_info = {'allocated': allocated, 'reserved': reserved, 'total': total_mem, 'unit': 'GB'}
                    except Exception as e_mem: logger.warning(f"get_status: 獲取 GPU 記憶體信息時出錯: {e_mem}")
            except Exception as e_stat:
                logger.error(f"get_status: 獲取模型設備時出錯: {e_stat}")
                device_type, device_details, gpu_memory_info = "unknown", {'type': 'unknown'}, None
            return {
                "model_name": self.model_name, "initialized": True, "current_device": device_type,
                "device_details": device_details, "last_used": self.last_model_use_time,
                "gpu_available": torch.cuda.is_available(), "gpu_memory": gpu_memory_info,
                "max_context_length": self.model_max_context_length,
                "monitor_thread_alive": monitor_alive,
                "load_in_4bit_setting": self.load_in_4bit,
                "inactivity_timeout_setting": self.inactivity_timeout,
                "monitor_check_interval_setting": self.monitor_check_interval
            }

    def shutdown(self): # 簡化 shutdown，不再有 full_shutdown 參數
        """釋放所有相關資源"""
        with self._model_management_lock:
            logger.info("開始釋放 LLM 相關資源...")
            self._shutdown_flag.set()

            if self._device_monitor_thread is not None and self._device_monitor_thread.is_alive():
                logger.info("等待設備監控線程結束...")
                self._device_monitor_thread.join(timeout=self.monitor_check_interval + 1)
                if self._device_monitor_thread.is_alive():
                    logger.warning("設備監控線程未能正常結束。")
            self._device_monitor_thread = None

            if self.model is not None:
                try:
                    if next(self.model.parameters()).device.type != "cpu":
                        logger.info("將 LLM 模型移至 CPU...")
                        self.model = self.model.to("cpu")
                except Exception as e: logger.warning(f"關閉前將模型移至CPU失敗: {e}")
                del self.model; self.model = None
                logger.info("LLM 模型已卸載")

            if self.tokenizer is not None:
                del self.tokenizer; self.tokenizer = None
                logger.info("分詞器已卸載")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU 快取已清理")

            self.model_max_context_length = None
            self.last_model_use_time = 0
            self._shutdown_flag.clear() # 為下次可能的啟動做準備
            logger.info("LLM 相關資源已釋放。")

    def _ensure_device_monitor_started(self):
        """確保設備監控線程已啟動 (如果模型在 GPU 上)"""
        with self._model_management_lock: # 確保在此方法內部訪問 self.model 等是線程安全的
            if self.model is None or next(self.model.parameters()).device.type != 'cuda':
                if self._device_monitor_thread is not None and self._device_monitor_thread.is_alive():
                    logger.debug("模型不在 GPU 上，但監控執行緒仍在運行。將其停止。")
                    self._shutdown_flag.set()
                    self._device_monitor_thread.join(timeout=self.monitor_check_interval + 1) # 給予時間結束
                    if self._device_monitor_thread.is_alive(): # 再次檢查
                         logger.warning("_ensure_device_monitor_started: 監控線程未能停止。")
                    self._device_monitor_thread = None
                    self._shutdown_flag.clear() # 清除標誌，以備將來使用
                return

            if self._device_monitor_thread is not None and self._device_monitor_thread.is_alive():
                return

            if self._shutdown_flag.is_set(): # 如果之前被要求關閉，現在要重啟，則清除標誌
                self._shutdown_flag.clear()

            logger.info(f"設備監控線程準備啟動 (檢查間隔: {self.monitor_check_interval}s, GPU閒置超時: {self.inactivity_timeout}s)")
            self._device_monitor_thread = threading.Thread(
                target=self._monitor_model_device_activity, daemon=True
            )
            self._device_monitor_thread.start()

    def _monitor_model_device_activity(self):
        """監控模型活動，在閒置時將模型移至 CPU"""
        logger.info("模型設備活動監控線程已啟動。")
        while not self._shutdown_flag.wait(self.monitor_check_interval):
            with self._model_management_lock:
                if self.model is None: continue
                try:
                    current_model_device_type = next(self.model.parameters()).device.type
                except Exception as e_get_params:
                    logger.error(f"監控線程：獲取模型設備類型時出錯: {e_get_params}", exc_info=True)
                    continue
                if current_model_device_type == "cuda":
                    current_time = time.time()
                    idle_time = current_time - self.last_model_use_time
                    if idle_time > self.inactivity_timeout:
                        logger.info(f"模型在 GPU 上閒置超過 {self.inactivity_timeout} 秒 ({idle_time:.2f}s)，準備移至CPU...")
                        try:
                            self.model = self.model.to("cpu")
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            logger.info("模型已成功移至CPU，GPU記憶體已釋放。")
                        except Exception as e_to_cpu:
                            logger.error(f"監控線程：將模型移至CPU時發生錯誤: {e_to_cpu}", exc_info=True)
        logger.info("模型設備活動監控線程已停止。")