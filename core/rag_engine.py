# core/rag_engine.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
import re

from config import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_TEMPERATURE,
    DEFAULT_MAX_TEMPERATURE,
    DEFAULT_MIN_MAX_NEW_TOKENS,
    DEFAULT_MAX_MAX_NEW_TOKENS
)

logger = logging.getLogger(__name__)

class RAGEngine:
    """檢索增強生成 (RAG) 引擎"""

    def __init__(self,
                 model_manager, # 傳入 ModelManager 的實例
                 db_manager,    # 傳入 DBManager 的實例
                 system_prompt: Optional[str] = None,
                 default_temperature: Optional[float] = None,
                 default_max_new_tokens: Optional[int] = None,
                 min_temperature: Optional[float] = None,
                 max_temperature: Optional[float] = None,
                 min_max_new_tokens: Optional[int] = None,
                 max_max_new_tokens: Optional[int] = None):
        """
        初始化 RAGEngine。

        Args:
            model_manager: ModelManager 的實例。
            db_manager: DBManager 的實例。
            system_prompt: 預設的系統提示詞。
            default_temperature: 生成時的預設溫度。
            default_max_new_tokens: 生成時預設的最大新 token 數。
            min_temperature: 允許的最小溫度。
            max_temperature: 允許的最大溫度。
            min_max_new_tokens: 允許的最小 max_new_tokens。
            max_max_new_tokens: 允許的最大 max_new_tokens。
        """
        self.model_manager = model_manager
        self.db_manager = db_manager

        # 使用傳入參數或配置檔案的預設值
        self.system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.default_temperature: float = default_temperature if default_temperature is not None else DEFAULT_TEMPERATURE
        self.default_max_new_tokens: int = default_max_new_tokens if default_max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
        
        self.min_temperature: float = min_temperature if min_temperature is not None else DEFAULT_MIN_TEMPERATURE
        self.max_temperature: float = max_temperature if max_temperature is not None else DEFAULT_MAX_TEMPERATURE
        self.min_max_new_tokens: int = min_max_new_tokens if min_max_new_tokens is not None else DEFAULT_MIN_MAX_NEW_TOKENS
        self.max_max_new_tokens: int = max_max_new_tokens if max_max_new_tokens is not None else DEFAULT_MAX_MAX_NEW_TOKENS

        logger.info(f"RAGEngine initialized with: default_temp={self.default_temperature}, "
                    f"default_max_tokens={self.default_max_new_tokens}")

        self._last_rag_resources: List[Dict[str, Any]] = []

    @property
    def model_max_context_length(self) -> int:
        """獲取模型最大上下文長度"""
        if self.model_manager.model_max_context_length is None:
            # 嘗試初始化以獲取 (通常 tokenizer 初始化即可)
            _, _, max_len = self.model_manager.initialize(force_cpu_init=True)
            return max_len if max_len is not None else 2048 # 提供一個最終備用值
        return self.model_manager.model_max_context_length

    def _validate_parameters(self, temperature: float, max_new_tokens: int) -> Tuple[float, int]:
        """驗證並調整參數範圍"""
        # 使用實例變數中儲存的範圍限制
        temperature = max(self.min_temperature, min(self.max_temperature, temperature))
        max_new_tokens = max(self.min_max_new_tokens, min(self.max_max_new_tokens, max_new_tokens))
        return temperature, max_new_tokens

    def build_memory(self, history: List[Dict[str, str]], enable: bool = True,
                     base_token_count: int = 0, reserve_for_context_and_query: int = 1000) -> str:
        """構建歷史對話字符串"""
        if not enable or not history:
            return ""

        # 確保 model_max_context_length 是最新的
        max_context = self.model_max_context_length
        available_tokens = max_context - base_token_count - reserve_for_context_and_query

        if available_tokens <= 0:
            logger.warning("build_memory: No available tokens for history.")
            return ""

        memory_parts = []
        current_token_count = 0

        for item in reversed(history):
            role = item.get("role", "")
            content = item.get("content", "")

            if role == "user":
                formatted_entry = f"用戶: {content}\n"
            elif role == "assistant":
                formatted_entry = f"助手: {content}\n"
            else:
                continue

            estimated_tokens = self.model_manager.count_tokens(formatted_entry)

            if current_token_count + estimated_tokens > available_tokens:
                break

            memory_parts.insert(0, formatted_entry)
            current_token_count += estimated_tokens

        if memory_parts:
            return "### 對話歷史\n" + "".join(memory_parts) + "\n"
        return ""

    def process_search_results(self, query: str, source_files: Optional[List[str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """處理搜索結果，返回上下文和源資料"""
        try:
            # 假設 db_manager.search 接受 query 和 source_files
            results = self.db_manager.search(query=query, source_files=source_files)
            if not results:
                self._last_rag_resources = []
                return "", []

            context_parts = []
            resources = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                metadata = doc.metadata
                source_info = self._format_source_info(metadata)
                resource = {
                    'index': i,
                    'source_file_name': metadata.get('source_file_name', '未知文檔'),
                    'page': metadata.get('page'),
                    'chunk_index': metadata.get('chunk_index'), # 假設有 chunk_index
                    'formatted_info': source_info,
                    # 'score': metadata.get('score') # 如果您的 search result 包含 score
                }
                resources.append(resource)
                display_content = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"**文檔 {i}** ({source_info}):\n{display_content}")

            context = "\n\n".join(context_parts)
            self._last_rag_resources = resources
            return context, resources
        except Exception as e:
            logger.error(f"處理搜索結果出錯: {e}", exc_info=True)
            self._last_rag_resources = []
            return "", []

    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """格式化源文件信息"""
        parts = []
        if 'source_file_name' in metadata: parts.append(f"來源: {metadata['source_file_name']}")
        if 'page' in metadata and metadata['page'] is not None: parts.append(f"第 {metadata['page']} 頁")
        if 'chunk_index' in metadata and metadata['chunk_index'] is not None: parts.append(f"段落 {metadata['chunk_index']}")
        return " | ".join(parts) if parts else "未知來源"

    def _build_prompt(self, query: str, use_rag: bool = True, enable_memory: bool = True,
                     history: Optional[List[Dict[str, str]]] = None,
                     source_files: Optional[List[str]] = None) -> str:
        """構建完整的提示詞 (移除 max_new_tokens_for_reply 參數，因為它不影響 prompt 本身)"""
        if history is None: history = []

        # 1. 系統提示詞 - 使用實例變數
        prompt_parts = [self.system_prompt]
        base_tokens = self.model_manager.count_tokens(self.system_prompt)

        # 2. RAG 檢索上下文
        rag_context_str = ""
        if use_rag:
            rag_context_str, _ = self.process_search_results(query, source_files)
            if rag_context_str:
                rag_section = f"### 相關資料\n{rag_context_str}"
                prompt_parts.append(rag_section)
                base_tokens += self.model_manager.count_tokens(rag_section)

        # 3. 對話歷史
        # reserve_for_context_and_query: 給予問題、回答指示詞、以及一些緩衝的 token 數量
        # 您可能需要根據模型的特性微調這個保留值
        # 這裡的 1000 是一個較大的估計值，可能需要根據實際情況調整
        # 例如，如果 query 很長，這裡的 reserve 可能需要更大
        query_tokens = self.model_manager.count_tokens(query)
        # 粗略估計回答指示詞的 token（如 "### 回答\n"）
        answer_instruction_tokens = self.model_manager.count_tokens("\n\n### 用戶問題\n\n### 回答\n")
        reserve_tokens = query_tokens + answer_instruction_tokens + 200 # 200作為緩衝和其他固定部分的 token

        memory_str = self.build_memory(history, enable_memory, base_tokens, reserve_tokens)
        if memory_str:
            prompt_parts.append(memory_str)

        # 4. 當前問題
        question_section = f"### 用戶問題\n{query}\n\n### 回答\n"
        prompt_parts.append(question_section)

        full_prompt = "\n\n".join(filter(None, prompt_parts)) # 使用 filter(None, ...) 避免空部分產生多餘換行

        # 檢查總 token 是否超出限制 (可選，但建議)
        total_prompt_tokens = self.model_manager.count_tokens(full_prompt)
        max_len = self.model_max_context_length
        # 留一些空間給模型的輸出
        # 如果 max_new_tokens 是固定的，可以從 max_len 中減去它
        # 這裡我們假設至少要留 DEFAULT_MAX_NEW_TOKENS 的空間
        available_for_prompt = max_len - self.default_max_new_tokens # 或一個更動態的值
        
        if total_prompt_tokens > available_for_prompt :
            logger.warning(
                f"構建的 Prompt token 數量 ({total_prompt_tokens}) 可能超過模型單次輸入上限 "
                f"(上下文長度 {max_len} - 預期輸出 {self.default_max_new_tokens} = {available_for_prompt})。"
                "可能會導致截斷或錯誤。考慮縮短歷史、RAG內容或問題。"
            )
            # 這裡可以加入更積極的截斷邏輯，例如從 prompt_parts 中移除最不重要的部分
            # 但這會讓邏輯更複雜，目前僅作警告

        logger.debug(f"Prompt 構建完成, RAG={'啟用' if use_rag and rag_context_str else '停用或無結果'}, "
                     f"記憶={'啟用' if enable_memory and memory_str else '停用或無內容'}")
        logger.debug(f"完整 Prompt (前200字符): {full_prompt[:200]}...")
        return full_prompt

    def get_rag_info(self, query: str, source_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """獲取 RAG 檢索信息"""
        context, resources = self.process_search_results(query, source_files)
        return {'context': context, 'resources': resources, 'total_found': len(resources), 'query': query}

    def generate(self, query: str, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None,
                 use_rag: bool = True, enable_memory: bool = True, history: Optional[List[Dict[str, str]]] = None,
                 source_files: Optional[List[str]] = None,
                 generation_params: Optional[Dict[str, Any]] = None) -> str:
        """生成回應"""
        # 使用實例的預設值或傳入的參數
        current_temp = temperature if temperature is not None else self.default_temperature
        current_max_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        current_temp, current_max_tokens = self._validate_parameters(current_temp, current_max_tokens)

        prompt = self._build_prompt(query, use_rag, enable_memory, history, source_files)
        final_generation_params = {
            "temperature": current_temp,
            "max_new_tokens": current_max_tokens,
            "do_sample": True if current_temp > 0.001 else False, # 接近0也認為是greedy
            **(generation_params or {})
        }
        try:
            response = self.model_manager.generate_response(prompt=prompt, **final_generation_params)
            return self._post_process_response(response)
        except Exception as e:
            logger.error(f"RAGEngine.generate: 調用 ModelManager 生成回應失敗: {e}", exc_info=True)
            return self._post_process_response(f"抱歉，處理您的請求時發生錯誤: {type(e).__name__}")

    def stream(self, query: str, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None,
               use_rag: bool = True, enable_memory: bool = True, history: Optional[List[Dict[str, str]]] = None,
               source_files: Optional[List[str]] = None,
               generation_params: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """流式生成回應"""
        current_temp = temperature if temperature is not None else self.default_temperature
        current_max_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        current_temp, current_max_tokens = self._validate_parameters(current_temp, current_max_tokens)

        prompt = self._build_prompt(query, use_rag, enable_memory, history, source_files)
        final_generation_params = {
            "temperature": current_temp,
            "max_new_tokens": current_max_tokens,
            "do_sample": True if current_temp > 0.001 else False,
            **(generation_params or {})
        }
        try:
            full_response = []
            for new_text in self.model_manager.generate_stream_response(prompt=prompt, **final_generation_params):
                full_response.append(new_text)
                yield new_text
            # 流結束後，可以選擇是否對 full_response 做一次 _post_process_response
            # 但由於是流式，通常在客戶端組裝後再處理，或者每個 chunk 單獨處理（如果需要）
            # logger.debug(f"Streamed full response (before post-processing): {''.join(full_response)}")
        except Exception as e:
            logger.error(f"RAGEngine.stream: 調用 ModelManager 生成流式回應失敗: {e}", exc_info=True)
            yield self._post_process_response(f"抱歉，處理您的流式請求時發生錯誤: {type(e).__name__}")


    def get_last_rag_resources(self) -> List[Dict[str, Any]]:
        """獲取最後一次生成時使用的 RAG 資源"""
        return self._last_rag_resources.copy() # 回傳副本以避免外部修改

    def _post_process_response(self, response: str) -> str:
        """後處理回應"""
        if not response:
            return "抱歉，我無法為這個問題提供有效的回覆。"
        cleaned = re.sub(r'\n{3,}', '\n\n', response) # 清理多餘的換行
        cleaned = re.sub(r' {2,}', ' ', cleaned)      # 清理多餘的空格
        cleaned = cleaned.strip()
        return cleaned or "抱歉，我無法為這個問題提供有效的回覆。" # 如果清理後為空
