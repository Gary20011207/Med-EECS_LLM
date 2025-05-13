# core/rag_engine.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator, Union
from datetime import datetime
import threading
import re
from transformers import TextIteratorStreamer

# 導入配置
try:
    from config import (
        SYSTEM_PROMPT,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_NEW_TOKENS,
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
        MIN_MAX_NEW_TOKENS,
        MAX_MAX_NEW_TOKENS
    )
except ImportError:
    # 預設配置
    SYSTEM_PROMPT = """您是一個專業的醫療助手，專門協助醫護人員解答關於 ERAS (Enhanced Recovery After Surgery) 手術加速康復計劃的問題。

您的任務是基於提供的 ERAS 指引文件，為醫護人員提供準確、專業的回答。如果問題超出您的知識範圍或提供的文件內容，請誠實告知您無法回答該問題，並建議咨詢專業醫療人員。

請保持回答簡潔明了，並在適當時引用相關指引。"""
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_NEW_TOKENS = 1000
    MIN_TEMPERATURE = 0.1
    MAX_TEMPERATURE = 2.0
    MIN_MAX_NEW_TOKENS = 50
    MAX_MAX_NEW_TOKENS = 4000

logger = logging.getLogger(__name__)

class RAGEngine:
    """檢索增強生成 (RAG) 引擎"""

    def __init__(self, model_manager, db_manager):
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.system_prompt = SYSTEM_PROMPT
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self._lock = threading.Lock()
        self._last_rag_resources = []  # 保存源資料元數據
    
    @property
    def model(self):
        """獲取模型，確保在 GPU 上"""
        return self.model_manager.get_model(ensure_on_gpu=True)
    
    @property
    def tokenizer(self):
        """獲取分詞器"""
        return self.model_manager.tokenizer
    
    @property
    def model_max_context_length(self):
        """獲取模型最大上下文長度"""
        return self.model_manager.model_max_context_length
    
    def _validate_parameters(self, temperature: float, max_new_tokens: int) -> Tuple[float, int]:
        """驗證並調整參數範圍"""
        temperature = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, temperature))
        max_new_tokens = max(MIN_MAX_NEW_TOKENS, min(MAX_MAX_NEW_TOKENS, max_new_tokens))
        return temperature, max_new_tokens
    
    def build_memory(self, history: List[Dict[str, str]], enable: bool = True, 
                     base_token_count: int = 0, reserve_for_context_and_query: int = 1000) -> str:
        """構建歷史對話字符串"""
        if not enable or not history:
            return ""
        
        max_context = self.model_max_context_length
        available_tokens = max_context - base_token_count - reserve_for_context_and_query
        
        if available_tokens <= 0:
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
        """處理搜索結果，返回上下文和源資料
        
        Args:
            query: 查詢文本
            source_files: 指定要搜尋的源文件
            
        Returns:
            (context: 格式化的上下文字符串, resources: 源資料元數據列表)
        """
        try:
            results = self.db_manager.search(query=query, source_files=source_files)
            
            if not results:
                # 即使沒有結果也要保存空列表
                self._last_rag_resources = []
                return "", []
            
            context_parts = []
            resources = []
            
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                metadata = doc.metadata
                
                # 格式化源信息
                source_info = self._format_source_info(metadata)
                
                # 保存資源元數據
                resource = {
                    'index': i,
                    'source_file_name': metadata.get('source_file_name', '未知文檔'),
                    'page': metadata.get('page'),
                    'chunk_index': metadata.get('chunk_index'),
                    'formatted_info': source_info
                }
                resources.append(resource)
                
                # 截斷內容用於上下文
                display_content = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"**文檔 {i}** ({source_info}):\n{display_content}")
            
            context = "\n\n".join(context_parts)
            
            # 保存最後的資源 (所有方法共用)
            self._last_rag_resources = resources
            
            return context, resources
            
        except Exception as e:
            logger.error(f"處理搜索結果出錯: {e}")
            # 出錯時也要保存空列表
            self._last_rag_resources = []
            return "", []
    
    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """格式化源文件信息"""
        parts = []
        
        if 'source_file_name' in metadata:
            parts.append(f"來源: {metadata['source_file_name']}")
        
        if 'page' in metadata and metadata['page'] is not None:
            parts.append(f"第 {metadata['page']} 頁")
        
        if 'chunk_index' in metadata and metadata['chunk_index'] is not None:
            parts.append(f"段落 {metadata['chunk_index']}")
        
        return " | ".join(parts) if parts else "未知來源"
    
    def _build_prompt(self, query: str, use_rag: bool = True, enable_memory: bool = True,
                     history: List[Dict[str, str]] = None, max_new_tokens_for_reply: int = None,
                     source_files: Optional[List[str]] = None) -> str:
        """構建完整的提示詞"""
        if history is None:
            history = []
        
        if max_new_tokens_for_reply is None:
            max_new_tokens_for_reply = self.default_max_new_tokens
        
        # 1. 系統提示詞
        prompt_parts = [self.system_prompt]
        base_tokens = self.model_manager.count_tokens(self.system_prompt)
        
        # 2. RAG 檢索上下文
        if use_rag:
            context, _ = self.process_search_results(query, source_files)
            if context:
                rag_section = f"### 相關資料\n{context}"
                prompt_parts.append(rag_section)
                base_tokens += self.model_manager.count_tokens(rag_section)
        
        # 3. 對話歷史
        memory = self.build_memory(history, enable_memory, base_tokens, 1000)
        if memory:
            prompt_parts.append(memory)
        
        # 4. 當前問題
        question_section = f"### 用戶問題\n{query}\n\n### 回答\n"
        prompt_parts.append(question_section)
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Prompt 構建完成, 長度={len(full_prompt)}, RAG={use_rag}, 記憶={enable_memory and bool(memory)}")
        
        return full_prompt
    
    def get_rag_info(self, query: str, source_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """獲取 RAG 檢索信息"""
        context, resources = self.process_search_results(query, source_files)
        return {
            'context': context,
            'resources': resources,
            'total_found': len(resources),
            'query': query
        }
    
    def generate(self, query: str, temperature: float = None, max_new_tokens: int = None,
                 use_rag: bool = True, enable_memory: bool = True, history: List[Dict[str, str]] = None,
                 source_files: Optional[List[str]] = None) -> str:
        """生成回應"""
        if temperature is None:
            temperature = self.default_temperature
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        temperature, max_new_tokens = self._validate_parameters(temperature, max_new_tokens)
        prompt = self._build_prompt(query, use_rag, enable_memory, history, max_new_tokens, source_files)

        with self._lock:
            model = self.model
            tokenizer = self.tokenizer
            
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return self._post_process_response(response)

    def stream(self, query: str, temperature: float = None, max_new_tokens: int = None,
               use_rag: bool = True, enable_memory: bool = True, history: List[Dict[str, str]] = None,
               source_files: Optional[List[str]] = None) -> Generator[str, None, None]:
        """流式生成回應"""
        if temperature is None:
            temperature = self.default_temperature
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        temperature, max_new_tokens = self._validate_parameters(temperature, max_new_tokens)
        prompt = self._build_prompt(query, use_rag, enable_memory, history, max_new_tokens, source_files)

        with self._lock:
            model = self.model
            tokenizer = self.tokenizer
            
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "streamer": streamer,
            }

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join()
    
    def get_last_rag_resources(self) -> List[Dict[str, Any]]:
        """獲取最後一次生成時使用的 RAG 資源"""
        return self._last_rag_resources.copy()
    
    def _post_process_response(self, response: str) -> str:
        """後處理回應"""
        if not response:
            return "抱歉，我無法為這個問題提供有效的回覆。"
        
        # 清理多餘的換行和空格
        cleaned = re.sub(r'\n{3,}', '\n\n', response)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned or "抱歉，我無法為這個問題提供有效的回覆。"