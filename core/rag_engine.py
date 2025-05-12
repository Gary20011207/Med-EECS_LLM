# core/rag_engine.py
import logging
from typing import List, Dict, Any, Union, Optional, Generator, Tuple
import json
import re
from datetime import datetime
import threading

from langchain_core.documents import Document

# 導入配置
try:
    from config import (
        SYSTEM_PROMPT,
        RAG_TOP_K,
        LLM_MODEL_NAME,
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
    RAG_TOP_K = 5
    LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
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
        """初始化 RAG 引擎
        
        Args:
            model_manager: LLM 模型管理器實例
            db_manager: 向量資料庫管理器實例
        """
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.top_k = RAG_TOP_K
        self.system_prompt = SYSTEM_PROMPT
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self._lock = threading.Lock()
        
        logger.info("RAG 引擎初始化完成")
        logger.info(f"  系統提示詞長度: {len(self.system_prompt)} 字符")
        logger.info(f"  預設檢索個數: {self.top_k}")
        logger.info(f"  預設溫度: {self.default_temperature}")
        logger.info(f"  預設最大 Token 數: {self.default_max_new_tokens}")
    
    def _validate_parameters(self, temperature: float, max_new_tokens: int) -> Tuple[float, int]:
        """驗證並調整參數範圍
        
        Args:
            temperature: 溫度參數
            max_new_tokens: 最大新生成 token 數
            
        Returns:
            調整後的 (溫度, 最大 token 數)
        """
        # 驗證溫度範圍
        if temperature < MIN_TEMPERATURE:
            logger.warning(f"溫度參數 {temperature} 小於最小值 {MIN_TEMPERATURE}，將調整為 {MIN_TEMPERATURE}")
            temperature = MIN_TEMPERATURE
        elif temperature > MAX_TEMPERATURE:
            logger.warning(f"溫度參數 {temperature} 大於最大值 {MAX_TEMPERATURE}，將調整為 {MAX_TEMPERATURE}")
            temperature = MAX_TEMPERATURE
        
        # 驗證 token 數範圍
        if max_new_tokens < MIN_MAX_NEW_TOKENS:
            logger.warning(f"最大 token 數 {max_new_tokens} 小於最小值 {MIN_MAX_NEW_TOKENS}，將調整為 {MIN_MAX_NEW_TOKENS}")
            max_new_tokens = MIN_MAX_NEW_TOKENS
        elif max_new_tokens > MAX_MAX_NEW_TOKENS:
            logger.warning(f"最大 token 數 {max_new_tokens} 大於最大值 {MAX_MAX_NEW_TOKENS}，將調整為 {MAX_MAX_NEW_TOKENS}")
            max_new_tokens = MAX_MAX_NEW_TOKENS
        
        return temperature, max_new_tokens
    
    def build_memory(self, 
                    history: List[Dict[str, str]], 
                    enable: bool = True,
                    base_token_count: int = 0,
                    reserve_for_context_and_query: int = 1000) -> str:
        """構建歷史對話字符串
        
        注意：此函數不構建新的歷史記錄，而是將現有的歷史記錄格式化為可用於提示詞的字符串。
        歷史記錄的管理（添加、刪除）由前端處理。
        
        Args:
            history: 對話歷史列表，格式為 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            enable: 是否啟用歷史記憶
            base_token_count: 基礎 token 數量（系統提示和其他固定內容）
            reserve_for_context_and_query: 為上下文和查詢保留的 token 數
            
        Returns:
            格式化的歷史對話字符串
        """
        if not enable or not history:
            logger.debug("歷史記憶已禁用或無歷史記錄")
            return ""
        
        # 獲取模型的最大上下文長度
        try:
            # 處理不同的返回值格式
            result = self.model_manager.get_model_and_tokenizer(
                update_last_used_time=False,
                ensure_on_gpu=False
            )
            
            if isinstance(result, tuple):
                if len(result) >= 2:
                    model = result[0]
                    tokenizer = result[1]
                    # 如果有第三個值，忽略它
                else:
                    logger.warning(f"get_model_and_tokenizer 返回的元組長度不足: {len(result)}")
                    model = result[0] if len(result) > 0 else None
                    tokenizer = None
            else:
                model = result
                tokenizer = None
            
            if model and hasattr(model, 'config'):
                max_context_length = getattr(model.config, 'max_position_embeddings', 4096)
            else:
                max_context_length = 4096
                
        except Exception as e:
            logger.warning(f"無法獲取模型配置，使用預設最大上下文長度: {e}")
            max_context_length = 4096
        
        # 計算可用於歷史記憶的 token 數
        available_tokens = max_context_length - base_token_count - reserve_for_context_and_query
        
        if available_tokens <= 0:
            logger.warning("可用 token 數不足，跳過歷史記憶")
            return ""
        
        # 從最新的對話開始，逐步添加歷史對話
        memory_str = ""
        current_token_count = 0
        
        # 倒序處理歷史，優先保留最近的對話
        for item in reversed(history):
            role = item.get("role", "")
            content = item.get("content", "")
            
            if role == "user":
                formatted_entry = f"用戶: {content}\n"
            elif role == "assistant":
                formatted_entry = f"助手: {content}\n"
            else:
                logger.debug(f"跳過未知角色: {role}")
                continue
            
            # 估算 token 數（粗略估計：中文 1個字約 1.5 token，英文 1個詞約 1 token）
            estimated_tokens = len(formatted_entry) * 1.2
            
            if current_token_count + estimated_tokens > available_tokens:
                logger.debug(f"歷史記憶 token 數達到上限，停止添加")
                break
            
            memory_str = formatted_entry + memory_str
            current_token_count += estimated_tokens
        
        if memory_str:
            memory_str = "### 對話歷史\n" + memory_str + "\n"
        
        logger.debug(f"歷史記憶構建完成: 長度={len(memory_str)}字符, 估計token數={current_token_count}")
        return memory_str
    
    def _build_prompt(self,
                     query: str,
                     use_rag: bool = True,
                     enable_memory: bool = True,
                     history: List[Dict[str, str]] = None,
                     max_new_tokens_for_reply: int = None,
                     source_files: Optional[List[str]] = None) -> Tuple[str, str]:
        """構建完整的提示詞
        
        Args:
            query: 用戶問題
            use_rag: 是否使用 RAG 檢索
            enable_memory: 是否啟用對話歷史
            history: 對話歷史
            max_new_tokens_for_reply: 最大新生成 token 數
            source_files: 指定要搜索的源文件列表
            
        Returns:
            (完整提示詞, 上下文字符串)
        """
        if history is None:
            history = []
        
        if max_new_tokens_for_reply is None:
            max_new_tokens_for_reply = self.default_max_new_tokens
        
        # 1. 系統提示詞
        prompt_parts = [self.system_prompt]
        base_length = len(self.system_prompt)
        
        # 2. RAG 檢索上下文
        context_str = ""
        if use_rag:
            context_str = self._get_rag_context(query, source_files)
            if context_str:
                prompt_parts.append(f"### 相關資料\n{context_str}")
        
        # 3. 對話歷史
        current_length = sum(len(part) for part in prompt_parts)
        memory_str = self.build_memory(
            history=history,
            enable=enable_memory,
            base_token_count=current_length * 1.2,  # 粗略估計 token 數
            reserve_for_context_and_query=1000
        )
        
        if memory_str:
            prompt_parts.append(memory_str)
        
        # 4. 當前問題
        prompt_parts.append(f"### 用戶問題\n{query}\n\n### 回答\n")
        
        full_prompt = "\n".join(prompt_parts)
        
        # 日誌記錄
        estimated_total_tokens = len(full_prompt) * 1.2
        logger.debug(f"提示詞構建完成:")
        logger.debug(f"  使用 RAG: {use_rag}")
        logger.debug(f"  包含歷史: {enable_memory and bool(memory_str)}")
        logger.debug(f"  總長度: {len(full_prompt)} 字符")
        logger.debug(f"  估計 token 數: {estimated_total_tokens}")
        
        return full_prompt, context_str
    
    def _get_rag_context(self, query: str, source_files: Optional[List[str]] = None) -> str:
        """從向量資料庫檢索相關上下文
        
        Args:
            query: 查詢問題
            source_files: 指定要搜索的源文件列表
            
        Returns:
            格式化的上下文字符串
        """
        try:
            # 檢查資料庫連接
            if not self.db_manager:
                logger.warning("資料庫管理器不存在，跳過 RAG 檢索")
                return ""
            
            # 檢查資料庫狀態
            db_status = self.db_manager.get_status()
            if not db_status.get('db_connected', False):
                logger.warning("向量資料庫未連接，跳過 RAG 檢索")
                return ""
            
            # 搜索相關文檔
            results = self.db_manager.search(
                query=query,
                k=self.top_k,
                source_files=source_files
            )
            
            if not results:
                logger.info(f"RAG 檢索未找到相關內容: '{query[:50]}...'")
                return ""
            
            # 格式化檢索結果
            context_parts = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                source_info = self._extract_source_info(doc.metadata)
                
                # 限制每個文檔片段的長度，避免過長
                if len(content) > 500:
                    content = content[:500] + "..."
                
                context_parts.append(f"**文檔 {i}** ({source_info}):\n{content}")
            
            context_str = "\n\n".join(context_parts)
            logger.debug(f"RAG 檢索成功，返回 {len(results)} 個相關文檔")
            
            # 記錄使用的源文件（如果有指定）
            if source_files:
                logger.debug(f"指定源文件搜索: {source_files}")
            
            return context_str
            
        except Exception as e:
            logger.error(f"RAG 檢索出錯: {e}", exc_info=True)
            return ""
    
    def _extract_source_info(self, metadata: Dict[str, Any]) -> str:
        """從元數據提取源文件信息"""
        source_info_parts = []
        
        # 文件名
        if 'source_pdf' in metadata:
            source_info_parts.append(f"來源: {metadata['source_pdf']}")
        elif 'source_file_name' in metadata:
            source_info_parts.append(f"來源: {metadata['source_file_name']}")
        
        # 頁碼信息
        if 'page' in metadata:
            source_info_parts.append(f"第 {metadata['page']} 頁")
        
        # 文本塊索引
        if 'chunk_index' in metadata:
            source_info_parts.append(f"段落 {metadata['chunk_index']}")
        
        return " | ".join(source_info_parts) if source_info_parts else "未知來源"
    
    def generate_reply(self,
                      query: str,
                      use_rag: bool = True,
                      enable_memory: bool = True,
                      history: List[Dict[str, str]] = None,
                      max_new_tokens: int = None,
                      temperature: float = None,
                      source_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成非串流回覆
        
        Args:
            query: 用戶問題
            use_rag: 是否使用 RAG 檢索
            enable_memory: 是否啟用對話歷史
            history: 對話歷史
            max_new_tokens: 最大新生成 token 數 (None 表示使用預設值)
            temperature: 溫度參數 (None 表示使用預設值)
            source_files: 指定要搜索的源文件列表
            
        Returns:
            包含回覆和元數據的字典
        """
        if history is None:
            history = []
        
        # 驗證 query
        if not query or not query.strip():
            logger.warning("收到空的查詢請求")
            return {
                "response": "抱歉，請提供一個有效的問題。",
                "used_rag": use_rag,
                "context_provided": None,
                "generation_time": 0.0,
                "error": "Empty query provided",
                "temperature": temperature or self.default_temperature,
                "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
                "source_files": source_files,
                "timestamp": datetime.now().isoformat()
            }
        
        # 使用預設值或驗證參數
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        temperature, max_new_tokens = self._validate_parameters(temperature, max_new_tokens)
        
        start_time = datetime.now()
        
        try:
            # 構建提示詞
            prompt, context = self._build_prompt(
                query=query,
                use_rag=use_rag,
                enable_memory=enable_memory,
                history=history,
                max_new_tokens_for_reply=max_new_tokens,
                source_files=source_files
            )
            
            # 獲取模型和分詞器
            # 處理不同的返回值格式
            result = self.model_manager.get_model_and_tokenizer()
            
            if isinstance(result, tuple):
                if len(result) >= 2:
                    model = result[0]
                    tokenizer = result[1]
                else:
                    logger.error(f"get_model_and_tokenizer 返回的元組長度不足: {len(result)}")
                    raise ValueError(f"Expected at least 2 values, got {len(result)}")
            else:
                logger.error(f"get_model_and_tokenizer 返回類型錯誤: {type(result)}")
                raise ValueError(f"Expected tuple, got {type(result)}")
            
            # 檢查模型和 tokenizer 是否有效
            if model is None or tokenizer is None:
                raise ValueError("Model or tokenizer is None")
            
            # 生成回覆
            with self._lock:
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                
                # 如果模型在 GPU 上，將輸入移到相同設備
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                logger.info(f"開始生成回覆... (temperature={temperature}, max_new_tokens={max_new_tokens})")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # 解碼回覆
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            
            # 後處理回覆
            cleaned_response = self._post_process_response(generated_text)
            
            # 計算用時
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # 返回結果
            result = {
                "response": cleaned_response,
                "used_rag": use_rag,
                "context_provided": context if use_rag else None,
                "generation_time": generation_time,
                "token_count": len(outputs[0]) - len(inputs['input_ids'][0]),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "source_files": source_files,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"非串流回覆生成完成，用時: {generation_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"生成回覆時出錯: {e}", exc_info=True)
            error_result = {
                "response": f"抱歉，處理您的問題時遇到了錯誤: {str(e)}",
                "used_rag": use_rag,
                "context_provided": None,
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "source_files": source_files,
                "timestamp": datetime.now().isoformat()
            }
            return error_result
    
    def stream_response(self,
                       query: str,
                       use_rag: bool = True,
                       enable_memory: bool = True,
                       history: List[Dict[str, str]] = None,
                       max_new_tokens: int = None,
                       temperature: float = None,
                       source_files: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
        """生成串流回覆 - 改進版本
        
        Args:
            query: 用戶問題
            use_rag: 是否使用 RAG 檢索
            enable_memory: 是否啟用對話歷史
            history: 對話歷史
            max_new_tokens: 最大新生成 token 數 (None 表示使用預設值)
            temperature: 溫度參數 (None 表示使用預設值)
            source_files: 指定要搜索的源文件列表
            
        Yields:
            包含回覆片段和元數據的字典
        """
        if history is None:
            history = []
        
        # 驗證 query
        if not query or not query.strip():
            logger.warning("收到空的串流查詢請求")
            yield {
                "type": "error",
                "error": "Empty query provided",
                "generation_time": 0.0,
                "temperature": temperature or self.default_temperature,
                "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
                "timestamp": datetime.now().isoformat()
            }
            return
        
        # 使用預設值或驗證參數
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        temperature, max_new_tokens = self._validate_parameters(temperature, max_new_tokens)
        
        start_time = datetime.now()
        full_response = ""
        
        try:
            # 構建提示詞
            prompt, context = self._build_prompt(
                query=query,
                use_rag=use_rag,
                enable_memory=enable_memory,
                history=history,
                max_new_tokens_for_reply=max_new_tokens,
                source_files=source_files
            )
            
            # 獲取模型和分詞器
            result = self.model_manager.get_model_and_tokenizer()
            
            if isinstance(result, tuple) and len(result) >= 2:
                model = result[0]
                tokenizer = result[1]
            else:
                raise ValueError(f"Expected tuple with at least 2 values, got {type(result)}")
            
            # 檢查模型和 tokenizer 是否有效
            if model is None or tokenizer is None:
                raise ValueError("Model or tokenizer is None")
            
            # 發送初始元數據
            yield {
                "type": "start",
                "used_rag": use_rag,
                "context_provided": context if use_rag else None,
                "source_files": source_files,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
            with self._lock:
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                
                # 移動輸入到模型設備
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                logger.info(f"開始串流生成回覆... (temperature={temperature}, max_new_tokens={max_new_tokens})")
                
                # 嘗試使用真正的串流
                use_real_streaming = False
                
                try:
                    # 檢查是否支持 TextIteratorStreamer
                    from transformers import TextIteratorStreamer
                    import threading
                    
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
                    
                    # 在單獨的線程中運行生成
                    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # 逐漸收集並發送新的文本
                    for new_text in streamer:
                        full_response += new_text
                        yield {
                            "type": "chunk",
                            "chunk": new_text,
                            "full_response": full_response,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    thread.join()
                    use_real_streaming = True
                    
                    logger.info("使用真正的串流實現")
                    
                except (ImportError, Exception) as e:
                    logger.warning(f"無法使用真正的串流: {e}，使用模擬串流")
                
                # 使用模擬串流（如果真正的串流不可用）
                if not use_real_streaming:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    # 解碼完整回覆
                    generated_text = tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # 模擬串流輸出（使用字符而不是詞）
                    chunk_size = 10  # 每次發送 10 個字符
                    
                    for i in range(0, len(generated_text), chunk_size):
                        chunk = generated_text[i:i+chunk_size]
                        full_response += chunk
                        
                        yield {
                            "type": "chunk",
                            "chunk": chunk,
                            "full_response": full_response,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # 添加更短的延遲以提供更好的用戶體驗
                        import time
                        time.sleep(0.02)
                    
                    logger.info("使用模擬串流實現")
            
            # 後處理完整回覆
            cleaned_response = self._post_process_response(full_response)
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # 計算生成的 token 數（簡化估算）
            if 'outputs' in locals():
                token_count = len(outputs[0]) - len(inputs['input_ids'][0])
            else:
                # 簡化的 token 估算
                token_count = len(full_response.split())
            
            # 發送完成元數據
            yield {
                "type": "end",
                "full_response": cleaned_response,
                "generation_time": generation_time,
                "token_count": token_count,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"串流回覆生成完成，用時: {generation_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"串流生成時出錯: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "timestamp": datetime.now().isoformat()
            }
    
    def _post_process_response(self, response: str) -> str:
        """後處理生成的回覆
        
        Args:
            response: 原始回覆文本
            
        Returns:
            清理後的回覆文本
        """
        if not response:
            return "抱歉，我無法為這個問題提供有效的回覆。"
        
        # 移除多餘的空白和換行
        cleaned = re.sub(r'\n{3,}', '\n\n', response)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # 移除開頭和結尾的空白
        cleaned = cleaned.strip()
        
        # 確保回覆不為空
        if not cleaned:
            cleaned = "抱歉，我無法為這個問題提供有效的回覆。"
        
        return cleaned