# apps/RAG_QA_stream.py
import os
import threading
import torch
from transformers import TextIteratorStreamer
import time 
import traceback 
import logging # 新增 logging 導入

# --- 日誌設定 ---
# 與其他模組保持一致的日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 導入後端模組 ---
try:
    # 使用相對導入 (假設此文件在 apps 套件內)
    from .LLM_init import (
        get_llm_model_and_tokenizer,  
        get_vector_db, # 假設 LLM_init.py 中已改名
        count_tokens,              
        initialize_llm_model,    # 用於 __main__ 測試
        shutdown_llm_resources   # 用於 __main__ 測試
    )
    # 為了在 __main__ 中控制 LLM_init 的閒置超時和訪問其常數
    import apps.LLM_init as llm_init_module
    from apps.RAG_build import reset_and_rebuild_vectordb as rag_builder_func
    RAG_BUILD_MODULE_AVAILABLE = True
except ImportError as e:
    logger.error(f"導入 apps 子模組失敗: {e}", exc_info=True)
    RAG_BUILD_MODULE_AVAILABLE = False # 標記 RAG_build 不可用於 __main__
    # 為了讓模組在某些情況下仍能加載（例如僅用於不依賴RAG_build的函數），
    # 可以不直接 sys.exit()，但 __main__ 中的測試會受影響。


# --- 組態設定 (Configuration) ---
SYSTEM_PROMPT = (
    "你是「ERAS 醫療專案管理系統」中的智慧個案管理師，專門負責協助病患完成術前、術後的衛教與追蹤。"
    "你的回答應依據 ERAS（Enhanced Recovery After Surgery）指引內容，並以清楚、簡單、友善的語氣引導病患完成待辦事項。"
    "若無法確定答案，請提醒病患聯繫醫療團隊。請勿提及自己是大型語言模型或內部開發細節，只需以 ERAS 個管師的身份專業回應。"
)
RAG_TOP_K = 5  # RAG 檢索時返回最相似的 k 個文檔

# =======================
# 歷史記憶建構器 (Memory Builder)
# =======================
def build_memory(
    hist: list,
    enable: bool,
    base_tok: int = 0,
    reserve_for_context_and_query: int = 1024 
) -> str:
    if not enable or not hist:
        return "" 

    try:
        # 被動模式獲取模型配置，不影響閒置計時或設備
        _, _, model_max_total_context = get_llm_model_and_tokenizer(
            update_last_used_time=False, ensure_on_gpu=False
        )
        if model_max_total_context is None: # 如果模型未初始化
            logger.warning("build_memory: 模型尚未初始化，無法獲取最大上下文長度。")
            return ""
    except Exception as e_get_model:
        logger.error(f"build_memory: 獲取模型配置時出錯: {e_get_model}", exc_info=True)
        return ""


    available_tokens_for_memory = reserve_for_context_and_query - base_tok

    if available_tokens_for_memory <= 0:
        return ""

    memory_string_blocks = []
    current_memory_tokens = 0
    i = len(hist) - 1

    while i >= 1: 
        if hist[i]["role"] == "assistant" and hist[i-1]["role"] == "user":
            block_str = (f"<|im_start|>user\n{hist[i-1]['content']}\n<|im_end|>\n"
                         f"<|im_start|>assistant\n{hist[i]['content']}\n<|im_end|>\n")
            block_token_length = count_tokens(block_str)
            
            if current_memory_tokens + block_token_length > available_tokens_for_memory:
                break
            
            memory_string_blocks.append(block_str)
            current_memory_tokens += block_token_length
            i -= 2 
        else:
            i -= 1 
            
    return "".join(reversed(memory_string_blocks))

# =======================
# 提示詞建構器 (_build_prompt)
# =======================
def _build_prompt(
    query: str,
    use_rag: bool,
    enable_mem: bool,
    hist: list,
    max_new_tokens_for_reply: int 
) -> tuple[str, list]:
    try:
        # 被動模式獲取模型配置
        _, _, model_max_total_context = get_llm_model_and_tokenizer(
            update_last_used_time=False, ensure_on_gpu=False
        )
        if model_max_total_context is None:
            logger.error("_build_prompt: 模型未初始化，無法構建提示詞。")
            # 返回一個基礎提示詞，避免完全失敗
            return (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                    f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n"), []
        
        base_prompt_for_token_calc = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                                      f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
        tokens_for_base_structure_no_rag_mem = count_tokens(base_prompt_for_token_calc)
        
        available_tokens_for_dynamic_parts = model_max_total_context - max_new_tokens_for_reply - tokens_for_base_structure_no_rag_mem
        
        rag_context_str = ""
        retrieved_sources = []
        
        # --- RAG 檢索 (條件化) ---
        if use_rag: # 只有當 use_rag 為 True 時才執行 RAG 檢索
            max_tokens_for_rag = int(available_tokens_for_dynamic_parts * 0.6) # RAG 優先佔用60%動態空間
            if max_tokens_for_rag > 0:
                try:
                    vector_db = get_vector_db() # 獲取向量資料庫實例
                    if vector_db:
                        docs = vector_db.similarity_search(query, k=RAG_TOP_K)
                        if docs:
                            current_rag_tokens = 0
                            rag_chunks = []
                            for d in docs:
                                chunk = f"[{d.metadata.get('source_pdf','Unknown Source')}]\n{d.page_content.strip()}"
                                chunk_token_length = count_tokens(chunk + "\n\n")
                                if current_rag_tokens + chunk_token_length > max_tokens_for_rag:
                                    break
                                rag_chunks.append(chunk)
                                current_rag_tokens += chunk_token_length
                                if d.metadata.get("source_pdf"):
                                    retrieved_sources.append(d.metadata["source_pdf"])
                            if rag_chunks:
                                rag_context_str = "\n\n".join(rag_chunks)
                    else:
                        logger.warning("_build_prompt: use_rag=True 但向量資料庫未連接。")
                except Exception as e_rag:
                    logger.error(f"RAG 檢索過程中發生錯誤: {e_rag}", exc_info=True)
            else:
                logger.info("_build_prompt: use_rag=True 但沒有足夠的 token 空間給 RAG。")
        else:
            logger.info("_build_prompt: use_rag=False，跳過 RAG 檢索。")
        
        # --- 構建系統提示 ---
        if rag_context_str:
            system_prompt_full = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
                                  f"請參考以下資訊來回答問題 (Context):\n{rag_context_str}\n<|im_end|>\n")
        else:
            system_prompt_full = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        
        tokens_for_system_prompt_and_rag = count_tokens(system_prompt_full)
        tokens_for_query_wrapper = count_tokens(f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n") 

        base_tok_for_memory_calc = tokens_for_system_prompt_and_rag + tokens_for_query_wrapper
        reserve_for_memory_calc = model_max_total_context - max_new_tokens_for_reply
        
        memory_str = build_memory(hist, enable_mem, 
                                  base_tok=base_tok_for_memory_calc, 
                                  reserve_for_context_and_query=reserve_for_memory_calc)
        
        final_prompt_str = (system_prompt_full + 
                            memory_str + 
                            f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
        
        return final_prompt_str, sorted(list(set(retrieved_sources)))

    except Exception as e:
        logger.error(f"構建提示詞時發生嚴重錯誤: {e}", exc_info=True)
        basic_prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                        f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
        return basic_prompt, []

# =========================
# 非串流回覆 API (Non-Streaming Reply API)
# =========================
def generate_reply(
    query: str,
    use_rag: bool = True, # <--- 新增
    enable_memory: bool = True,
    history: list = None,
    max_new_tokens: int = 512, 
    temperature: float = 0.1   
) -> tuple[list, dict]:
    if history is None: history = []
    reply_text = ""
    sources = []
    try:
        # 實際生成回覆時，使用 get_llm_model_and_tokenizer 的預設行為 (更新時間，確保在GPU)
        model, tokenizer, _ = get_llm_model_and_tokenizer() 
        if not model or not tokenizer: # 增加檢查
            raise RuntimeError("無法獲取 LLM 模型或分詞器。")

        prompt_str, sources = _build_prompt(query, use_rag, enable_memory, history, max_new_tokens)
        
        inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None: attention_mask = attention_mask.to(model.device)
        
        with torch.no_grad():
            generation_params = {
                'max_new_tokens': max_new_tokens, 'temperature': max(0.01, temperature),
                'do_sample': (temperature > 0.01), 'pad_token_id': tokenizer.eos_token_id
            }
            if attention_mask is not None: generation_params['attention_mask'] = attention_mask
            output_ids = model.generate(input_ids, **generation_params)
        
        response_ids = output_ids[0][input_ids.shape[1]:]
        reply_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if reply_text.endswith("<|im_end|>"): reply_text = reply_text[:-len("<|im_end|>")].strip()
        
    except Exception as e:
        logger.error(f"生成非串流回覆時發生嚴重錯誤: {e}", exc_info=True)
        reply_text = "抱歉，處理您的請求時遇到一個內部錯誤，請稍後再試。"
        sources = [] 

    updated_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": reply_text}]
    result_dict = {"reply": reply_text, "sources": sources, "updated_history": updated_history}
    return updated_history, result_dict

# =======================
# 串流回覆 API (Streaming Reply API)
# =======================
def stream_response(
    query: str,
    use_rag: bool = True, # <--- 新增
    enable_memory: bool = True,
    history: list = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1
):
    if history is None: history = []
    sources = [] 
    model = None
    tokenizer = None

    try:
        # 實際生成回覆時，使用 get_llm_model_and_tokenizer 的預設行為
        model, tokenizer, _ = get_llm_model_and_tokenizer()
        if not model or not tokenizer: # 增加檢查
            raise RuntimeError("無法獲取 LLM 模型或分詞器以進行串流。")

        prompt_str, sources = _build_prompt(query, use_rag, enable_memory, history, max_new_tokens)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids_on_device = inputs.input_ids.to(model.device)
        attention_mask_on_device = inputs.get("attention_mask")
        if attention_mask_on_device is not None:
            attention_mask_on_device = attention_mask_on_device.to(model.device)

        generation_kwargs = {
            'max_new_tokens': max_new_tokens, 'temperature': max(0.01, temperature),
            'do_sample': (temperature > 0.01), 'streamer': streamer,
            'pad_token_id': tokenizer.eos_token_id,
        }
        if attention_mask_on_device is not None:
            generation_kwargs['attention_mask'] = attention_mask_on_device
        
        def generation_thread_func(thread_input_ids, thread_gen_kwargs):
            try:
                with torch.no_grad():
                    model.generate(input_ids=thread_input_ids, **thread_gen_kwargs)
            except Exception as e_thread:
                logger.error(f"串流生成線程內部錯誤: {e_thread}", exc_info=True)
        
        thread = threading.Thread(target=generation_thread_func, args=(input_ids_on_device, generation_kwargs))
        thread.daemon = True
        thread.start()

        prefix_str = f"資料來源：{', '.join(sources)}\n\n" if sources and use_rag else "" # 僅在 use_rag 時添加來源前綴
        cumulative_reply_for_yield = prefix_str
        actual_model_reply_accumulator = ""     

        for text_chunk in streamer:
            if not text_chunk: continue
            actual_model_reply_accumulator += text_chunk
            cumulative_reply_for_yield += text_chunk
            
            temp_history_snapshot = history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": actual_model_reply_accumulator}
            ]
            yield {
                "reply": cumulative_reply_for_yield, 
                "sources": sources if use_rag else [], # 僅在 use_rag 時返回 sources
                "updated_history": temp_history_snapshot 
            }
        
        thread.join(timeout=15) 
        if thread.is_alive():
            logger.warning("警告：串流生成線程在 join 時超時，可能未完全結束。")

        final_complete_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": actual_model_reply_accumulator}
        ]
        yield {
            "reply": cumulative_reply_for_yield, 
            "sources": sources if use_rag else [],
            "updated_history": final_complete_history, 
            "status": "completed" 
        }

    except Exception as e_main:
        logger.error(f"串流回覆主流程發生嚴重錯誤: {e_main}", exc_info=True)
        error_reply_text = "抱歉，處理您的串流請求時遇到一個內部錯誤，請稍後再試。"
        error_history_snapshot = history + [
            {"role": "user", "content": query}, {"role": "assistant", "content": error_reply_text}
        ]
        yield {
            "reply": error_reply_text, "sources": sources if use_rag else [], 
            "updated_history": error_history_snapshot, "status": "error"
        }

# =======================
# 測試函數 (Test Function)
# =======================
def _test_module():
    logging.info("\n" + "="*20 + " RAG_QA_stream 模組功能測試 (優化版) " + "="*20)
    try:
        logging.info("\n===== 測試一：非串流回覆 (使用 RAG) =====")
        test_query_non_stream_rag = "手術前一天我可以吃什麼？請詳細說明。"
        logging.info(f"查詢 (use_rag=True): \"{test_query_non_stream_rag}\"")
        
        _, non_stream_result_rag = generate_reply(query=test_query_non_stream_rag, use_rag=True)
        
        logging.info(f"\n非串流回覆 (RAG) 生成完成:")
        logging.info(f"回覆內容:\n{non_stream_result_rag['reply']}")
        if non_stream_result_rag["sources"]:
            logging.info(f"參考來源: {', '.join(non_stream_result_rag['sources'])}")
        
        current_history = non_stream_result_rag["updated_history"]

        logging.info("\n===== 測試二：非串流回覆 (不使用 RAG) =====")
        test_query_non_stream_no_rag = "請簡單介紹一下 ERAS 計畫。"
        logging.info(f"查詢 (use_rag=False): \"{test_query_non_stream_no_rag}\"")
        
        _, non_stream_result_no_rag = generate_reply(query=test_query_non_stream_no_rag, use_rag=False, history=current_history)
        
        logging.info(f"\n非串流回覆 (無RAG) 生成完成:")
        logging.info(f"回覆內容:\n{non_stream_result_no_rag['reply']}")
        if non_stream_result_no_rag["sources"]: # 理論上應該為空
            logging.info(f"參考來源 (無RAG時應無來源): {', '.join(non_stream_result_no_rag['sources'])}")
        current_history = non_stream_result_no_rag["updated_history"]


        logging.info("\n===== 測試三：串流回覆 (使用 RAG) - 啟用記憶 =====")
        test_query_stream_rag = "那麼術後初期，飲食上又有哪些需要注意的呢？"
        logging.info(f"查詢 (use_rag=True, 基於上次對話): \"{test_query_stream_rag}\"")
        logging.info("開始串流輸出 (每個文本塊打印一個'.', 完成後顯示完整回覆):")
        
        final_streamed_reply_content_rag = "串流未成功獲取內容"
        final_streamed_sources_rag = []
        
        for chunk_data in stream_response(
            query=test_query_stream_rag, 
            use_rag=True,
            history=current_history,
            max_new_tokens=150 
        ):
            if chunk_data.get("status") == "completed":
                logging.info("\n串流狀態 (RAG): 完成")
                final_streamed_reply_content_rag = chunk_data["reply"]
                final_streamed_sources_rag = chunk_data["sources"]
                break 
            # ... (其餘打印邏輯與前一版類似)
            print(".", end="", flush=True)
            final_streamed_reply_content_rag = chunk_data["reply"] 
            final_streamed_sources_rag = chunk_data.get("sources", [])
        print()
        logging.info(f"\n最終串流回覆內容 (RAG):\n{final_streamed_reply_content_rag}")
        if final_streamed_sources_rag:
            logging.info(f"參考來源 (RAG): {', '.join(final_streamed_sources_rag)}")
        
        logging.info("\n" + "="*20 + " RAG_QA_stream 模組測試結束 " + "="*20)

    except KeyboardInterrupt:
        logging.warning("\n測試被用戶手動中斷。")
    except Exception as e_test:
        logging.error(f"\n測試過程中發生未預期錯誤: {e_test}", exc_info=True)

# ===========================
# 主執行區塊 (Standalone Run)
# ===========================
if __name__ == "__main__":
    logging.info("RAG_QA_stream.py 作為獨立腳本運行 (優化版)...")

    if not RAG_BUILD_MODULE_AVAILABLE:
        logging.critical("RAG_build 模組無法導入，無法執行資料庫建立，測試可能失敗。")
        # 決定是否在此處退出
        # sys.exit("RAG_build 依賴缺失。")
    else:
        try:
            logger.info("\n[主測試步驟 0] 執行 RAG_build.py 以確保向量資料庫存在...")
            db_instance_from_rag = rag_builder_func(
                pdf_folder=llm_init_module.PDF_FILES_PATH,        
                db_path=llm_init_module.DB_PATH,                  
                emb_model=llm_init_module.EMBEDDINGS_MODEL_NAME,  
                force_reset=True,                 
                pdf_filenames=None # 處理所有 PDF
            )
            if db_instance_from_rag:
                logger.info(f"RAG_build.py 成功執行，向量資料庫 '{llm_init_module.DB_PATH}' 已建立/更新。")
            else:
                logger.error("RAG_build.py 執行失敗。")
        except Exception as e_rag_build_main:
            logger.error(f"執行 RAG_build 時發生錯誤: {e_rag_build_main}", exc_info=True)


    RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT = 20 
    original_llm_init_timeout = llm_init_module._current_inactivity_timeout
    llm_init_module._current_inactivity_timeout = RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT
    logging.info(f"已臨時設定 LLM_init 模組的閒置超時為: {RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT} 秒")

    try:
        logging.info("步驟：初始化 LLM 模型 (將強制在 CPU 載入模型)...")
        initialize_llm_model(force_cpu_init=True)
        logging.info("LLM 初始化完成。")
        
        _test_module() 
        
        wait_time_for_cpu_move = llm_init_module.MONITOR_CHECK_INTERVAL_SECONDS + RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT + 10
        logging.info(f"\n測試函數執行完畢。等待 {wait_time_for_cpu_move} 秒以觀察 LLM 是否因閒置自動移回 CPU...")
        time.sleep(wait_time_for_cpu_move)
        
        logging.info("\n等待結束後，再次檢查 GPU 狀態 (使用被動模式)：")
        # 呼叫 get_llm_model_and_tokenizer 以被動模式檢查設備
        model_final_check, _, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
        if model_final_check:
            final_device = next(model_final_check.parameters()).device
            logging.info(f"模型最終設備: {final_device}")
            if final_device.type == "cpu":
                logging.info("模型已按預期移回CPU。")
            else:
                logging.warning("模型未按預期移回CPU。")
        else:
            logging.error("無法獲取最終模型狀態。")


    except Exception as e_main_run:
        logging.error(f"在 RAG_QA_stream.py 獨立運行過程中發生主錯誤: {e_main_run}", exc_info=True)
    finally:
        logging.info("\n獨立運行測試結束。開始執行資源清理...")
        shutdown_llm_resources() 
        llm_init_module._current_inactivity_timeout = original_llm_init_timeout 
        logging.info(f"LLM_init 模組的閒置超時已恢復為: {original_llm_init_timeout} 秒")
        logging.info("所有資源清理完畢。RAG_QA_stream.py 獨立運行結束。")
