# apps/RAG_QA_stream.py
import os
import threading
import torch
from transformers import TextIteratorStreamer
import time 
import traceback 

from apps.LLM_init import (
    get_llm_model_and_tokenizer,  
    get_vector_db_instance as get_vector_db,  
    count_tokens,                
    initialize_llm_model,         # 用於 __main__ 測試
    initialize_vector_database,   # 用於 __main__ 測試
    shutdown_llm_resources        # 用於 __main__ 測試
)
# 為了在 __main__ 中控制 LLM_init 的閒置超時，我們需要訪問它 
import apps.LLM_init as llm_init_module


# --- 組態設定 (Configuration) ---
SYSTEM_PROMPT = (
    "你是「ERAS 醫療專案管理系統」中的智慧個案管理師，專門負責協助病患完成術前、術後的衛教與追蹤。"
    "你的回答應依據 ERAS（Enhanced Recovery After Surgery）指引內容，並以清楚、簡單、友善的語氣引導病患完成待辦事項。"
    "若無法確定答案，請提醒病患聯繫醫療團隊。請勿提及自己是大型語言模型或內部開發細節，只需以 ERAS 個管師的身份專業回應。"
)
# 對於 RAG 檢索的設定
RAG_TOP_K = 5  # RAG 檢索時返回最相似的 k 個文檔

# =======================
# 歷史記憶建構器 (Memory Builder)
# =======================
def build_memory(
    hist: list,
    enable: bool,
    base_tok: int = 0,
    reserve_for_context_and_query: int = 1024 # 為 RAG context 和當前 query 預留的 token
) -> str:
    """
    構建歷史對話記憶字串。
    會根據可用 token 空間，從最近的對話開始加入。

    Args:
        hist (list): 對話歷史列表，格式為 [{"role": "user/assistant", "content": "..."}]。
        enable (bool): 是否啟用記憶功能。
        base_tok (int): 已被系統提示、RAG上下文、當前問題等佔用的 token 數量。
        reserve_for_context_and_query (int): 需要為 RAG 上下文、系統提示、當前問題和模型回答預留的最小 token 數量。
                                            實際記憶體可用 token = max_ctx - base_tok - max_new_tokens_for_reply.
                                            這裡的 reserve 是指除了記憶本身之外，還需要預留多少空間。
                                            一個更精確的計算應該是 max_total_tokens_for_prompt - base_tok_of_current_elements.
    Returns:
        str: 構建好的歷史對話字串，用於插入到提示詞中。
    """
    if not enable or not hist:
        return "" # 如果禁用記憶或歷史為空，則返回空字串

    _, _, max_ctx = get_llm_model_and_tokenizer() # 獲取模型最大上下文長度
    # 可用於記憶的 token 數量 = 模型總上下文長度 - 已用 token (系統提示、RAG、當前問題) - 為回答預留的 token
    # 假設為回答預留 max_tokens (例如 512 或 1024)
    # 這裡的 reserve 來自舊版，現在改名為 reserve_for_context_and_query，其意義更像是 max_ctx - (已用 + 預計回答)
    # 實際上，記憶體能用的 token 應該是： max_ctx - base_tok (sys_prompt + RAG + query) - 預期LLM回覆的max_tokens
    # 在 _build_prompt 中，傳入的 reserve 參數是 max_ctx - max_tokens (用於LLM生成)。
    # 所以這裡的 reserve 實際上是 `max_tokens_for_memory = max_ctx - max_new_tokens_for_reply - base_tok_for_non_memory_parts`

    # 在 _build_prompt 中, build_memory 的 reserve 參數傳入的是 max_ctx - max_tokens_for_generation
    # base_tok 傳入的是 tlen(sys) + tlen(query)
    # 所以記憶體能用的token是 (max_ctx - max_tokens_for_generation) - (tlen(sys) + tlen(query))
    # 這與 reserve - base_tok 的計算方式一致，其中 reserve 是指 prompt總長度上限
    available_tokens_for_memory = reserve_for_context_and_query - base_tok # reserve_for_context_and_query 此處其實是 max_prompt_len

    memory_string_blocks = []
    current_memory_tokens = 0
    i = len(hist) - 1

    # 從最近的對話開始遍歷
    while i >= 1: # 至少需要一對 user-assistant 對話
        if hist[i]["role"] == "assistant" and hist[i-1]["role"] == "user":
            # Qwen模型的對話格式
            block_str = (f"<|im_start|>user\n{hist[i-1]['content']}\n<|im_end|>\n"
                         f"<|im_start|>assistant\n{hist[i]['content']}\n<|im_end|>\n")
            
            block_token_length = count_tokens(block_str) # 計算此對話塊的 token 長度
            
            if current_memory_tokens + block_token_length > available_tokens_for_memory:
                # 如果加入此塊會超出可用 token，則停止
                break
            
            memory_string_blocks.append(block_str)
            current_memory_tokens += block_token_length
            i -= 2 # 跳過這一對 user 和 assistant
        else:
            i -= 1 # 如果不是成對的，向前移動一個（理論上歷史應該是成對的）
            
    if not memory_string_blocks:
        return "" # 如果沒有可用的記憶塊

    return "".join(reversed(memory_string_blocks)) # 將記憶塊倒序（ chronological order）後合併


def _build_prompt(
    query: str,
    sel_pdf: str,
    enable_mem: bool,
    hist: list,
    max_new_tokens_for_reply: int # LLM預計生成回覆的最大token數
) -> tuple[str, list]:
    """
    構建最終的提示詞，包括系統指令、RAG檢索到的上下文、對話歷史以及當前用戶問題。

    Args:
        query (str): 當前用戶的問題。
        sel_pdf (str): 用戶選擇的 PDF 範圍 ("All PDFs", "No PDFs", 或特定 PDF 檔案名)。
        enable_mem (bool): 是否啟用對話歷史記憶。
        hist (list): 對話歷史列表。
        max_new_tokens_for_reply (int): 模型預計生成回覆的最大 token 數，用於計算可用於提示詞的空間。

    Returns:
        tuple[str, list]: 構建好的完整提示詞字串，以及引用的來源文件列表。
    """
    try:
        _, _, model_max_total_context = get_llm_model_and_tokenizer() # 獲取模型的總最大上下文長度
        
        # --- RAG 檢索 ---
        rag_context_str = ""
        retrieved_sources = []
        # 計算可用於 RAG 上下文的 token 數
        # 需要預留空間給：系統提示、(可能的)歷史記憶、當前問題、以及模型的回覆
        # 這裡的策略是，RAG的內容長度 + 系統提示 + 問題 + 記憶 < model_max_total_context - max_new_tokens_for_reply
        # 我們先為 RAG 設定一個大致的上限，例如總上下文的 1/3 或 1/2，或者更動態計算
        # 假設 RAG 上下文的 token 上限為 (model_max_total_context - max_new_tokens_for_reply) * 0.5 (或一個固定值如2048)
        # 這裡的 limit (max_ctx - max_tokens) 是指整個 prompt (不含LLM回覆) 的長度上限
        # RAG 上下文不能超過這個上限扣掉 system_prompt 和 query 的長度。記憶體也會消耗這個空間。
        # 簡化：RAG 上下文的可用 token 數 = (總上下文 - 預期回覆長度) * 比例 - 其他固定提示詞長度

        # 先計算除了 RAG 和記憶之外，其他部分佔用的 token
        # 基礎提示詞結構 (不含 RAG 和記憶)
        base_prompt_for_token_calc = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                                      f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
        tokens_for_base_structure = count_tokens(base_prompt_for_token_calc)
        
        # 可用於 RAG + 歷史記憶的總 token 數
        available_tokens_for_rag_and_memory = model_max_total_context - max_new_tokens_for_reply - tokens_for_base_structure
        
        # 為 RAG 分配一部分 token，例如 50% (可調整)
        # 另外一部分給 memory，如果RAG用的少，memory可以多用
        max_tokens_for_rag = int(available_tokens_for_rag_and_memory * 0.7) # 假設RAG佔70%優先權

        if sel_pdf != "No PDFs" and max_tokens_for_rag > 0: # 只有在選擇了PDF且有空間時才進行RAG
            try:
                vector_db = get_vector_db()
                search_kwargs = {"k": RAG_TOP_K}
                if sel_pdf != "All PDFs": 
                    search_kwargs["filter"] = {"source_pdf": sel_pdf}
                
                # 執行相似性搜索
                docs = vector_db.similarity_search(query, **search_kwargs)
                
                if docs:
                    current_rag_tokens = 0
                    rag_chunks = []
                    for d in docs:
                        # 構建每個文檔塊的內容，包含來源信息
                        chunk = f"[{d.metadata.get('source_pdf','Unknown Source')}]\n{d.page_content.strip()}"
                        chunk_token_length = count_tokens(chunk + "\n\n") # 計算 token，加上分隔符
                        
                        if current_rag_tokens + chunk_token_length > max_tokens_for_rag:
                            break # 如果超出 RAG 的 token 限制，則停止添加
                        
                        rag_chunks.append(chunk)
                        current_rag_tokens += chunk_token_length
                        if d.metadata.get("source_pdf"):
                            retrieved_sources.append(d.metadata["source_pdf"])
                    
                    if rag_chunks:
                        rag_context_str = "\n\n".join(rag_chunks) # 將所有選中的文檔塊合併為 RAG 上下文

            except Exception as e:
                print(f"RAG 檢索過程中發生錯誤: {e}") # 使用 print 或 logger
                traceback.print_exc()
        
        # --- 構建系統提示 (包含 RAG 上下文) ---
        # Qwen 格式的系統提示
        if rag_context_str:
            system_prompt_full = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
                                  f"請參考以下資訊來回答問題 (Context):\n{rag_context_str}\n<|im_end|>\n")
        else:
            system_prompt_full = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        
        tokens_for_system_prompt_and_rag = count_tokens(system_prompt_full)
        tokens_for_query = count_tokens(f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n") # query + assistant 開頭

        # --- 構建歷史記憶 ---
        # 計算可用於歷史記憶的 token 數
        # 可用 token = 模型總上下文 - 新回覆 token - (系統提示+RAG) - 當前問題提示
        tokens_available_for_memory_actually = (model_max_total_context - 
                                                max_new_tokens_for_reply - 
                                                tokens_for_system_prompt_and_rag - 
                                                tokens_for_query)

        # base_tok for build_memory should be 0, as available_tokens_for_memory_actually is the actual limit for memory
        # reserve for build_memory is this available_tokens_for_memory_actually
        memory_str = build_memory(hist, enable_mem, base_tok=0, reserve_for_context_and_query=tokens_available_for_memory_actually)
        
        # --- 構建最終提示詞 ---
        final_prompt_str = (system_prompt_full + 
                            memory_str + 
                            f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n") # 模型接續寫作的部分
        
        # print(f"[_build_prompt] Final prompt length: {count_tokens(final_prompt_str)} tokens")
        return final_prompt_str, sorted(list(set(retrieved_sources))) #確保來源列表元素唯一並排序

    except Exception as e:
        print(f"構建提示詞時發生嚴重錯誤: {e}")
        traceback.print_exc()
        # 在發生嚴重錯誤時，返回一個不包含RAG和歷史的基礎提示詞
        basic_prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                        f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n")
        return basic_prompt, []

# =========================
# GPU 狀態檢查 (GPU Check & Optimization)
# =========================
def check_gpu_status():
    """檢查當前LLM模型的GPU狀態並輸出相關資訊。"""
    try:
        model, _, _ = get_llm_model_and_tokenizer() # 獲取模型實例
        
        if model is None:
            print("GPU狀態檢查：模型尚未初始化。")
            return

        device = next(model.parameters()).device # 獲取模型參數所在的設備
        
        print(f"模型當前運行的設備: {device}")
        
        if device.type == "cuda":
            cuda_id = device.index if device.index is not None else torch.cuda.current_device()
            print(f"正在使用 CUDA 設備 ID: {cuda_id}")
            
            if torch.cuda.is_available(): # 雙重確認 CUDA 可用性
                allocated_memory = torch.cuda.memory_allocated(cuda_id) / (1024 ** 3)  # 轉換為 GB
                reserved_memory = torch.cuda.memory_reserved(cuda_id) / (1024 ** 3)    # 轉換為 GB
                print(f"GPU 記憶體使用: 已分配 {allocated_memory:.2f} GB, 已保留 {reserved_memory:.2f} GB")
        else:
            print("警告: 模型目前不在 GPU 上運行!")
            if torch.cuda.is_available():
                print("但偵測到系統中存在可用的 CUDA 設備。")

    except Exception as e:
        print(f"檢查 GPU 狀態時發生錯誤: {e}")
        traceback.print_exc()

# =========================
# 非串流回覆 API (Non-Streaming Reply API)
# =========================
def generate_reply(
    query: str,
    selected_pdf: str = "All PDFs",
    enable_memory: bool = True,
    history: list = None,
    max_new_tokens: int = 512, # 增加預設最大生成 token 數
    temperature: float = 0.1   # 預設溫度稍低，使其更具確定性
) -> tuple[list, dict]:
    """
    生成完整的單次回覆（非串流模式）。

    Args:
        query (str): 用戶的查詢語句。
        selected_pdf (str): 用於RAG的PDF範圍 ("All PDFs", "No PDFs", 或特定文件名)。
        enable_memory (bool): 是否啟用對話歷史記憶。
        history (list, optional): 對話歷史列表。預設為 None (將初始化為空列表)。
        max_new_tokens (int): LLM 生成回覆的最大 token 數量。
        temperature (float): 控制生成文本隨機性的溫度參數 (0.0 表示確定性)。

    Returns:
        tuple[list, dict]: 更新後的對話歷史列表，以及包含回覆、來源和歷史的結果字典。
    """
    if history is None:
        history = [] # 初始化歷史列表
    
    func_start_time = time.time()
    print(f"開始處理非串流查詢: '{query[:50]}...'") # 打印部分查詢以利追蹤
    
    try:
        check_gpu_status() # 檢查 GPU 狀態
        
        model, tokenizer, _ = get_llm_model_and_tokenizer() # 獲取模型和分詞器
        
        # 構建提示詞，同時獲取引用的來源
        prompt_str, sources = _build_prompt(query, selected_pdf, enable_memory, history, max_new_tokens)
        
        # 將提示詞轉換為模型輸入
        inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False) # add_special_tokens=False 因為prompt已包含特殊符號
        input_ids = inputs.input_ids.to(model.device)
        # 注意: Qwen 等模型可能不需要 attention_mask，或 tokenizer 會自動處理
        attention_mask = inputs.get("attention_mask") 
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        print(f"輸入 Input IDs 設備: {input_ids.device}, Attention Mask 設備: {attention_mask.device if attention_mask is not None else 'N/A'}")
        print(f"開始生成非串流回覆 (max_new_tokens={max_new_tokens}, temperature={temperature})...")
        
        with torch.no_grad(): # 在推斷時不需要計算梯度
            generation_params = {
                'max_new_tokens': max_new_tokens,
                'temperature': max(0.01, temperature), # 溫度不能為0，否則可能出錯或行為怪異
                'do_sample': (temperature > 0.01),    # 僅在溫度高於某個閾值時才進行採樣
                'pad_token_id': tokenizer.eos_token_id # 處理填充
            }
            if attention_mask is not None:
                generation_params['attention_mask'] = attention_mask
            
            # 生成模型輸出
            output_ids = model.generate(input_ids, **generation_params)
        
        # 從輸出中移除輸入部分 (提示詞部分)
        response_ids = output_ids[0][input_ids.shape[1]:]
        # 解碼生成的回覆
        reply_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Qwen模型本身可能已處理<|im_end|>，但做個保險去除
        if reply_text.endswith("<|im_end|>"):
            reply_text = reply_text[:-len("<|im_end|>")].strip()

        processing_time = time.time() - func_start_time
        print(f"非串流回覆生成完成。耗時: {processing_time:.2f} 秒。回覆字數: {len(reply_text)}")
        
    except Exception as e:
        print(f"生成非串流回覆時發生嚴重錯誤: {e}")
        traceback.print_exc()
        reply_text = "抱歉，處理您的請求時遇到一個內部錯誤，請稍後再試。"
        sources = []
        processing_time = time.time() - func_start_time
        print(f"生成出錯。耗時: {processing_time:.2f} 秒")
    
    # 更新對話歷史
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": reply_text}
    ]
    
    # 構建結果字典
    result_dict = {
        "reply": reply_text,
        "sources": sources,
        "updated_history": updated_history # 包含完整的更新後歷史
    }
    
    return updated_history, result_dict # 返回歷史和結果字典 

# =======================
# 串流回覆 API (Streaming Reply API) - 修正版
# =======================
def stream_response(
    query: str,
    selected_pdf: str = "All PDFs",
    enable_memory: bool = True,
    history: list = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1
):
    if history is None:
        history = []

    func_start_time = time.time()
    # print(f"開始處理串流查詢 (修正版 v2): '{query[:50]}...'")
    
    sources = [] # 初始化，確保在try外面可見
    model = None
    tokenizer = None

    try:
        # check_gpu_status() # 可選，可能增加初始延遲
        model, tokenizer, _ = get_llm_model_and_tokenizer()
        prompt_str, sources = _build_prompt(query, selected_pdf, enable_memory, history, max_new_tokens)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # --- 輸入準備 (移出線程，確保 model.device 在主線程中被訪問) ---
        inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids_on_device = inputs.input_ids.to(model.device) # 直接傳遞已在目標設備上的張量
        attention_mask_on_device = inputs.get("attention_mask")
        if attention_mask_on_device is not None:
            attention_mask_on_device = attention_mask_on_device.to(model.device)

        generation_kwargs = {
            # 'input_ids' 會在線程中被賦值，但這裡先定義結構
            'max_new_tokens': max_new_tokens,
            'temperature': max(0.01, temperature),
            'do_sample': (temperature > 0.01),
            'streamer': streamer,
            'pad_token_id': tokenizer.eos_token_id,
        }
        if attention_mask_on_device is not None:
            generation_kwargs['attention_mask'] = attention_mask_on_device
        
        # --- 定義生成線程函數 ---
        def generation_thread_func(thread_input_ids, thread_gen_kwargs): # 傳遞 input_ids 和 kwargs
            try:
                with torch.no_grad():
                    model.generate(input_ids=thread_input_ids, **thread_gen_kwargs) # 使用傳入的 input_ids
            except Exception as e_thread:
                print(f"串流生成線程內部錯誤: {e_thread}")
                traceback.print_exc()
            finally:
                # 確保 streamer 在線程結束時被正確關閉 (TextIteratorStreamer 可能需要這個)
                # 通常 TextIteratorStreamer 會在迭代結束時自然停止，但顯式關閉可能更安全
                # (經查 TextIteratorStreamer 沒有顯式的 close 方法，它依賴 StopIteration)
                pass


        thread = threading.Thread(target=generation_thread_func, args=(input_ids_on_device, generation_kwargs))
        thread.daemon = True
        thread.start()

        # --- 處理串流輸出 (與用戶原始版本相似) ---
        prefix_str = f"資料來源：{', '.join(sources)}\n\n" if sources else ""
        cumulative_reply_for_yield = prefix_str  # 這是將要 yield 出去的完整累加字串 (包含前綴)
        actual_model_reply_accumulator = ""     # 這是模型實際生成的內容累加 (不含前綴，用於歷史記錄)
        
        # token_count_for_log = 0 # 如果需要可以取消註釋
        # last_log_time = time.time()

        for text_chunk in streamer:
            if not text_chunk: continue

            actual_model_reply_accumulator += text_chunk
            cumulative_reply_for_yield += text_chunk
            
            # token_count_for_log +=1
            # current_time = time.time()
            # if current_time - last_log_time > 5.0:
            #     elapsed = current_time - func_start_time
            #     print(f"串流生成中... 已生成約 {token_count_for_log} 文本塊，耗時: {elapsed:.2f} 秒")
            #     last_log_time = current_time
            
            temp_history_snapshot = history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": actual_model_reply_accumulator} # 歷史用不含前綴的純模型回覆
            ]
            yield {
                "reply": cumulative_reply_for_yield, # 給前端顯示的，包含前綴的累計回覆
                "sources": sources,
                "updated_history": temp_history_snapshot # 使用者原始的 key name
            }
        
        thread.join(timeout=15) # 給予線程足夠時間結束，例如15秒
        if thread.is_alive():
            print("警告：串流生成線程在 join 時超時，可能未完全結束。")

        # total_processing_time = time.time() - func_start_time
        # print(f"串流回覆生成完成。總計約 {token_count_for_log} 文本塊。總耗時: {total_processing_time:.2f} 秒")
        
        # 串流結束後，發送一個最終狀態 (可選，但有助於前端處理)
        final_complete_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": actual_model_reply_accumulator}
        ]
        yield {
            "reply": cumulative_reply_for_yield, # 最終完整回覆 (含前綴)
            "sources": sources,
            "updated_history": final_complete_history, # 最終完整歷史
            "status": "completed" # 標記串流結束
        }

    except Exception as e_main:
        print(f"串流回覆主流程發生嚴重錯誤: {e_main}")
        traceback.print_exc()
        error_reply_text = "抱歉，處理您的串流請求時遇到一個內部錯誤，請稍後再試。"
        error_history_snapshot = history + [
            {"role": "user", "content": query}, {"role": "assistant", "content": error_reply_text}
        ]
        yield {
            "reply": error_reply_text, "sources": sources, # 即使出錯，sources 可能已獲取
            "updated_history": error_history_snapshot, "status": "error"
        }
    # finally:
        # current_time = time.time()
        # print(f"串流處理函數結束。總耗時: {current_time - func_start_time:.2f} 秒")


# =======================
# 測試函數 (Test Function) - 更新以處理新的 stream_response yield 結構
# =======================
def _test_module():
    """內部測試函數，用於驗證本模組的核心功能。"""
    print("\n" + "="*20 + " RAG_QA_stream 模組功能測試 " + "="*20)
    
    try:
        # check_gpu_status() # 初始 GPU 狀態檢查 (測試時可開啟)
        
        print("\n===== 測試一：非串流回覆 (Non-Streaming Reply) =====")
        test_query_non_stream = "手術前一天我可以吃什麼？請詳細說明。"
        print(f"查詢: \"{test_query_non_stream}\"")
        
        _, non_stream_result = generate_reply(query=test_query_non_stream)
        
        print(f"\n非串流回覆生成完成:")
        print(f"回覆內容:\n{non_stream_result['reply']}")
        if non_stream_result["sources"]:
            print(f"參考來源: {', '.join(non_stream_result['sources'])}")
        
        current_history = non_stream_result["updated_history"]

        print("\n===== 測試二：串流回覆 (Streaming Reply) - 啟用記憶 =====")
        test_query_stream = "那麼術後初期，飲食上又有哪些需要注意的呢？"
        print(f"查詢 (基於上次對話): \"{test_query_stream}\"")
        print("開始串流輸出 (每個文本塊打印一個'.', 完成後顯示完整回覆):")
        
        final_streamed_reply_content = "串流未成功獲取內容"
        final_streamed_sources = []
        stream_chunk_count = 0

        for chunk_data in stream_response(
            query=test_query_stream, 
            history=current_history,
            max_new_tokens=150 # 測試時可以設小一點加快速度
        ):
            if chunk_data.get("status") == "completed":
                print("\n串流狀態: 完成")
                final_streamed_reply_content = chunk_data["reply"]
                final_streamed_sources = chunk_data["sources"]
                break 
            elif chunk_data.get("status") == "error":
                print("\n串流狀態: 錯誤")
                final_streamed_reply_content = chunk_data["reply"]
                break
            
            # 打印進度點
            print(".", end="", flush=True)
            # 也可以選擇在這裡打印 chunk_data["reply"] 來觀察即時效果
            # os.system('clear') # 如果需要清屏效果
            # print(chunk_data["reply"]) # 顯示累計的回覆
            # time.sleep(0.05) # 減慢輸出速度以便觀察

            stream_chunk_count +=1
            # 暫時保存，以防 "completed" 狀態未收到
            final_streamed_reply_content = chunk_data["reply"] 
            final_streamed_sources = chunk_data.get("sources", [])


        print(f"\n串流回覆接收完成 (約 {stream_chunk_count} 個主要數據塊)。")
        print(f"\n最終串流回覆內容:\n{final_streamed_reply_content}")
        if final_streamed_sources:
            print(f"參考來源: {', '.join(final_streamed_sources)}")
        
        print("\n" + "="*20 + " RAG_QA_stream 模組測試結束 " + "="*20)

    except KeyboardInterrupt:
        print("\n測試被用戶手動中斷。")
    except Exception as e_test:
        print(f"\n測試過程中發生未預期錯誤: {e_test}")
        traceback.print_exc()

# ===========================
# 主執行區塊 (Standalone Run)
# ===========================
if __name__ == "__main__":
    print("RAG_QA_stream.py 作為獨立腳本運行...")

    # --- 設定 LLM_init 的測試用閒置超時 ---
    # 這是為了在 RAG_QA_stream.py 的獨立測試中觀察 LLM_init 的閒置轉移CPU功能
    # 注意：直接修改其他模組的內部變數不是最佳實踐，但用於整合測試是權宜之計。
    # 更好的方法是在 LLM_init.py 中提供一個設定超時的函數。
    RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT = 20  # 秒，設定一個比 LLM_init 監控間隔短的時間，或確保等待足夠長
                                            # LLM_init 監控線程預設30秒檢查一次
    
    original_llm_init_timeout = llm_init_module._current_inactivity_timeout
    llm_init_module._current_inactivity_timeout = RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT
    print(f"已臨時設定 LLM_init 模組的閒置超時為: {RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT} 秒 (原為: {original_llm_init_timeout} 秒)")

    try:
        print("步驟：初始化 LLM 模型和向量資料庫 (將強制在 CPU 載入模型)...")
        # 應用啟動時，通常先在 CPU 載入模型，避免下載時佔用 GPU
        initialize_llm_model(force_cpu_init=True)
        initialize_vector_database()
        print("LLM 和向量資料庫初始化完成。")
        
        # 執行核心功能測試
        _test_module()

        # --- 觀察 LLM 是否因閒置而移回 CPU ---
        # get_llm_model_and_tokenizer() (在 _test_module 內部被調用) 會將模型移至 GPU
        # 現在我們等待足夠長的時間，讓 LLM_init 中的監控線程有機會工作
        # 等待時間應大於 (監控線程檢查間隔 + 設定的閒置超時)
        # LLM_init 監控線程默認每 30s 檢查。若 RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT = 20s
        # 則至少等待 30s + 20s = 50s。為保險起見，可以稍長一些。
        # 或者，如果監控間隔可以配置，應將其配置得比超時短。
        # 假設 LLM_init 的監控間隔是 30 秒
        wait_time_for_cpu_move = llm_init_module.DEFAULT_INACTIVITY_TIMEOUT # 使用LLM_Init的預設檢查間隔
        if hasattr(llm_init_module, '_monitor_check_interval_seconds'): # 如果定義了檢查間隔
            wait_time_for_cpu_move = llm_init_module._monitor_check_interval_seconds
        else: #否則用預設的30s
            wait_time_for_cpu_move = 30 
        
        wait_time_for_cpu_move += (RAG_QA_TEST_LLM_INACTIVITY_TIMEOUT + 10) # 總等待時間

        print(f"\n測試函數執行完畢。等待 {wait_time_for_cpu_move} 秒以觀察 LLM 是否因閒置自動移回 CPU...")
        time.sleep(wait_time_for_cpu_move)
        
        print("\n等待結束後，再次檢查 GPU 狀態：")
        check_gpu_status() # 檢查模型是否已回到 CPU

    except Exception as e_main_run:
        print(f"在 RAG_QA_stream.py 獨立運行過程中發生主錯誤: {e_main_run}")
        traceback.print_exc()
    finally:
        print("\n獨立運行測試結束。開始執行資源清理...")
        shutdown_llm_resources() # 清理 LLM_init 中的資源
        # 恢復 LLM_init 模組的原始閒置超時設定
        llm_init_module._current_inactivity_timeout = original_llm_init_timeout
        print(f"LLM_init 模組的閒置超時已恢復為: {original_llm_init_timeout} 秒")
        print("所有資源清理完畢。RAG_QA_stream.py 獨立運行結束。")