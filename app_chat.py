# app_chat.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import os
import time
import signal
import atexit
import sys
import logging
from datetime import datetime
import uuid
import torch

# --- 日誌配置 ---
logging.basicConfig(
    level=logging.INFO, # INFO 級別在測試和開發時較為實用
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_chat.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('app_chat_simplified')

# --- 將 apps 目錄添加到 Python 路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: # 避免重複添加
    sys.path.insert(0, current_dir)

# --- 導入後端模組 ---
try:
    import apps.LLM_init as llm_init_module
    from apps.LLM_init import (
        initialize_llm_model, initialize_vector_database,
        get_llm_model_and_tokenizer, get_vector_db_instance,
        shutdown_llm_resources
    )
    import apps.RAG_QA_stream as rag_qa_module
    logger.info("成功導入後端 apps 模組。")
except ImportError as e:
    logger.critical(f"導入 apps 模듈失敗: {e}", exc_info=True)
    sys.exit("關鍵後端模組導入失敗，應用程式無法啟動。")

app = Flask(__name__)

# --- 全域應用狀態 (簡化版，適用於測試) ---
chat_sessions_history = {} # 存儲對話歷史: { "session_id": [messages] }

# --- Flask 路由 ---

@app.route('/')
def index():
    # logger.info(f"服務主頁面 chat_test.html") # 可選日誌
    return render_template('chat_test.html')

@app.route('/api/status', methods=['GET'])
def get_app_status():
    # logger.info("請求 API 狀態") # 可選日誌
    try:
        model, _, _ = get_llm_model_and_tokenizer() # 確保模型已載入
        device_info = {}
        gpu_memory_info = None

        if model: # 僅在模型成功載入後嘗試獲取設備信息
            device = next(iter(model.parameters())).device
            device_info['type'] = device.type
            if device.type == 'cuda':
                device_info['index'] = device.index if hasattr(device, 'index') and device.index is not None else torch.cuda.current_device()
                try:
                    gpu_id = device_info['index']
                    allocated = round(torch.cuda.memory_allocated(gpu_id) / (1024 ** 3), 2)
                    reserved = round(torch.cuda.memory_reserved(gpu_id) / (1024 ** 3), 2)
                    gpu_memory_info = {'allocated': allocated, 'reserved': reserved, 'unit': 'GB'}
                except Exception as e_gpu:
                    logger.warning(f"獲取 GPU 記憶體資訊出錯: {e_gpu}")
        else:
            device_info = {'type': '模型未載入或未知設備'}

        return jsonify({
            'status': 'ok', 'timestamp': datetime.now().isoformat(),
            'model_name': llm_init_module.LLM_MODEL_NAME,
            'model_device': device_info, 'gpu_memory': gpu_memory_info,
            'active_chat_sessions': len(chat_sessions_history)
        })
    except Exception as e:
        logger.error(f"獲取系統狀態時出錯: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat_request():
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
    selected_pdf = data.get('selected_pdf', 'All PDFs')
    enable_memory = data.get('enable_memory', True)

    if not query: return jsonify({'error': '訊息內容不可為空'}), 400
    
    logger.info(f"非串流請求 [Sess:{session_id}, PDF:{selected_pdf}]: '{query[:50]}...'")
    current_history = chat_sessions_history.get(session_id, [])
    
    try:
        _, response_data = rag_qa_module.generate_reply(
            query=query, selected_pdf=selected_pdf,
            enable_memory=enable_memory, history=current_history
        )
        chat_sessions_history[session_id] = response_data["updated_history"]
        return jsonify({
            'reply': response_data['reply'],
            'sources': response_data.get('sources', [])
        })
    except Exception as e:
        logger.error(f"非串流回覆生成出錯 [Sess:{session_id}]: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def handle_stream_chat_request():
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"stream_sess_{uuid.uuid4().hex[:8]}")
    selected_pdf = data.get('selected_pdf', 'All PDFs')
    enable_memory = data.get('enable_memory', True)

    if not query: return jsonify({'error': '訊息內容不可為空'}), 400

    logger.info(f"串流請求 [Sess:{session_id}, PDF:{selected_pdf}]: '{query[:50]}...'")
    current_history = chat_sessions_history.get(session_id, [])

    def event_stream_generator():
        final_session_history = list(current_history) # 操作副本
        try:
            for chunk in rag_qa_module.stream_response(
                query=query, selected_pdf=selected_pdf,
                enable_memory=enable_memory, history=current_history
            ):
                yield f"data: {json.dumps(chunk)}\n\n" # SSE 格式
                if "updated_history" in chunk: # 持續更新，以便最終保存
                    final_session_history = chunk["updated_history"]
                if chunk.get("status") in ["completed", "error"]: # 串流結束標記
                    break
        except GeneratorExit: # 客戶端斷開連接
            logger.info(f"客戶端斷開串流 [Sess:{session_id}]。")
        except Exception as e_stream:
            logger.error(f"串流生成過程出錯 [Sess:{session_id}]: {e_stream}", exc_info=True)
            error_payload = {'reply': f"串流錯誤: {str(e_stream)}", 'sources':[], 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            chat_sessions_history[session_id] = final_session_history # 保存最終的歷史記錄
            logger.info(f"串流結束，已更新 Session [{session_id}] 歷史。")
            
    return Response(stream_with_context(event_stream_generator()), content_type='text/event-stream')

@app.route('/api/pdfs', methods=['GET'])
def get_available_pdfs():
    pdf_dir = llm_init_module.PDF_FILES_PATH
    # logger.info(f"請求 PDF 列表，路徑: {pdf_dir}") # 可選日誌
    try:
        if not os.path.isdir(pdf_dir): # 簡化檢查
            logger.warning(f"PDF 文件夾 {pdf_dir} 不存在或不是目錄。")
            return jsonify(["No PDFs"])
        
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        options = ["All PDFs"] + pdf_files + ["No PDFs"] if pdf_files else ["All PDFs", "No PDFs"]
        return jsonify(options)
    except Exception as e:
        logger.error(f"獲取 PDF 列表出錯: {e}", exc_info=True)
        return jsonify({'error': '無法讀取 PDF 文件列表'}), 500

@app.route('/api/chat/history/clear', methods=['POST'])
def clear_session_history():
    data = request.json
    session_id = data.get('session_id')
    if not session_id: return jsonify({'error': '未提供 session_id'}), 400

    if session_id in chat_sessions_history:
        del chat_sessions_history[session_id]
        logger.info(f"已清除 Session [{session_id}] 歷史。")
        return jsonify({'success': True, 'message': f'Session {session_id} 歷史已清除。'})
    else:
        # logger.warning(f"嘗試清除不存在的 Session [{session_id}] 歷史。") # 可選日誌
        return jsonify({'success': False, 'message': '指定的 Session ID 無歷史記錄。'}), 404

# --- 應用程式初始化與清理 ---
def perform_initialization():
    logger.info("Flask 應用：初始化後端資源...")
    try:
        initialize_llm_model(force_cpu_init=True)
        initialize_vector_database()
        logger.info("Flask 應用：後端資源初始化完成。")
    except Exception as e:
        logger.critical(f"後端資源初始化失敗: {e}", exc_info=True)
        sys.exit("關鍵資源初始化失敗，應用退出。")

def perform_cleanup():
    logger.info("Flask 應用：清理後端資源...")
    shutdown_llm_resources()
    logger.info("Flask 應用：後端資源清理完成。")

atexit.register(perform_cleanup)

def handle_system_signal(signal_received, frame):
    logger.warning(f"收到系統信號 {signal_received}。準備關閉...")
    sys.exit(0) # 將觸發 atexit 註冊的 cleanup

signal.signal(signal.SIGINT, handle_system_signal)
signal.signal(signal.SIGTERM, handle_system_signal)

if __name__ == '__main__':
    logger.info("Flask 應用準備啟動...")
    
    # --- Debug 模式說明 ---
    # `debug=True` 會啟用 Flask 的調試模式，主要特性：
    # 1. 自動重載 (Auto-Reloader): 當程式碼變更時，伺服器會自動重啟，方便開發。
    #    這也是為何下方會有 `os.environ.get("WERKZEUG_RUN_MAIN")` 的檢查，
    #    以避免在 reloader 的子進程中重複執行昂貴的初始化操作。
    # 2. 交互式調試器 (Interactive Debugger): 若應用發生未捕獲的錯誤，
    #    瀏覽器會顯示一個帶有堆疊追蹤和控制台的調試介面。
    # 注意：切勿在生產環境中啟用 debug=True，因其存在安全風險。
    #
    # `is_debug_mode` 變數用於統一控制 Flask app.run 的 debug 和 use_reloader 參數。
    # 您可以通過環境變數 FLASK_DEBUG=1 來啟用它，或者直接修改下面的賦值。
    # 對於測試應用，建議啟用 debug 模式。
    is_debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1' or True # 開發測試時預設開啟
    
    # 僅在主進程 (或非 debug 模式下) 執行初始化
    if not is_debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        perform_initialization()
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            logger.info("Flask reloader 主進程：資源已初始化。")
    elif is_debug_mode: # Werkzeug reloader 的子進程
        logger.info("Flask reloader 子進程：等待主進程初始化資源...")
        # 在子進程中，我們通常不重複初始化，但可能需要短暫等待主進程完成。
        # 然而，LLM_init 中的 get 函數有自己的 lazy-loading，所以這裡不阻塞。

    try: # 啟動後日誌，確認模型狀態
        model, _, _ = get_llm_model_and_tokenizer() # 會觸發GPU載入(如果之前在CPU)
        if model:
             logger.info(f"LLM ({llm_init_module.LLM_MODEL_NAME}) 已就緒於設備: {next(model.parameters()).device}")
        else:
             logger.warning("LLM 未能成功獲取。")
    except Exception as e:
        logger.error(f"啟動時檢查模型狀態失敗: {e}", exc_info=True)

    logger.info(f"啟動 Flask 伺服器 (Debug Mode: {is_debug_mode})...")
    app.run(host='0.0.0.0', port=5001, debug=is_debug_mode, threaded=True, use_reloader=is_debug_mode)