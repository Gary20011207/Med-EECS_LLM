# app_chat.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import os
import time
# import threading # Flask 處理線程，通常不需要在此級別顯式使用 threading 除非有特殊後台任務
import signal
import atexit
import sys
import logging
from datetime import datetime
import uuid # 用於生成唯一的 session_id (雖然前端生成，但備用)
import torch # 需要導入 torch 以便在 status API 中檢查 CUDA 設備

# --- 日誌配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_chat.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('app_chat')

# --- 將 apps 目錄添加到 Python 路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假設 apps 目錄與 app_chat.py 在同一級別
if current_dir not in sys.path: # 避免重複添加
    sys.path.insert(0, current_dir)


# --- 導入後端模組 ---
try:
    # 將整個 LLM_init 模組導入並賦予別名
    import apps.LLM_init as llm_init_module
    # 從 LLM_init 模組中導入需要的特定函數
    from apps.LLM_init import (
        initialize_llm_model,
        initialize_vector_database,
        get_llm_model_and_tokenizer,
        get_vector_db_instance, # 注意：在 LLM_init.py 中此函數名為 get_vector_db_instance
        shutdown_llm_resources
    )
    # 導入 RAG_QA_stream 模組
    import apps.RAG_QA_stream as rag_qa_module
    logger.info("成功導入 apps 模組 (LLM_init, RAG_QA_stream)。")
except ImportError as e:
    logger.error(f"導入 apps 模組失敗: {e}", exc_info=True)
    logger.error("請確保 apps 目錄在 Python 路徑中 (或與 app_chat.py 同級)，並且包含 __init__.py 以及所需的 .py 檔案。")
    sys.exit(1)

app = Flask(__name__)

# --- 全域應用狀態 (簡化版) ---
chat_sessions_history = {}

# --- Flask 路由 ---

@app.route('/')
def index():
    logger.info(f"請求主頁面: {request.remote_addr}")
    return render_template('chat_test.html')

@app.route('/api/status', methods=['GET'])
def get_app_status():
    logger.info("請求 API 狀態")
    try:
        model, _, _ = get_llm_model_and_tokenizer()
        device_info = {}
        gpu_memory_info = None

        if model:
            device = next(iter(model.parameters())).device
            device_info = {
                'type': device.type,
                'index': device.index if device.type == 'cuda' and hasattr(device, 'index') and device.index is not None else (torch.cuda.current_device() if device.type == 'cuda' else None)
            }
            if device_info['type'] == 'cuda' and device_info['index'] is not None:
                try:
                    gpu_id = device_info['index']
                    allocated = round(torch.cuda.memory_allocated(gpu_id) / (1024 ** 3), 2)
                    reserved = round(torch.cuda.memory_reserved(gpu_id) / (1024 ** 3), 2)
                    gpu_memory_info = {'allocated': allocated, 'reserved': reserved, 'unit': 'GB'}
                except Exception as e_gpu:
                    logger.warning(f"獲取 GPU 記憶體資訊時出錯: {e_gpu}")
            elif device_info['type'] == 'cuda' and device_info['index'] is None:
                 logger.warning("模型在 CUDA 上，但無法確定具體 device index。")


        else:
            device_info = {'type': 'N/A - Model not loaded or on unknown device'}

        status_payload = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'model_name': llm_init_module.LLM_MODEL_NAME, # 現在可以使用 llm_init_module
            'model_device': device_info,
            'gpu_memory': gpu_memory_info,
            'active_chat_sessions': len(chat_sessions_history),
        }
        return jsonify(status_payload)
    
    except Exception as e:
        logger.error(f"獲取系統狀態時發生錯誤: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat_request():
    try:
        data = request.json
        query = data.get('message', '').strip()
        session_id = data.get('session_id', f"session_{uuid.uuid4()}") # 生成預設 session_id
        selected_pdf = data.get('selected_pdf', 'All PDFs')
        enable_memory = data.get('enable_memory', True)

        if not query: return jsonify({'error': '訊息內容不可為空'}), 400
        
        logger.info(f"收到非串流聊天請求 [Session: {session_id}, PDF: {selected_pdf}]: '{query[:60]}...'")
        current_history = chat_sessions_history.get(session_id, [])
        
        _, response_data_dict = rag_qa_module.generate_reply(
            query=query, selected_pdf=selected_pdf,
            enable_memory=enable_memory, history=current_history
        )
        
        chat_sessions_history[session_id] = response_data_dict["updated_history"]
        logger.info(f"非串流回覆完成 [Session: {session_id}]。")
        
        return jsonify({
            'reply': response_data_dict['reply'],
            'sources': response_data_dict.get('sources', [])
        })
    except Exception as e:
        logger.error(f"處理非串流聊天請求時出錯: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def handle_stream_chat_request():
    try:
        data = request.json
        query = data.get('message', '').strip()
        session_id = data.get('session_id', f"stream_session_{uuid.uuid4()}")
        selected_pdf = data.get('selected_pdf', 'All PDFs')
        enable_memory = data.get('enable_memory', True)

        if not query: return jsonify({'error': '訊息內容不可為空'}), 400

        logger.info(f"收到串流聊天請求 [Session: {session_id}, PDF: {selected_pdf}]: '{query[:60]}...'")
        current_history = chat_sessions_history.get(session_id, [])

        def event_stream_generator():
            final_accumulated_history = list(current_history) # 複製一份，避免多請求間的修改衝突
            try:
                for chunk in rag_qa_module.stream_response(
                    query=query, selected_pdf=selected_pdf,
                    enable_memory=enable_memory, history=current_history # 傳遞當前會話歷史
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if "updated_history" in chunk: # 持續用最新的歷史快照更新
                        final_accumulated_history = chunk["updated_history"]
                    if chunk.get("status") in ["completed", "error"]:
                        break
            except GeneratorExit:
                logger.info(f"客戶端斷開串流連接 [Session: {session_id}]。")
            except Exception as e_stream:
                logger.error(f"串流生成過程中發生錯誤 [Session: {session_id}]: {e_stream}", exc_info=True)
                error_payload = {'reply': f"串流錯誤: {str(e_stream)}", 'sources':[], 'status': 'error'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                chat_sessions_history[session_id] = final_accumulated_history
                logger.info(f"串流結束，已更新 Session [{session_id}] 的歷史記錄。")
        
        return Response(stream_with_context(event_stream_generator()), content_type='text/event-stream')
    except Exception as e:
        logger.error(f"設定串流聊天時出錯: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

@app.route('/api/pdfs', methods=['GET'])
def get_available_pdfs():
    # 使用 llm_init_module 來訪問在 LLM_init.py 中定義的 PDF_FILES_PATH
    pdf_dir = llm_init_module.PDF_FILES_PATH
    logger.info(f"請求 PDF 列表，掃描路徑: {pdf_dir}")
    try:
        if not os.path.exists(pdf_dir) or not os.path.isdir(pdf_dir):
            logger.warning(f"PDF 文件夾 {pdf_dir} 不存在。")
            return jsonify(["No PDFs"])
        
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        options = ["All PDFs"]
        if not pdf_files:
            options.append("No PDFs")
        else:
            options.extend(pdf_files)
            options.append("No PDFs")
        return jsonify(options)
    except Exception as e:
        logger.error(f"獲取 PDF 列表時出錯: {e}", exc_info=True)
        return jsonify({'error': '無法讀取 PDF 文件列表'}), 500

@app.route('/api/chat/history/clear', methods=['POST'])
def clear_session_history():
    try:
        data = request.json
        session_id = data.get('session_id')
        if not session_id: return jsonify({'error': '未提供 session_id'}), 400

        if session_id in chat_sessions_history:
            del chat_sessions_history[session_id]
            logger.info(f"已清除 Session [{session_id}] 的聊天歷史。")
            return jsonify({'success': True, 'message': f'Session {session_id} 歷史已清除。'})
        else:
            logger.warning(f"嘗試清除不存在的 Session [{session_id}] 的歷史。")
            return jsonify({'success': False, 'message': '指定的 Session ID 不存在歷史記錄。'}), 404
    except Exception as e:
        logger.error(f"清除聊天歷史時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

# --- 應用程式初始化與清理 ---
def perform_initialization():
    logger.info("Flask 應用程式：開始初始化後端資源...")
    try:
        # 使用 llm_init_module.initialize_llm_model 等，或直接使用已導入的名稱
        initialize_llm_model(force_cpu_init=True)
        initialize_vector_database()
        logger.info("Flask 應用程式：後端資源初始化完成。")
    except Exception as e:
        logger.critical(f"後端資源初始化失敗: {e}", exc_info=True)
        sys.exit("關鍵資源初始化失敗，應用程式無法啟動。")

def perform_cleanup():
    logger.info("Flask 應用程式：開始清理後端資源...")
    shutdown_llm_resources() # 使用已導入的名稱
    logger.info("Flask 應用程式：後端資源清理完成。")

atexit.register(perform_cleanup)

def handle_system_signal(signal_received, frame):
    logger.warning(f"收到系統信號 {signal_received}。準備關閉應用程式...")
    # cleanup 會由 atexit 處理
    sys.exit(0) # 正常退出，觸發 atexit

signal.signal(signal.SIGINT, handle_system_signal)
signal.signal(signal.SIGTERM, handle_system_signal)

if __name__ == '__main__':
    logger.info("Flask 應用程式準備啟動...")
    # 確保僅在主進程中執行初始化，避免 Flask reloader 執行兩次
    # 這種檢查方法更為通用
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        perform_initialization()
    elif app.debug and not os.environ.get("WERKZEUG_RUN_MAIN"):
        logger.info("Flask reloader 正在運行，初始化將由主 reloader 進程處理。")
    else: # 非 debug 模式，直接初始化
        perform_initialization()


    try:
        model, _, _ = get_llm_model_and_tokenizer()
        if model:
             # 現在 llm_init_module.LLM_MODEL_NAME 可以被正確訪問
             logger.info(f"主要 LLM 模型 ({llm_init_module.LLM_MODEL_NAME}) 已準備就緒，運行於設備: {next(model.parameters()).device}")
        else:
             logger.warning("主要 LLM 模型未能成功初始化或獲取。")
    except Exception as e:
        logger.error(f"啟動時檢查模型狀態失敗: {e}", exc_info=True) # 錯誤日誌會包含堆疊跟踪

    is_debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1' or app.debug
    # use_reloader 只有在 debug=True 時才有效。如果 debug=False，它會被忽略。
    # 在生產中，通常使用 Gunicorn/uWSGI，它們有自己的進程管理。
    logger.info(f"啟動 Flask 應用伺服器... (Debug: {is_debug_mode})")
    app.run(host='0.0.0.0', port=5001, debug=is_debug_mode, threaded=True, use_reloader=is_debug_mode)