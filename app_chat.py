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
import torch # 需要導入 torch

# --- 日誌配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_chat.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('app_chat_passive_status')

# --- 將 apps 目錄添加到 Python 路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 導入後端模組 ---
try:
    import apps.LLM_init as llm_init_module
    from apps.LLM_init import (
        initialize_llm_model, initialize_vector_database,
        get_llm_model_and_tokenizer, # 現在這個函數有新參數
        get_vector_db_instance,
        shutdown_llm_resources
    )
    import apps.RAG_QA_stream as rag_qa_module
    logger.info("成功導入後端 apps 模組。")
except ImportError as e:
    logger.critical(f"導入 apps 模組失敗: {e}", exc_info=True)
    sys.exit("關鍵後端模組導入失敗，應用程式無法啟動。")

app = Flask(__name__)

# --- 全域應用狀態 ---
chat_sessions_history = {}

# --- Flask 路由 ---

@app.route('/')
def index():
    return render_template('chat_test.html')

@app.route('/api/status', methods=['GET'])
def get_app_status():
    # logger.info("請求 API 狀態 (被動模式)") # 可選日誌
    try:
        # 使用新參數調用，以避免影響閒置計時器或改變模型設備狀態
        model, _, _ = get_llm_model_and_tokenizer(
            update_last_used_time=False, 
            ensure_on_gpu=False # 僅檢查當前狀態，不強制移動到GPU
        )
        device_info = {}
        gpu_memory_info = None

        if model:
            device = next(iter(model.parameters())).device
            device_info['type'] = device.type
            if device.type == 'cuda':
                # 確保 device.index 存在且有效
                current_cuda_device_idx = device.index if hasattr(device, 'index') and device.index is not None else torch.cuda.current_device()
                device_info['index'] = current_cuda_device_idx
                try:
                    # 使用確定的 current_cuda_device_idx
                    allocated = round(torch.cuda.memory_allocated(current_cuda_device_idx) / (1024 ** 3), 2)
                    reserved = round(torch.cuda.memory_reserved(current_cuda_device_idx) / (1024 ** 3), 2)
                    gpu_memory_info = {'allocated': allocated, 'reserved': reserved, 'unit': 'GB'}
                except Exception as e_gpu:
                    logger.warning(f"獲取 GPU ID {current_cuda_device_idx} 記憶體資訊時出錯: {e_gpu}")
        else:
            device_info = {'type': '模型未載入或未知設備'}
            logger.info("API 狀態：模型實例為 None。")


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
        # 正常聊天請求，需要更新時間並確保模型在GPU (如果可用)
        # get_llm_model_and_tokenizer 預設 update_last_used_time=True, ensure_on_gpu=True
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
        final_session_history = list(current_history)
        try:
            # 串流聊天請求，也需要更新時間並確保模型在GPU
            # rag_qa_module.stream_response 內部會調用 get_llm_model_and_tokenizer (預設行為)
            for chunk in rag_qa_module.stream_response(
                query=query, selected_pdf=selected_pdf,
                enable_memory=enable_memory, history=current_history
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                if "updated_history" in chunk:
                    final_session_history = chunk["updated_history"]
                if chunk.get("status") in ["completed", "error"]:
                    break
        except GeneratorExit:
            logger.info(f"客戶端斷開串流 [Sess:{session_id}]。")
        except Exception as e_stream:
            logger.error(f"串流生成過程出錯 [Sess:{session_id}]: {e_stream}", exc_info=True)
            error_payload = {'reply': f"串流錯誤: {str(e_stream)}", 'sources':[], 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            chat_sessions_history[session_id] = final_session_history
            logger.info(f"串流結束，已更新 Session [{session_id}] 歷史。")
            
    return Response(stream_with_context(event_stream_generator()), content_type='text/event-stream')

@app.route('/api/pdfs', methods=['GET'])
def get_available_pdfs():
    pdf_dir = llm_init_module.PDF_FILES_PATH
    try:
        if not os.path.isdir(pdf_dir):
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
    sys.exit(0)

signal.signal(signal.SIGINT, handle_system_signal)
signal.signal(signal.SIGTERM, handle_system_signal)

if __name__ == '__main__':
    logger.info("Flask 應用準備啟動...")
    is_debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1' or True # 開發測試時預設開啟
    
    if not is_debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        perform_initialization()
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            logger.info("Flask reloader 主進程：資源已初始化。")
    elif is_debug_mode:
        logger.info("Flask reloader 子進程：等待主進程初始化資源...")

    try:
        # 啟動後，第一次獲取模型狀態時，使用被動模式，避免不必要的GPU加載
        model, _, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
        if model:
             logger.info(f"LLM ({llm_init_module.LLM_MODEL_NAME}) 初始狀態於設備: {next(model.parameters()).device}")
        else:
             logger.warning("LLM 未能成功獲取初始狀態。")
    except Exception as e:
        logger.error(f"啟動時檢查模型狀態失敗: {e}", exc_info=True)

    logger.info(f"啟動 Flask 伺服器 (Debug Mode: {is_debug_mode})...")
    app.run(host='0.0.0.0', port=5001, debug=is_debug_mode, threaded=True, use_reloader=is_debug_mode)
