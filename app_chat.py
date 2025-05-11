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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_chat.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('app_chat_complete')

# --- 將 apps 目錄添加到 Python 路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 應用程式級別的組態設定 (直接在此處定義) ---
APP_LLM_MODEL_NAME: str = "Qwen/Qwen2.5-14B-Instruct-1M" 
APP_PDF_ROOT: str = "./PDFS"                             
APP_DB_PATH: str = "./VectorDB"                          
APP_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" 

# --- 導入後端模組 ---
try:
    from apps.LLM_init import (
        initialize_llm_model, 
        get_llm_model_and_tokenizer, 
        get_vector_db, # 假設已改名
        shutdown_llm_resources,
        reconnect_vector_db
    )
    import apps.RAG_QA_stream as rag_qa_module
    from apps.RAG_build import reset_and_rebuild_vectordb 
    
    logger.info("成功導入後端 apps 模組。")
except ImportError as e:
    logger.critical(f"導入 apps 模組失敗: {e}", exc_info=True)
    sys.exit("關鍵後端模組導入失敗，應用程式無法啟動。")

app = Flask(__name__)

# --- 全域應用狀態 ---
chat_sessions_history = {}
APP_INITIALIZING = True # 應用程式初始化標誌

# --- Flask 路由 ---

@app.route('/')
def index():
    return render_template('chat_test.html')

@app.route('/api/status', methods=['GET'])
def get_app_status():
    global APP_INITIALIZING 
    if APP_INITIALIZING:
        logger.info("API 狀態請求：應用程式仍在初始化中。")
        return jsonify({
            'status': 'initializing',
            'timestamp': datetime.now().isoformat(),
            'message': 'Application is currently initializing. Please try again shortly.'
        })
    try:
        model, _, _ = get_llm_model_and_tokenizer(
            update_last_used_time=False, 
            ensure_on_gpu=False
        )
        device_info = {}
        gpu_memory_info = None

        if model:
            device = next(iter(model.parameters())).device
            device_info['type'] = device.type
            if device.type == 'cuda':
                idx = device.index if hasattr(device, 'index') and device.index is not None else torch.cuda.current_device()
                device_info['index'] = idx
                try:
                    allocated = round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 2)
                    reserved = round(torch.cuda.memory_reserved(idx) / (1024 ** 3), 2)
                    gpu_memory_info = {'allocated': allocated, 'reserved': reserved, 'unit': 'GB'}
                except Exception as e_gpu:
                    logger.warning(f"獲取 GPU ID {idx} 記憶體資訊時出錯: {e_gpu}")
        else:
            device_info = {'type': '模型未載入'}
        
        db_instance = get_vector_db() 
        db_status = {
            "connected": db_instance is not None, "path": APP_DB_PATH, 
            "doc_count": db_instance._collection.count() if db_instance and hasattr(db_instance, '_collection') and db_instance._collection is not None else "N/A"
        }

        return jsonify({
            'status': 'ok', 'timestamp': datetime.now().isoformat(),
            'model_name': APP_LLM_MODEL_NAME, 
            'model_device': device_info, 'gpu_memory': gpu_memory_info,
            'vector_db': db_status, 'active_chat_sessions': len(chat_sessions_history)
        })
    except Exception as e:
        logger.error(f"獲取系統狀態時出錯: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat_request():
    if APP_INITIALIZING: return jsonify({'error': '應用程式初始化中，請稍候。'}), 503
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
    use_rag_param = data.get('use_rag', True) 
    enable_memory = data.get('enable_memory', True)

    if not query: return jsonify({'error': '訊息內容不可為空'}), 400
    
    logger.info(f"非串流請求 [Sess:{session_id}, UseRAG:{use_rag_param}]: '{query[:50]}...'")
    current_history = chat_sessions_history.get(session_id, [])
    
    try:
        _, response_data = rag_qa_module.generate_reply(
            query=query, use_rag=use_rag_param, 
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
    if APP_INITIALIZING: 
        def error_stream():
            error_payload = {'error': '應用程式初始化中，請稍候。', 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=503)

    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"stream_sess_{uuid.uuid4().hex[:8]}")
    use_rag_param = data.get('use_rag', True) 
    enable_memory = data.get('enable_memory', True)

    if not query: 
        def error_stream_no_query():
            error_payload = {'error': '訊息內容不可為空', 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream_no_query()), content_type='text/event-stream', status=400)

    logger.info(f"串流請求 [Sess:{session_id}, UseRAG:{use_rag_param}]: '{query[:50]}...'")
    current_history = chat_sessions_history.get(session_id, [])

    def event_stream_generator():
        final_session_history = list(current_history) 
        try:
            for chunk in rag_qa_module.stream_response(
                query=query, use_rag=use_rag_param, 
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
    pdf_dir = APP_PDF_ROOT 
    try:
        if not os.path.isdir(pdf_dir):
            logger.warning(f"PDF 文件夾 {pdf_dir} 不存在或不是目錄。")
            return jsonify({"chat_options": ["No PDFs"], "rebuild_options": []})
        
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        
        chat_options = ["All PDFs"] + (pdf_files if pdf_files else []) + ["No PDFs"]
        rebuild_options = list(pdf_files) 

        return jsonify({"chat_options": chat_options, "rebuild_options": rebuild_options})
    except Exception as e:
        logger.error(f"獲取 PDF 列表時出錯: {e}", exc_info=True)
        return jsonify({'error': '無法讀取 PDF 文件列表', "chat_options": ["No PDFs"], "rebuild_options": []}), 500

@app.route('/api/chat/history/clear', methods=['POST'])
def clear_session_history():
    if APP_INITIALIZING: return jsonify({'error': '應用程式初始化中，請稍候。'}), 503
    data = request.json
    session_id = data.get('session_id')
    if not session_id: return jsonify({'error': '未提供 session_id'}), 400
    if session_id in chat_sessions_history:
        del chat_sessions_history[session_id]
        logger.info(f"已清除 Session [{session_id}] 歷史。")
        return jsonify({'success': True, 'message': f'Session {session_id} 歷史已清除。'})
    else:
        return jsonify({'success': False, 'message': '指定的 Session ID 無歷史記錄。'}), 404

@app.route('/api/db/rebuild', methods=['POST'])
def rebuild_vector_database_api():
    if APP_INITIALIZING: return jsonify({'error': '應用程式初始化中，無法重建資料庫。'}), 503
    data = request.json
    pdf_filenames_to_process = data.get('pdf_filenames', None) 
    force_reset_db = data.get('force_reset', True) 
    logger.info(f"收到重建向量資料庫請求。檔案: {pdf_filenames_to_process}, 強制重置: {force_reset_db}")
    try:
        result_db = reset_and_rebuild_vectordb( 
            pdf_folder=APP_PDF_ROOT,    
            db_path=APP_DB_PATH,        
            emb_model=APP_EMBED_MODEL,  
            force_reset=force_reset_db,
            pdf_filenames=pdf_filenames_to_process
        )
        if result_db:
            logger.info("向量資料庫重建成功。重置內部DB連接...")
            reconnect_vector_db() 
            new_db_instance = get_vector_db()
            if new_db_instance:
                 logger.info(f"重新連接到向量資料庫成功。記錄數: {new_db_instance._collection.count() if hasattr(new_db_instance, '_collection') and new_db_instance._collection is not None else 'N/A'}")
            else:
                 logger.warning("重新連接到向量資料庫似乎失敗了。")
            return jsonify({'success': True, 'message': '向量資料庫已成功重建。'})
        else:
            logger.error("向量資料庫重建失敗。")
            return jsonify({'success': False, 'message': '向量資料庫重建失敗。'}), 500
    except Exception as e:
        logger.error(f"重建向量資料庫 API 出錯: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'重建錯誤: {str(e)}'}), 500

# --- 應用程式初始化與清理 ---
def perform_initialization():
    global APP_INITIALIZING 
    logger.info(f"Flask 應用：開始初始化後端資源 (PID: {os.getpid()})...")
    APP_INITIALIZING = True
    try:
        # 步驟 1: 建立/重建向量資料庫 (使用所有PDF，強制重置)
        logger.info("執行首次向量資料庫建立 (使用所有PDF)...")
        initial_db = reset_and_rebuild_vectordb( 
            pdf_folder=APP_PDF_ROOT,
            db_path=APP_DB_PATH,
            emb_model=APP_EMBED_MODEL,
            force_reset=True, 
            pdf_filenames=None 
        )
        if initial_db:
            logger.info(f"首次向量資料庫建立成功。路徑: {APP_DB_PATH}")
        else:
            logger.error(f"首次向量資料庫建立失敗！RAG 功能可能受影響。")
        
        # 步驟 2: LLM 模型初始化 (強制CPU首次載入)
        logger.info(f"初始化 LLM 模型 (預期首次到CPU)...")
        initialize_llm_model(force_cpu_init=True)
        
        # 步驟 3: 向量資料庫連接
        logger.info(f"嘗試連接到向量資料庫...")
        db_conn = get_vector_db() 
        if not db_conn:
            logger.warning(f"未能成功連接到向量資料庫 '{APP_DB_PATH}'。")
        else:
            logger.info(f"成功連接到向量資料庫。")
        
        logger.info(f"Flask 應用：後端資源初始化完成 (PID: {os.getpid()})。")
    except Exception as e:
        logger.critical(f"後端資源初始化失敗 (PID: {os.getpid()}): {e}", exc_info=True)
        sys.exit("關鍵資源初始化失敗，應用退出。")
    finally:
        APP_INITIALIZING = False 


def perform_cleanup():
    logger.info(f"Flask 應用 (PID: {os.getpid()}): 清理後端資源...")
    shutdown_llm_resources()
    logger.info(f"Flask 應用 (PID: {os.getpid()}): 後端資源清理完成。")

atexit.register(perform_cleanup)

def handle_system_signal(signal_received, frame):
    logger.warning(f"收到系統信號 {signal_received}。準備關閉 (PID: {os.getpid()})...")
    sys.exit(0) 

signal.signal(signal.SIGINT, handle_system_signal)
signal.signal(signal.SIGTERM, handle_system_signal)

if __name__ == '__main__':
    logger.info(f"Flask 應用主腳本開始執行 (PID: {os.getpid()})...")
    
    # 簡化啟動：始終以非偵錯模式執行初始化和運行，移除 reloader 相關邏輯
    # 這樣 perform_initialization 只會執行一次。
    
    logger.info(f"執行應用程式初始化 (PID: {os.getpid()})...")
    perform_initialization() 

    # 應用啟動後，檢查一次模型狀態 (使用被動模式)
    try:
        logger.info(f"進程 {os.getpid()}: 應用啟動後檢查模型初始狀態...")
        model, _, _ = get_llm_model_and_tokenizer(update_last_used_time=False, ensure_on_gpu=False)
        if model:
             logger.info(f"LLM ({APP_LLM_MODEL_NAME}) 初始狀態於設備: {next(model.parameters()).device} (PID: {os.getpid()})")
        else:
             logger.warning(f"LLM 未能成功獲取初始狀態 (PID: {os.getpid()})。")
    except Exception as e:
        logger.error(f"啟動時檢查模型狀態失敗 (PID: {os.getpid()}): {e}", exc_info=True)

    logger.info(f"啟動 Flask 伺服器 (Debug Mode: False, PID: {os.getpid()})...")
    # debug=False 和 use_reloader=False 確保單一進程和無自動重載
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)
