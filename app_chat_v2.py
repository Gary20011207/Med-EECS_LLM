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

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_chat.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('app_chat')

# 確保可以導入項目模組
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 導入配置
try:
    import config
    logger.info("成功導入配置模組")
    logger.info(f"LLM 模型: {config.LLM_MODEL_NAME}")
    logger.info(f"資料庫路徑: {config.DB_PATH}")
    logger.info(f"Flask 主機: {config.FLASK_HOST}:{config.FLASK_PORT}")
except ImportError as e:
    logger.critical(f"無法導入配置模組: {e}", exc_info=True)
    sys.exit("無法導入配置，應用程式無法啟動")

# 導入 core 模組
try:
    from core.model_manager import ModelManager
    from core.db_manager import VectorDBManager
    from core.rag_engine import RAGEngine
    logger.info("成功導入 core 模組")
except ImportError as e:
    logger.critical(f"導入 core 模組失敗: {e}", exc_info=True)
    sys.exit("關鍵 core 模組導入失敗，應用程式無法啟動")

app = Flask(__name__)

# 全域應用狀態 - 移除會話歷史（由前端管理）
APP_INITIALIZING = True
APP_INITIALIZED = False
model_manager = None
db_manager = None
rag_engine = None

# Flask 路由
@app.route('/')
def index():
    """渲染聊天介面"""
    return render_template('chat_test_v2.html')  # 改為正確的模板名稱

@app.route('/api/status', methods=['GET'])
def get_app_status():
    """獲取應用程式狀態"""
    global APP_INITIALIZING, APP_INITIALIZED, model_manager, db_manager
    
    if APP_INITIALIZING:
        logger.info("API 狀態請求：應用程式仍在初始化中")
        return jsonify({
            'status': 'initializing',
            'timestamp': datetime.now().isoformat(),
            'message': 'Application is currently initializing. Please try again shortly.'
        })
    
    if not APP_INITIALIZED:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'message': 'Application failed to initialize properly.'
        }), 500
    
    try:
        status_info = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'model_name': config.LLM_MODEL_NAME,
            'active_chat_sessions': 0  # 移除會話追蹤
        }
        
         # 模型設備信息 
        if model_manager:
            model_status = model_manager.get_status()
            
            # 直接使用 ModelManager 提供的詳細信息
            status_info['model_device'] = model_status.get('device_details', {'type': '未知'})
            status_info['gpu_memory'] = model_status.get('gpu_memory')
        
        # 資料庫狀態
        if db_manager:
            db_status = db_manager.get_status()
            status_info['vector_db'] = {
                'connected': db_status['db_connected'],
                'path': db_status['db_path'],
                'doc_count': db_status['record_count'] if db_status['db_connected'] else 'N/A'
            }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"獲取系統狀態時出錯: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

def extract_sources_from_context(context: str) -> list:
    """從上下文中提取源文件信息"""
    sources = []
    if context:
        # 使用正則表達式或其他方式提取源文件信息
        import re
        # 例如，匹配 (來源: filename.pdf | 第 X 頁) 的格式
        pattern = r'\*\*文檔.*?\*\*\s*\((.+?)\):'
        matches = re.findall(pattern, context, re.MULTILINE)
        
        for match in matches:
            # 解析源信息
            parts = match.split(' | ')
            if parts:
                source_name = parts[0].replace('來源: ', '').strip()
                if source_name and source_name not in sources:
                    sources.append(source_name)
    
    return sources

@app.route('/api/chat', methods=['POST'])
def handle_chat_request():
    """處理非串流聊天請求"""
    if APP_INITIALIZING:
        return jsonify({'error': '應用程式初始化中，請稍候'}), 503
    
    if not APP_INITIALIZED or not rag_engine:
        return jsonify({'error': '應用程式尚未完全初始化'}), 503
    
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
    use_rag = data.get('use_rag', True)
    enable_memory = data.get('enable_memory', True)
    source_files = data.get('source_files', None)
    temperature = data.get('temperature', None)
    max_new_tokens = data.get('max_new_tokens', None)
    history = data.get('history', [])  # 從前端接收歷史記錄
    
    if not query:
        return jsonify({'error': '訊息內容不可為空'}), 400
    
    logger.info(f"非串流請求 [Session:{session_id}, UseRAG:{use_rag}]: '{query[:50]}...'")
    logger.info(f"收到歷史記錄 {len(history)} 條")
    
    try:
        # 呼叫 RAG 引擎生成回覆
        result = rag_engine.generate_reply(
            query=query,
            use_rag=use_rag,
            enable_memory=enable_memory,
            history=history,  # 使用前端傳來的歷史
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            source_files=source_files
        )
        
        # 從上下文提取源文件信息
        sources = []
        if result.get('context_provided') and use_rag:
            sources = extract_sources_from_context(result['context_provided'])
        
        # 返回回覆
        response_data = {
            'reply': result['response'],
            'sources': sources,
            'session_id': session_id,
            'used_rag': result['used_rag'],
            'generation_time': result['generation_time'],
            'token_count': result.get('token_count', 0),
            'temperature': result['temperature'],
            'max_new_tokens': result['max_new_tokens']
        }
        
        # 檢查是否有錯誤
        if 'error' in result:
            response_data['error'] = result['error']
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"非串流回覆生成出錯 [Session:{session_id}]: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def handle_stream_chat_request():
    """處理串流聊天請求"""
    if APP_INITIALIZING:
        def error_stream():
            error_payload = {'error': '應用程式初始化中，請稍候', 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=503)
    
    if not APP_INITIALIZED or not rag_engine:
        def error_stream():
            error_payload = {'error': '應用程式尚未完全初始化', 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=503)
    
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', f"stream_sess_{uuid.uuid4().hex[:8]}")
    use_rag = data.get('use_rag', True)
    enable_memory = data.get('enable_memory', True)
    source_files = data.get('source_files', None)
    temperature = data.get('temperature', None)
    max_new_tokens = data.get('max_new_tokens', None)
    history = data.get('history', [])  # 從前端接收歷史記錄
    
    if not query:
        def error_stream():
            error_payload = {'error': '訊息內容不可為空', 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=400)
    
    logger.info(f"串流請求 [Session:{session_id}, UseRAG:{use_rag}]: '{query[:50]}...'")
    logger.info(f"收到歷史記錄 {len(history)} 條")
    
    def event_stream_generator():
        accumulated_response = ""
        sources = []
        context_provided = None
        
        try:
            # 呼叫 RAG 引擎生成串流回覆
            for chunk in rag_engine.stream_response(
                query=query,
                use_rag=use_rag,
                enable_memory=enable_memory,
                history=history,  # 使用前端傳來的歷史
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                source_files=source_files
            ):
                if chunk['type'] == 'start':
                    # 發送開始事件，包含源文件信息
                    yield f"data: {json.dumps({'status': 'started', 'session_id': session_id, **chunk})}\n\n"
                    # 記錄上下文信息
                    context_provided = chunk.get('context_provided')
                    
                elif chunk['type'] == 'chunk':
                    # 發送文本片段
                    accumulated_response = chunk['full_response']
                    yield f"data: {json.dumps({'reply': accumulated_response, 'status': 'generating', **chunk})}\n\n"
                
                elif chunk['type'] == 'end':
                    # 發送完成事件
                    # 從上下文提取源文件信息
                    if context_provided and use_rag:
                        sources = extract_sources_from_context(context_provided)
                    
                    response_chunk = {
                        'reply': chunk['full_response'],
                        'status': 'completed',
                        'session_id': session_id,
                        'sources': sources,
                        'generation_time': chunk['generation_time'],
                        'token_count': chunk.get('token_count', 0),
                        'temperature': chunk['temperature'],
                        'max_new_tokens': chunk['max_new_tokens']
                    }
                    yield f"data: {json.dumps(response_chunk)}\n\n"
                    break
                
                elif chunk['type'] == 'error':
                    # 發送錯誤事件
                    error_chunk = {
                        'error': chunk['error'],
                        'status': 'error',
                        'session_id': session_id,
                        'generation_time': chunk.get('generation_time', 0)
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    break
            
        except GeneratorExit:
            logger.info(f"客戶端斷開串流 [Session:{session_id}]")
        except Exception as e_stream:
            logger.error(f"串流生成過程出錯 [Session:{session_id}]: {e_stream}", exc_info=True)
            error_payload = {
                'error': f"串流錯誤: {str(e_stream)}",
                'status': 'error',
                'session_id': session_id
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
    
    return Response(stream_with_context(event_stream_generator()), content_type='text/event-stream')

@app.route('/api/chat/history/clear', methods=['POST'])
def clear_session_history():
    """清除指定會話的歷史記錄 - 現在只是確認操作"""
    if APP_INITIALIZING:
        return jsonify({'error': '應用程式初始化中，請稍候'}), 503
    
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': '未提供 session_id'}), 400
    
    # 因為歷史記錄現在完全由前端管理，這裡只需要確認操作
    logger.info(f"收到清除 Session [{session_id}] 歷史記錄的請求")
    return jsonify({
        'success': True,
        'message': f'Session {session_id} 歷史清除確認'
    })

@app.route('/api/db/rebuild', methods=['POST'])
def rebuild_vector_database_api():
    """重建向量資料庫 - 處理所有PDF檔案"""
    if APP_INITIALIZING:
        return jsonify({'error': '應用程式初始化中，無法重建資料庫'}), 503
    
    if not db_manager:
        return jsonify({'error': '資料庫管理器未初始化'}), 503
    
    data = request.json
    force_reset = data.get('force_reset', True)
    
    logger.info(f"收到重建向量資料庫請求，強制重置: {force_reset}")
    logger.info("將處理所有PDF檔案")
    
    try:
        # db_manager.reset_and_rebuild_db() 預設會處理所有PDF檔案
        result_db = db_manager.reset_and_rebuild_db(
            pdf_folder=config.PDF_FOLDER,
            db_path=config.DB_PATH,
            emb_model=config.EMBEDDINGS_MODEL_NAME,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            force_reset=force_reset
        )
        
        if result_db:
            logger.info("向量資料庫重建成功")
            
            # 重新連接 RAG 引擎中的資料庫引用
            if rag_engine and hasattr(rag_engine, 'db_manager'):
                rag_engine.db_manager = db_manager
            
            return jsonify({
                'success': True,
                'message': '向量資料庫已成功重建'
            })
        else:
            logger.error("向量資料庫重建失敗")
            return jsonify({
                'success': False,
                'message': '向量資料庫重建失敗'
            }), 500
            
    except Exception as e:
        logger.error(f"重建向量資料庫 API 出錯: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'重建錯誤: {str(e)}'
        }), 500

@app.route('/api/db/source-files', methods=['GET'])
def get_db_source_files():
    """獲取資料庫中可用的源檔案列表"""
    if APP_INITIALIZING:
        return jsonify({'error': '應用程式初始化中，請稍候'}), 503
    
    if not db_manager:
        return jsonify({'error': '資料庫管理器未初始化'}), 503
    
    try:
        # 檢查資料庫連接
        db_status = db_manager.get_status()
        if not db_status['db_connected']:
            return jsonify({
                'source_files': [],
                'message': '資料庫未連接'
            })
        
        # 獲取可用的源檔案
        source_files = db_manager.get_available_source_files()
        
        return jsonify({
            'source_files': source_files,
            'count': len(source_files) if source_files else 0,
            'message': '成功獲取源檔案列表'
        })
        
    except Exception as e:
        logger.error(f"獲取資料庫源檔案時出錯: {e}", exc_info=True)
        return jsonify({
            'error': f'獲取源檔案失敗: {str(e)}',
            'source_files': []
        }), 500

# 應用程式初始化與清理
def perform_initialization():
    """執行應用程式初始化"""
    global APP_INITIALIZING, APP_INITIALIZED, model_manager, db_manager, rag_engine
    
    logger.info(f"Flask 應用：開始初始化後端資源 (PID: {os.getpid()})...")
    APP_INITIALIZING = True
    APP_INITIALIZED = False
    
    try:
        # 步驟 1: 初始化組件管理器
        logger.info("初始化組件管理器...")
        model_manager = ModelManager()
        db_manager = VectorDBManager()
        
        # 步驟 2: 建立/重建向量資料庫（使用所有 PDF，非強制重置）
        logger.info("執行首次向量資料庫建立（使用所有 PDF）...")
        initial_db = db_manager.reset_and_rebuild_db(
            pdf_folder=config.PDF_FOLDER,
            db_path=config.DB_PATH,
            emb_model=config.EMBEDDINGS_MODEL_NAME,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            force_reset=True  # 首次建立不強制重置
        )
        
        if initial_db:
            logger.info(f"首次向量資料庫建立成功")
        else:
            logger.warning(f"首次向量資料庫建立失敗，RAG 功能可能受影響")
        
        # 步驟 3: 初始化 LLM 模型（強制 CPU 首次載入）
        logger.info(f"初始化 LLM 模型（強制 CPU 首次載入）...")
        model_manager.initialize(force_cpu_init=True)
        
        # 步驟 4: 初始化 RAG 引擎
        logger.info("初始化 RAG 引擎...")
        rag_engine = RAGEngine(model_manager, db_manager)
        
        # 步驟 5: 連接向量資料庫
        logger.info("連接向量資料庫...")
        db_instance = db_manager.connect_db()
        if not db_instance:
            logger.warning(f"未能成功連接到向量資料庫")
        else:
            logger.info(f"成功連接到向量資料庫")
        
        APP_INITIALIZED = True
        logger.info(f"Flask 應用：後端資源初始化完成 (PID: {os.getpid()})")
        
    except Exception as e:
        logger.critical(f"後端資源初始化失敗 (PID: {os.getpid()}): {e}", exc_info=True)
        APP_INITIALIZED = False
        # 不要直接退出，讓應用程式繼續運行，但標記為未初始化
    finally:
        APP_INITIALIZING = False

def perform_cleanup():
    """執行應用程式清理"""
    global model_manager, db_manager, rag_engine
    
    logger.info(f"Flask 應用 (PID: {os.getpid()}): 清理後端資源...")
    
    if model_manager:
        try:
            model_manager.shutdown()
            logger.info("ModelManager 已清理")
        except Exception as e:
            logger.error(f"清理 ModelManager 時出錯: {e}")
    
    # db_manager 和 rag_engine 通常不需要特別的清理步驟
    logger.info(f"Flask 應用 (PID: {os.getpid()}): 後端資源清理完成")

# 註冊清理函數
atexit.register(perform_cleanup)

def handle_system_signal(signal_received, frame):
    """處理系統信號"""
    logger.warning(f"收到系統信號 {signal_received}。準備關閉 (PID: {os.getpid()})...")
    sys.exit(0)

# 註冊信號處理器
signal.signal(signal.SIGINT, handle_system_signal)
signal.signal(signal.SIGTERM, handle_system_signal)

if __name__ == '__main__':
    logger.info(f"Flask 應用主腳本開始執行 (PID: {os.getpid()})...")
    
    # 執行應用程式初始化
    logger.info(f"執行應用程式初始化 (PID: {os.getpid()})...")
    perform_initialization()
    
    # 應用啟動後，檢查一次模型狀態
    if APP_INITIALIZED and model_manager:
        try:
            logger.info(f"進程 {os.getpid()}: 應用啟動後檢查模型初始狀態...")
            result = model_manager.get_model_and_tokenizer(
                update_last_used_time=False,
                ensure_on_gpu=False
            )
            if isinstance(result, tuple) and len(result) >= 2:
                model = result[0]
                if model:
                    device = next(model.parameters()).device
                    logger.info(f"LLM ({config.LLM_MODEL_NAME}) 初始狀態於設備: {device} (PID: {os.getpid()})")
                else:
                    logger.warning(f"LLM 未能成功獲取初始狀態 (PID: {os.getpid()})")
        except Exception as e:
            logger.error(f"啟動時檢查模型狀態失敗 (PID: {os.getpid()}): {e}", exc_info=True)
    
    # 啟動 Flask 伺服器
    logger.info(f"啟動 Flask 伺服器 (Debug Mode: {config.FLASK_DEBUG}, PID: {os.getpid()})...")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
        use_reloader=config.FLASK_USE_RELOADER
    )