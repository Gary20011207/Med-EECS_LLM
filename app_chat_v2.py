# app_chat.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import os
import sys
import logging
from datetime import datetime
import uuid

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
except ImportError as e:
    logger.critical(f"無法導入配置模組: {e}", exc_info=True)
    sys.exit("無法導入配置，應用程式無法啟動")

# 導入 core 模組並初始化
try:
    from core.model_manager import ModelManager
    from core.db_manager import DBManager
    from core.rag_engine import RAGEngine
    
    # 初始化組件
    model_manager = ModelManager()
    model, tokenizer, model_max_length = model_manager.initialize()
    
    db_manager = DBManager()
    db = db_manager.connect_db()
    
    rag_engine = RAGEngine(model_manager, db_manager)
    logger.info("RAG 引擎初始化完成")
    
except ImportError as e:
    logger.critical(f"導入 core 模組失敗: {e}", exc_info=True)
    sys.exit("關鍵 core 模組導入失敗，應用程式無法啟動")
except Exception as e:
    logger.critical(f"初始化組件失敗: {e}", exc_info=True)
    sys.exit("組件初始化失敗，應用程式無法啟動")

app = Flask(__name__)

@app.route('/')
def index():
    """渲染聊天介面"""
    return render_template('chat_test_v2.html')

@app.route('/api/status', methods=['GET'])
def get_app_status():
    """獲取應用程式狀態"""
    try:
        status_info = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'model_name': config.DEFAULT_LLM_MODEL_NAME
        }
        
        # 模型狀態
        model_status = model_manager.get_status()
        status_info['model_device'] = model_status.get('device_details', {'type': '未知'})
        status_info['gpu_memory'] = model_status.get('gpu_memory')
        
        # 資料庫狀態
        db_sources = db_manager.get_available_source_files()
        status_info['vector_db'] = {
            'connected': True if db_sources else False,
            'doc_count': len(db_sources) if db_sources else 0
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"獲取系統狀態時出錯: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat_request():
    """處理非串流聊天請求"""
    try:
        data = request.json
        query = data.get('message', '').strip()
        session_id = data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
        use_rag = data.get('use_rag', True)
        enable_memory = data.get('enable_memory', True)
        source_files = data.get('source_files', None)
        temperature = data.get('temperature', None)
        max_new_tokens = data.get('max_new_tokens', None)
        history = data.get('history', [])
        
        if not query:
            return jsonify({'error': '訊息內容不可為空'}), 400
        
        logger.info(f"非串流請求 [Session:{session_id}]: '{query[:50]}...'")
        
        # 呼叫 RAG 引擎生成回覆
        response = rag_engine.generate(
            query=query,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_rag=use_rag,
            enable_memory=enable_memory,
            history=history,
            source_files=source_files
        )
        
        # 獲取使用的資源
        resources = rag_engine.get_last_rag_resources()
        
        # 整理源文件信息
        sources = []
        for resource in resources:
            source_name = resource['source_file_name']
            if source_name not in sources:
                sources.append(source_name)
        
        response_data = {
            'reply': response,
            'sources': sources,
            'session_id': session_id,
            'used_rag': use_rag,
            'resources': resources  # 提供詳細資源信息
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"非串流回覆生成出錯: {e}", exc_info=True)
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def handle_stream_chat_request():
    """處理串流聊天請求"""
    try:
        data = request.json
        query = data.get('message', '').strip()
        session_id = data.get('session_id', f"stream_sess_{uuid.uuid4().hex[:8]}")
        use_rag = data.get('use_rag', True)
        enable_memory = data.get('enable_memory', True)
        source_files = data.get('source_files', None)
        temperature = data.get('temperature', None)
        max_new_tokens = data.get('max_new_tokens', None)
        history = data.get('history', [])
        
        if not query:
            def error_stream():
                error_payload = {'error': '訊息內容不可為空', 'status': 'error'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=400)
        
        logger.info(f"串流請求 [Session:{session_id}]: '{query[:50]}...'")
        
        def event_stream_generator():
            try:
                # 發送開始事件
                yield f"data: {json.dumps({'status': 'started', 'session_id': session_id})}\n\n"
                
                # 呼叫 RAG 引擎進行串流生成
                for text_chunk in rag_engine.stream(
                    query=query,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_rag=use_rag,
                    enable_memory=enable_memory,
                    history=history,
                    source_files=source_files
                ):
                    chunk_data = {
                        'reply': text_chunk,
                        'status': 'generating',
                        'session_id': session_id
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # 串流完成，獲取資源信息
                resources = rag_engine.get_last_rag_resources()
                
                # 整理源文件信息
                sources = []
                for resource in resources:
                    source_name = resource['source_file_name']
                    if source_name not in sources:
                        sources.append(source_name)
                
                # 發送完成事件
                final_data = {
                    'status': 'completed',
                    'session_id': session_id,
                    'sources': sources,
                    'resources': resources,
                    'used_rag': use_rag
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                
            except GeneratorExit:
                logger.info(f"客戶端斷開串流 [Session:{session_id}]")
            except Exception as e:
                logger.error(f"串流生成過程出錯 [Session:{session_id}]: {e}", exc_info=True)
                error_payload = {
                    'error': f"串流錯誤: {str(e)}",
                    'status': 'error',
                    'session_id': session_id
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
        
        return Response(stream_with_context(event_stream_generator()), content_type='text/event-stream')
        
    except Exception as e:
        logger.error(f"串流請求失敗: {e}", exc_info=True)
        def error_stream():
            error_payload = {'error': str(e), 'status': 'error'}
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), content_type='text/event-stream', status=500)

@app.route('/api/chat/history/clear', methods=['POST'])
def clear_session_history():
    """清除指定會話的歷史記錄 - 由前端管理"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': '未提供 session_id'}), 400
    
    logger.info(f"收到清除 Session [{session_id}] 歷史記錄的請求")
    return jsonify({
        'success': True,
        'message': f'Session {session_id} 歷史清除確認'
    })

@app.route('/api/db/rebuild', methods=['POST'])
def rebuild_vector_database_api():
    """重建向量資料庫"""
    try:
        data = request.json
        force_reset = data.get('force_reset', True)
        
        logger.info(f"收到重建向量資料庫請求，強制重置: {force_reset}")
        
        # 重建資料庫
        result_db = db_manager.reset_and_rebuild_db()
        
        if result_db:
            logger.info("向量資料庫重建成功")
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
    try:
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

@app.route('/api/rag/info', methods=['POST'])
def get_rag_info():
    """獲取 RAG 檢索信息"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        source_files = data.get('source_files', None)
        
        if not query:
            return jsonify({'error': '查詢內容不可為空'}), 400
        
        # 獲取 RAG 信息
        rag_info = rag_engine.get_rag_info(query, source_files)
        
        return jsonify(rag_info)
        
    except Exception as e:
        logger.error(f"獲取 RAG 信息時出錯: {e}", exc_info=True)
        return jsonify({'error': f'獲取 RAG 信息失敗: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("啟動 Flask 伺服器...")
    app.run(
        host=config.DEFAULT_WEB_HOST,
        port=config.DEFAULT_WEB_PORT,
        debug=config.DEFAULT_WEB_DEBUG,
        threaded=True
    )