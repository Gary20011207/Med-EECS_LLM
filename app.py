from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort, Response, stream_with_context
import sqlite3
import os
import sys
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import logging
import uuid

# 設置日誌
logging.basicConfig(
    level=logging.DEBUG,
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

# Check for WEB parameter
WEB_DEV = "WEB" in sys.argv

if not WEB_DEV:
    # from apps.RAG_MEM import generate_reply
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
app.secret_key = 'your-secret-key'
DB_PATH = "chat.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

# Admin dashboard
@app.route("/admin")
def admin_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    if not user or user["is_admin"] != 1:
        conn.close()
        return abort(403)  # Forbidden

    users = conn.execute("SELECT id, name FROM users WHERE is_admin != 1").fetchall()
    conn.close()
    return render_template("admin.html", users=users)

@app.route("/admin/user/<int:user_id>")
def view_user_details(user_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    admin_user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    if not admin_user or admin_user["is_admin"] != 1:
        conn.close()
        return abort(403)  # Forbidden

    user = conn.execute("SELECT id, name, dob, attending_physician, current_expert FROM users WHERE id = ?", (user_id,)).fetchone()
    chat_history = conn.execute(
        "SELECT sender, message, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    ).fetchall()
    todo_list = conn.execute(
        "SELECT id, task, completed FROM todo_list WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    conn.close()

    if not user:
        return abort(404)  # User not found

    return render_template("user_details.html", user=user, chat_history=chat_history, todo_list=todo_list)

@app.route("/admin/add_todo/<int:user_id>", methods=["POST"])
def admin_add_todo(user_id):
    if "user_id" not in session or session.get("is_admin") != 1:
        return abort(403)  # Forbidden

    task = request.form.get("task")
    if not task:
        return "Task cannot be empty!", 400

    add_todo_item(user_id, task)
    return redirect(url_for("view_user_details", user_id=user_id))

@app.route("/admin/complete_todo/<int:user_id>/<int:item_id>", methods=["POST"])
def admin_complete_todo(user_id, item_id):
    if "user_id" not in session or session.get("is_admin") != 1:
        return abort(403)
    # Forbidden
    update_todo_item(item_id, 1)
    return redirect(url_for("view_user_details", user_id=user_id))

@app.route("/admin/uncomplete_todo/<int:user_id>/<int:item_id>", methods=["POST"])
def admin_uncomplete_todo(user_id, item_id):
    if "user_id" not in session or session.get("is_admin") != 1:
        return abort(403)
    # Forbidden
    update_todo_item(item_id, 0)
    return redirect(url_for("view_user_details", user_id=user_id))

@app.route("/admin/delete_todo/<int:user_id>/<int:item_id>", methods=["POST"])
def admin_delete_todo(user_id, item_id):
    if "user_id" not in session or session.get("is_admin") != 1:
        return abort(403)
    # Forbidden
    delete_todo_item(item_id)
    return redirect(url_for("view_user_details", user_id=user_id))

@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
def admin_delete_user(user_id):
    if "user_id" not in session or session.get("is_admin") != 1:
        return abort(403)
    # Forbidden
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM todo_list WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

# User registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        record_id = request.form["record_id"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)
        name = request.form["name"]
        attending_physician = request.form["attending_physician"]
        surgery_date = request.form["surgery_date"]
        gender = request.form["gender"]
        dob = request.form["dob"]
        phone = request.form["phone"]
        height = request.form["height"]
        weight = request.form["weight"]
        signup_blood_type = request.form["signup-blood-type"]
        signup_allergies = request.form["signup-allergies"]
        signup_email = request.form["signup-email"]
        signup_surgery_type = request.form["signup-surgery-type"]
        signup_diagnosis = request.form["signup-diagnosis"]
        signup_comorbidities = request.form["signup-comorbidities"]

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (record_id, password, name, attending_physician, surgery_date, gender, dob, phone, height, weight, signup_blood_type, signup_allergies, signup_email, signup_surgery_type, signup_diagnosis, signup_comorbidities) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record_id, hashed_pw, name, attending_physician, surgery_date, gender, dob, phone, height, weight, signup_blood_type, signup_allergies, signup_email, signup_surgery_type, signup_diagnosis, signup_comorbidities)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("admin_dashboard"))
        except sqlite3.IntegrityError:
            conn.close()
            return "病歷號已經存在!"
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template("register.html", current_date=current_date)

# User profile
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    if request.method == "POST":
        dob = request.form["dob"]
        attending_physician = request.form["attending_physician"]

        conn.execute("UPDATE users SET dob = ?, attending_physician = ? WHERE id = ?",
                     (dob, attending_physician, session["user_id"]))
        conn.commit()
        conn.close()
        return redirect(url_for("chat"))

    conn.close()
    return render_template("profile.html", user=user)

# User login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        record_id = request.form["record_id"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE record_id = ?", (record_id,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["record_id"] = user["record_id"]
            session["name"] = user["name"]
            session["is_admin"] = user["is_admin"]
            if user["is_admin"] == 1:
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("welcome"))
        else:
            return "病歷號或是密碼錯誤!"
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def welcome():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()

    if not user:
        return redirect(url_for("login"))

    return render_template("welcome.html", user=user)

# Get chat history
@app.route("/get_history")
def get_history():
    if "user_id" not in session:
        return jsonify([])

    conn = get_db_connection()
    messages = conn.execute(
        "SELECT sender, message FROM chat_history WHERE user_id = ? ORDER BY id",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify([{"sender": row["sender"], "message": row["message"]} for row in messages])

@app.route("/get_history/<doctor_name>")
def get_doctor_chat_history(doctor_name):
    if "user_id" not in session:
        return jsonify([])
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT sender, message FROM chat_history WHERE user_id = ? AND doctor_name = ? ORDER BY id",
        (session["user_id"], doctor_name)
    ).fetchall()
    conn.close()
    return jsonify([{"sender": row["sender"], "message": row["message"]} for row in rows])

# Get available PDFs
@app.route("/get_pdfs")
def get_pdfs():
    pdfs = [f for f in os.listdir("./PDFS") if f.endswith(".pdf")]
    pdfs = ["All PDFs"] + sorted(pdfs)
    return jsonify(pdfs)

def chatbot_response(user_input):
     if "hello" in user_input.lower():
         return "Hi there! How can I help you?"
     elif "bye" in user_input.lower():
         return "Goodbye! Have a nice day!"
     else:
         return "I'm just a simple bot, but I'm learning!"
     
# Functions for user data from database
def get_all_users():
    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return users
def get_user_by_record_id(record_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE record_id = ?", (record_id,)).fetchone()
    conn.close()
    return user
def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user
def get_chat_history(user_id):
    conn = get_db_connection()
    chat_history = conn.execute("SELECT * FROM chat_history WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return chat_history
def get_dob(user_id):
    conn = get_db_connection()
    dob = conn.execute("SELECT dob FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dob
def get_attending_physician(user_id):
    conn = get_db_connection()
    attending_physician = conn.execute("SELECT attending_physician FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return attending_physician
def get_current_expert(user_id): # Default (0), 營養師(1)、護理(2)、復健科(3)、麻醉科(4)、手術醫師(5)
    conn = get_db_connection()
    current_expert = conn.execute("SELECT current_expert FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return current_expert
def get_todo_list(user_id):
    conn = get_db_connection()
    todo_list = conn.execute("SELECT * FROM todo_list WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return todo_list
def add_todo_item(user_id, task):
    conn = get_db_connection()
    conn.execute("INSERT INTO todo_list (user_id, task) VALUES (?, ?)", (user_id, task))
    conn.commit()
    conn.close()
def update_todo_item(item_id, completed):
    conn = get_db_connection()
    conn.execute("UPDATE todo_list SET completed = ? WHERE id = ?", (completed, item_id))
    conn.commit()
    conn.close()
def delete_todo_item(item_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM todo_list WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

@app.route("/get_todo_list")
def get_todo_list_route():
    if "user_id" not in session:
        return jsonify([])

    todo_list = get_todo_list(session["user_id"])
    return jsonify([{"id": item["id"], "task": item["task"], "completed": bool(item["completed"])} for item in todo_list])

@app.route("/add_todo", methods=["POST"])
def add_todo_route():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    task = data.get("task")
    if not task:
        return jsonify({"error": "Task cannot be empty"}), 400

    add_todo_item(session["user_id"], task)
    return jsonify({"success": True})

@app.route("/update_todo/<int:item_id>", methods=["POST"])
def update_todo_route(item_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    completed = data.get("completed")
    if completed is None:
        return jsonify({"error": "Invalid request"}), 400

    update_todo_item(item_id, int(completed))
    return jsonify({"success": True})

@app.route("/add_doctor", methods=["POST"])
def add_doctor():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    name = data.get("name")
    avatar = data.get("avatar", "")
    if not name:
        return jsonify({"error": "Doctor name required"}), 400
    conn = get_db_connection()
    conn.execute("INSERT INTO doctor_rooms (user_id, doctor_name, avatar) VALUES (?, ?, ?)",
                 (session["user_id"], name, avatar))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/get_doctor_list")
def get_doctor_list():
    if "user_id" not in session:
        return jsonify([])
    conn = get_db_connection()
    doctors = conn.execute("SELECT doctor_name, avatar FROM doctor_rooms WHERE user_id = ?", (session["user_id"],)).fetchall()
    conn.close()
    return jsonify([dict(d) for d in doctors])

@app.route("/finish_chat", methods=["POST"])
def finish_chat():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    docter = data.get("docter")
    if not docter:
        return jsonify({"error": "Doctor name required"}), 400
    
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_db_connection()
    if docter == "anesthesiology":
        conn.execute("UPDATE users SET anesthesiology = ? WHERE id = ?", (today, session["user_id"],))
    if docter == "nutritionist":
        conn.execute("UPDATE users SET nutritionist = ? WHERE id = ?", (today, session["user_id"],))
    if docter == "pharmacist":
        conn.execute("UPDATE users SET pharmacist = ? WHERE id = ?", (today, session["user_id"],))
    if docter == "rehab":
        conn.execute("UPDATE users SET rehab = ? WHERE id = ?", (today, session["user_id"],))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": f"已結束與 {docter} 的對話"})

if __name__ == "__main__":
    # Initialize the database if it doesn't exist
    if not os.path.exists(DB_PATH):
        conn = get_db_connection()
        # Create a new SQLite database
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                is_admin INTEGER DEFAULT 0,
                record_id TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT,
                attending_physician TEXT NOT NULL,
                surgery_date DATE NOT NULL,
                gender TEXT DEFAULT NULL,
                dob DATE,
                phone TEXT,
                height REAL DEFAULT NULL,
                weight REAL DEFAULT NULL,
                signup_blood_type TEXT DEFAULT NULL,
                signup_allergies TEXT DEFAULT NULL,
                signup_email TEXT DEFAULT NULL,
                signup_surgery_type TEXT DEFAULT NULL,
                signup_diagnosis TEXT DEFAULT NULL,
                signup_comorbidities TEXT DEFAULT NULL,
                anesthesiology DATE DEFAULT NULL,
                nutritionist DATE DEFAULT NULL,
                pharmacist DATE DEFAULT NULL,
                rehab DATE DEFAULT NULL
            )
        ''')
        # Create a table for chat history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                doctor_name TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        # Create a table for todo list
        conn.execute('''
            CREATE TABLE IF NOT EXISTS todo_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Create an admin user
        conn.execute('''
            INSERT INTO users (is_admin, record_id, password, name, attending_physician, surgery_date) VALUES (?, ?, ?, ?, ?, ?)
        ''', (1, "admin", generate_password_hash("nimda"), "admin", "attending_physician", "2025-05-27"))
        conn.commit()
        conn.close()
        print("Database initialized.")
    # else:
    #     ensure_chat_history_column()
    # Run the Flask app
    logger.info("啟動 Flask 伺服器...")
    app.run(
        host=config.DEFAULT_WEB_HOST,
        port=config.DEFAULT_WEB_PORT,
        debug=(config.DEFAULT_WEB_DEBUG if not WEB_DEV else True),
        threaded=True
    )