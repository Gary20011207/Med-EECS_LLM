from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
import sqlite3
import os
import sys
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Check for WEB parameter
WEB_DEV = "WEB" in sys.argv

if not WEB_DEV:
    from apps.RAG_MEM import generate_reply

app = Flask(__name__)
app.secret_key = 'your-secret-key'
DB_PATH = "chat.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
        password = record_id
        hashed_pw = generate_password_hash(password)
        name = request.form["name"]
        attending_physician = request.form["attending_physician"]
        hospital_stay_date = request.form["hospital_stay_date"]
        surgery_date = request.form["surgery_date"]

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (record_id, password, name, attending_physician, hospital_stay_date, surgery_date) VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, hashed_pw, name, attending_physician, hospital_stay_date, surgery_date)
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

# Chat page (must be logged in)
@app.route("/chat")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html")

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

# Send a message
@app.route("/send_message", methods=["POST"])
def send_message():
    if "user_id" not in session:
        return jsonify({"reply": "Unauthorized"}), 401

    data = request.json
    user_input = data.get("message")
    doctor_name = data.get("doctor") or "default"  # 預設值

    conn = get_db_connection()
    rows = conn.execute(
        "SELECT sender, message FROM chat_history WHERE user_id = ? AND doctor_name = ? ORDER BY id DESC LIMIT 10",
        (session["user_id"], doctor_name)
    ).fetchall()
    conn.close()

    rows.reverse()  # 從舊到新
    history = [{"role": row["sender"], "content": row["message"]} for row in rows]
    if WEB_DEV:
        bot_reply = chatbot_response(user_input)
    else:
        new_history, _ = generate_reply(user_input, "All PDFs", True, history)
        bot_reply = new_history[-1][1]

    conn = get_db_connection()
    conn.execute("INSERT INTO chat_history (user_id, sender, message, doctor_name) VALUES (?, ?, ?, ?)", 
                 (session["user_id"], "user", user_input, doctor_name))
    conn.execute("INSERT INTO chat_history (user_id, sender, message, doctor_name) VALUES (?, ?, ?, ?)", 
                 (session["user_id"], "bot", bot_reply, doctor_name))
    conn.commit()
    conn.close()

    return jsonify({"reply": bot_reply})

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

# def ensure_chat_history_column():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("PRAGMA table_info(chat_history)")
#     columns = [row["name"] for row in cursor.fetchall()]
#     if "doctor_name" not in columns:
#         print("Adding missing column: doctor_name")
#         cursor.execute("ALTER TABLE chat_history ADD COLUMN doctor_name TEXT")
#         conn.commit()
#     conn.close()

if __name__ == "__main__":
    # Initialize the database if it doesn't exist
    if not os.path.exists(DB_PATH):
        conn = get_db_connection()
        # Create a new SQLite database
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                name TEXT,
                dob DATE,
                attending_physician TEXT NOT NULL,
                hospital_stay_date DATE NOT NULL,
                surgery_date DATE NOT NULL,
                phone TEXT,
                clinic_case_intake DATE DEFAULT NULL,
                clinic_measure_height DATE DEFAULT NULL,
                height REAL DEFAULT NULL,
                clinic_measure_weight DATE DEFAULT NULL,
                weight REAL DEFAULT NULL,
                clinic_provide_preop_education_and_consent DATE DEFAULT NULL,
                clinic_anxiety_survey_preop DATE DEFAULT NULL,
                clinic_instruction_satisfaction_survey_preop DATE DEFAULT NULL,
                anesthesiology_preop_assessment_and_explanation DATE DEFAULT NULL,
                nutritionist_dietary_assessment_and_guidance DATE DEFAULT NULL,
                pharmacist_medication_guidance DATE DEFAULT NULL,
                rehab_preop_functional_and_balance_assessment DATE DEFAULT NULL,
                rehab_preop_exercise_guidance DATE DEFAULT NULL,
                ward_nurse_education DATE DEFAULT NULL,
                ward_anxiety_survey_postop DATE DEFAULT NULL,
                ward_instruction_satisfaction_survey_postop DATE DEFAULT NULL,
                current_expert INTEGER DEFAULT 0 NOT NULL,
                has_completed_profile INTEGER DEFAULT 0 NOT NULL
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
            INSERT INTO users (record_id, password, attending_physician, hospital_stay_date, surgery_date, is_admin) VALUES (?, ?, ?, ?, ?, ?)
        ''', ("admin", generate_password_hash("nimda"), "nimda", "2025-05-12", "2025-05-12", 1))
        conn.commit()
        conn.close()
        print("Database initialized.")
    # else:
    #     ensure_chat_history_column()
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)