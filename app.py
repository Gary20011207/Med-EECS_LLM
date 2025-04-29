from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
import sqlite3
import os
import sys
from werkzeug.security import generate_password_hash, check_password_hash

# Check for WEB parameter
WEB_DEV = "WEB" in sys.argv

if not WEB_DEV:
    from apps.RAG_NAIVE2 import generate_reply

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

    chats = conn.execute('''
        SELECT users.username, chat_history.sender, chat_history.message, chat_history.timestamp
        FROM chat_history
        JOIN users ON users.id = chat_history.user_id
        ORDER BY chat_history.timestamp ASC
    ''').fetchall()

    conn.close()
    return render_template("admin.html", chats=chats)

# User registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            return "Username already exists!"
    return render_template("register.html")

# User login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("chat"))
        else:
            return "Invalid username or password!"
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# Chat page (must be logged in)
@app.route("/")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["username"])

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
    
    # 因為沒有選PDF了，直接預設問全部
    selected_pdf = "All PDFs"

    history = []
    if WEB_DEV:
        bot_reply = chatbot_response(user_input)
    else:
        new_history, _ = generate_reply(user_input, selected_pdf, history)
        bot_reply = new_history[-1][1]

    conn = get_db_connection()
    conn.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (session["user_id"], "user", user_input))
    conn.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (session["user_id"], "bot", bot_reply))
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

if __name__ == "__main__":
    # Initialize the database if it doesn't exist
    if not os.path.exists(DB_PATH):
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                dob DATE,
                surgery_type TEXT,
                current_expert INTEGER DEFAULT 0 NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.execute('''
            INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)
        ''', ("admin", generate_password_hash("nimda"), 1))
        conn.commit()
        conn.close()
        print("Database initialized.")
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)