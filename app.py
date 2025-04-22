from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key'
DB_PATH = "chat.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

# Chat route (must be logged in)
@app.route("/")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["username"])

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

@app.route("/send_message", methods=["POST"])
def send_message():
    if "user_id" not in session:
        return jsonify({"reply": "Unauthorized"}), 401

    user_input = request.json.get("message")
    bot_reply = chatbot_response(user_input)

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
    app.run(host="0.0.0.0", port=5000, debug=True)
