import sqlite3
import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from gtts import gTTS
from college_data import college_info

# Database setup
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    message TEXT,
    reply TEXT
)
""")
conn.commit()

app = Flask(__name__)

# API key comes from environment variable - never hardcoded!
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = """
তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের অফিশিয়াল AI সহকারী। তোমার নাম DPI Assistant।
সবসময় বাংলায় উত্তর দাও।
বন্ধুত্বপূর্ণ, আন্তরিক এবং সহায়ক হও।
শুধুমাত্র কলেজ সম্পর্কিত প্রশ্নের উত্তর দাও।
নিচের তথ্য ব্যবহার করে উত্তর দাও।
""" + college_info

chat_history = [{"role": "system", "content": system_prompt}]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]
    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=chat_history
    )

    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})

    cursor.execute(
        "INSERT INTO messages (user, message, reply) VALUES (?, ?, ?)",
        ("anonymous", user_input, reply)
    )
    conn.commit()

 tts = gTTS(text=reply, lang='bn')
tts.save("static/reply.mp3")

return jsonify({"reply": reply, "audio": "/static/reply.mp3"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)