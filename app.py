import sqlite3
import os
import re
from flask import Flask, render_template, request, jsonify
from groq import Groq
from college_data import college_info

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
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = """
তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের অফিশিয়াল AI সহকারী। তোমার নাম DPI Assistant।
সবসময় বাংলায় উত্তর দাও।
বন্ধুত্বপূর্ণ, আন্তরিক এবং সহায়ক হও।
শুধুমাত্র কলেজ সম্পর্কিত প্রশ্নের উত্তর দাও।

=== রুটিন সম্পর্কিত বিশেষ নিয়ম ===
যখন কেউ ক্লাস রুটিন সম্পর্কে জিজ্ঞেস করবে, তুমি সরাসরি রুটিন বলবে না।
ধাপ ১: কোন বিভাগের রুটিন জানতে চান?
ধাপ ২: কোন শিফট?
ধাপ ৩: কোন সেমিস্টার এবং গ্রুপ?
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

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)