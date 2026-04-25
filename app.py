import os
import re
from flask import Flask, render_template, request, jsonify
from groq import Groq
from college_data import college_info

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
    data = request.json
    user_input = data["message"]
    history = data.get("history", [])

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    reply = response.choices[0].message.content
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)