import os
import re
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from groq import Groq
from gtts import gTTS
from college_data import college_info

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Auto fallback models
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

system_prompt = """
তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের অফিশিয়াল AI সহকারী। তোমার নাম DPI Assistant।
সবসময় বাংলায় উত্তর দাও।
বন্ধুত্বপূর্ণ, আন্তরিক এবং সহায়ক হও।
শুধুমাত্র কলেজ সম্পর্কিত প্রশ্নের উত্তর দাও।

=== রুটিন সম্পর্কিত বিশেষ নিয়ম ===
যখন কেউ ক্লাস রুটিন সম্পর্কে জিজ্ঞেস করবে, তুমি সরাসরি রুটিন বলবে না।
প্রথমে এই তিনটি প্রশ্ন একে একে করবে:
ধাপ ১: কোন বিভাগের রুটিন জানতে চান? যেমন: ইলেকট্রিক্যাল, কম্পিউটার, সিভিল?
ধাপ ২: কোন শিফট? ১ম শিফট নাকি ২য় শিফট?
ধাপ ৩: কোন সেমিস্টার এবং গ্রুপ? যেমন: ৫ম সেমিস্টার, গ্রুপ C?
সব তথ্য পাওয়ার পরেই শুধু সেই নির্দিষ্ট রুটিন বলবে।
যদি সেই রুটিনের তথ্য না থাকে বলবে "এই রুটিনটি এখনো আমার কাছে নেই।"
""" + college_info

def clean_for_speech(text):
    text = re.sub(r'[^\w\s\u0980-\u09FF\u0020-\u007E]', '', text)
    text = re.sub(r'[\*\#\_\>\-\=\~\`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_response(messages):
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"Model {model} rate limited, trying next...")
                continue
            raise e
    return "দুঃখিত, এই মুহূর্তে সকল মডেল ব্যস্ত। কিছুক্ষণ পর আবার চেষ্টা করুন।"

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

    reply = get_response(messages)

    # Generate audio in memory
    clean_reply = clean_for_speech(reply)
    tts = gTTS(text=clean_reply, lang='bn')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

    return jsonify({"reply": reply, "audio": audio_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)