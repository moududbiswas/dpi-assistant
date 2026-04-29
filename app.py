import os
import re
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from groq import Groq
from gtts import gTTS
from supabase import create_client
from apscheduler.schedulers.background import BackgroundScheduler
from scraper import run_scraper


app = Flask(__name__)

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

scheduler = BackgroundScheduler()
scheduler.add_job(run_scraper, 'interval', hours=6)
scheduler.start()

MODEL = "llama-3.3-70b-versatile"


def get_college_data():
    try:
        qa = supabase.table("qa").select("question, answer").execute()
        teachers = supabase.table("teachers").select("*").execute()
        routines = supabase.table("routines").select("*").execute()
        notices = supabase.table("notices").select("*").execute()
        locations = supabase.table("locations").select("*").execute()

        data = ""

        if qa.data:
            data += "=== প্রশ্নোত্তর ===\n"
            for item in qa.data:
                data += f"প্রশ্ন: {item['question']}\nউত্তর: {item['answer']}\n\n"

        if teachers.data:
            data += "=== শিক্ষক তালিকা ===\n"
            for t in teachers.data:
                data += f"নাম: {t['name']} | বিষয়: {t['subject']} | সংক্ষেপ: {t['short_name']} | পদবি: {t['designation']}\n"

        if routines.data:
            data += "\n=== ক্লাস রুটিন ===\n"
            for r in routines.data:
                data += f"বিভাগ: {r['department']} | শিফট: {r['shift']} | সেমিস্টার: {r['semester']} | গ্রুপ: {r['group_name']} | দিন: {r['day']} | পিরিয়ড: {r['period']} | সময়: {r['start_time']} - {r['end_time']} PM | বিষয়: {r['subject']} | শিক্ষক: {r['teacher_short']} | রুম: {r['room']}\n"

        if notices.data:
            data += "\n=== নোটিশ ===\n"
            for n in notices.data:
                data += f"তারিখ: {n['date']} | শিরোনাম: {n['title']} | বিবরণ: {n['content']}\n"

        if locations.data:
            data += "\n=== লোকেশন ===\n"
            for l in locations.data:
                data += f"{l['name']}: {l['description']} | তলা: {l['floor']} | বিল্ডিং: {l['building']}\n"

        return data

    except Exception as e:
        print(f"Database error: {e}")
        return "ডেটাবেজ সংযোগে সমস্যা হয়েছে।"

def build_system_prompt():
    college_data = get_college_data()
    return """
তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের অফিশিয়াল AI সহকারী। তোমার নাম DPI Assistant।
সবসময় বাংলায় উত্তর দাও।
বন্ধুত্বপূর্ণ, আন্তরিক এবং সহায়ক হও।

=== অত্যন্ত গুরুত্বপূর্ণ নিয়ম ===
তুমি শুধুমাত্র নিচের তথ্য থেকে উত্তর দেবে।
যদি কোনো প্রশ্নের উত্তর নিচের তথ্যে না থাকে বলবে:
"দুঃখিত, এই তথ্যটি আমার কাছে এখনো নেই। আমাদের টিমকে জানান।"
নিজে থেকে কোনো তথ্য তৈরি করবে না বা অনুমান করবে না।
কলেজের বাইরের কোনো প্রশ্নের উত্তর দেবে না।

=== রুটিন সম্পর্কিত বিশেষ নিয়ম ===
যখন কেউ ক্লাস রুটিন সম্পর্কে জিজ্ঞেস করবে, সরাসরি রুটিন বলবে না।
ধাপ ১: কোন বিভাগের রুটিন জানতে চান?
ধাপ ২: কোন শিফট? ১ম নাকি ২য়?
ধাপ ৩: কোন সেমিস্টার এবং গ্রুপ?
সব তথ্য পাওয়ার পরেই শুধু সেই নির্দিষ্ট রুটিন বলবে।
যদি সেই রুটিনের তথ্য না থাকে বলবে "এই রুটিনটি এখনো আমার কাছে নেই।"

=== কলেজ তথ্য ===
""" + college_data

def clean_for_speech(text):
    text = re.sub(r'[^\w\s\u0980-\u09FF\u0020-\u007E]', '', text)
    text = re.sub(r'[\*\#\_\>\-\=\~\`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_response(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data["message"]
    history = data.get("history", [])

    messages = [{"role": "system", "content": build_system_prompt()}]
    messages += history
    messages.append({"role": "user", "content": user_input})

    reply = get_response(messages)

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