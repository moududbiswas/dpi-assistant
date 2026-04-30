import os
import re
import time
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from groq import Groq
from gtts import gTTS
from supabase import create_client

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL = "llama-3.1-8b-instant"


# ==============================
# SMART DATA FETCHING (RAG)
# ==============================
def get_relevant_data(user_question):
    question_lower = user_question.lower()
    data = ""

    try:
        # Routine related
        if any(word in question_lower for word in [
            "রুটিন", "ক্লাস", "routine", "class", "সময়", "পিরিয়ড",
            "কখন", "schedule", "তারিখ", "বার", "দিন"
        ]):
            routines = supabase.table("routines").select(
                "department, shift, semester, group_name, day, period, start_time, end_time, subject, teacher_short, room"
            ).limit(25).execute()
            if routines.data:
                data += "=== ক্লাস রুটিন ===\n"
                for r in routines.data:
                    data += f"{r['department']}|{r['shift']}|{r['semester']}|{r['group_name']}|{r['day']}|{r['period']}|{r['start_time']}-{r['end_time']}|{r['subject']}|{r['teacher_short']}|{r['room']}\n"

        # Teacher related
        if any(word in question_lower for word in [
            "শিক্ষক", "স্যার", "ম্যাম", "teacher", "instructor",
            "প্রভাষক", "অধ্যাপক", "শিক্ষিকা", "কে পড়ান"
        ]):
            teachers = supabase.table("teachers").select(
                "name, subject, short_name, designation"
            ).limit(20).execute()
            if teachers.data:
                data += "=== শিক্ষক তালিকা ===\n"
                for t in teachers.data:
                    data += f"{t['name']} | {t['subject']} | {t['short_name']} | {t['designation']}\n"

        # Notice related
        if any(word in question_lower for word in [
            "নোটিশ", "বিজ্ঞপ্তি", "notice", "circular",
            "ঘোষণা", "জানানো", "সর্বশেষ", "নতুন"
        ]):
            notices = supabase.table("notices").select(
                "title, content, date"
            ).limit(5).order("created_at", desc=True).execute()
            if notices.data:
                data += "=== সাম্প্রতিক নোটিশ ===\n"
                for n in notices.data:
                    content = (n.get('content') or '')[:200]
                    data += f"• {n['title']} ({n['date']}): {content}\n"

        # Location related
        if any(word in question_lower for word in [
            "কোথায়", "রুম", "ওয়াশরুম", "টয়লেট", "ক্যান্টিন",
            "লাইব্রেরি", "where", "room", "কক্ষ", "তলা", "lab", "workshop", "floor"
        ]):
            locations = supabase.table("locations").select(
                "name, description, floor, building"
            ).limit(15).execute()
            if locations.data:
                data += "=== লোকেশন ===\n"
                for l in locations.data:
                    data += f"{l['name']}: {l['description']} | তলা: {l['floor']} | বিল্ডিং: {l['building']}\n"

        # Always include Q&A but limited
        qa = supabase.table("qa").select(
            "question, answer"
        ).limit(10).execute()
        if qa.data:
            data += "\n=== সাধারণ প্রশ্নোত্তর ===\n"
            for item in qa.data:
                data += f"প্রশ্ন: {item['question']}\nউত্তর: {item['answer']}\n\n"

        # Fallback if nothing matched
        if not data:
            data = "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫। সরকারি পলিটেকনিক।"

        print(f"Relevant data size: {len(data)} chars")
        return data

    except Exception as e:
        print(f"Data fetch error: {e}")
        return "ডেটাবেজ সংযোগে সমস্যা হয়েছে।"


# ==============================
# SYSTEM PROMPT
# ==============================
def build_system_prompt(user_question=""):
    relevant_data = get_relevant_data(user_question)
    return """
তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের অফিশিয়াল AI সহকারী। তোমার নাম DPI Assistant।
সবসময় বাংলায় উত্তর দাও।
বন্ধুত্বপূর্ণ, আন্তরিক এবং সহায়ক হও।

=== অত্যন্ত গুরুত্বপূর্ণ নিয়ম ===
তুমি শুধুমাত্র ঢাকা পলিটেকনিক ইনস্টিটিউট সম্পর্কিত প্রশ্নের উত্তর দেবে।
শুধুমাত্র নিচের তথ্য থেকে উত্তর দেবে।
যদি কোনো প্রশ্নের উত্তর নিচের তথ্যে না থাকে বলবে:
"দুঃখিত, এই তথ্যটি আমার কাছে এখনো নেই। আমাদের টিমকে জানান।"
নিজে থেকে কোনো তথ্য তৈরি করবে না বা অনুমান করবে না।

নিচের ধরনের প্রশ্নের উত্তর দেবে না:
- রাজনৈতিক প্রশ্ন
- ধর্মীয় বিতর্ক
- অশ্লীল বা অসামাজিক প্রশ্ন
- ব্যক্তিগত সম্পর্ক বা প্রেম সম্পর্কিত প্রশ্ন
- কলেজের বাইরের যেকোনো বিষয়
- হ্যাকিং বা অবৈধ কাজ

এসব প্রশ্নে বিনয়ের সাথে বলবে:
"দুঃখিত, আমি শুধুমাত্র ঢাকা পলিটেকনিক ইনস্টিটিউট সম্পর্কিত প্রশ্নের উত্তর দিতে পারি।"

=== রুটিন সম্পর্কিত বিশেষ নিয়ম ===
যখন কেউ ক্লাস রুটিন সম্পর্কে জিজ্ঞেস করবে, সরাসরি রুটিন বলবে না।
ধাপ ১: কোন বিভাগের রুটিন জানতে চান?
ধাপ ২: কোন শিফট? ১ম নাকি ২য়?
ধাপ ৩: কোন সেমিস্টার এবং গ্রুপ?
সব তথ্য পাওয়ার পরেই শুধু সেই নির্দিষ্ট রুটিন বলবে।
যদি সেই রুটিনের তথ্য না থাকে বলবে "এই রুটিনটি এখনো আমার কাছে নেই।"

=== কলেজ তথ্য ===
""" + relevant_data


# ==============================
# HELPERS
# ==============================
def clean_for_speech(text):
    text = re.sub(r'[^\w\s\u0980-\u09FF\u0020-\u007E]', '', text)
    text = re.sub(r'[\*\#\_\>\-\=\~\`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_response(messages):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return "দুঃখিত, এই মুহূর্তে উত্তর দিতে পারছি না। একটু পরে আবার চেষ্টা করুন।"


def save_conversation(user_message, bot_reply):
    try:
        supabase.table("conversations").insert({
            "user_message": user_message,
            "bot_reply": bot_reply
        }).execute()
    except Exception as e:
        print(f"Conversation save error: {e}")


# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data["message"]
    history = data.get("history", [])

    # Limit history to last 4 messages only
    history = history[-4:]

    # Smart fetch — only relevant data based on question
    messages = [{"role": "system", "content": build_system_prompt(user_input)}]
    messages += history
    messages.append({"role": "user", "content": user_input})

    reply = get_response(messages)

    # Save conversation
    save_conversation(user_input, reply)

    # Generate audio
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