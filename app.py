import os
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from supabase import create_client

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set in environment variables!")
else:
    print("GEMINI_API_KEY loaded OK")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"  # stable pinned model name

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ==============================
# SMART DATA FETCHING (RAG)
# ==============================
def get_relevant_data(user_question):
    q = user_question.lower()
    data = ""

    try:
        # Routine
        if any(w in q for w in [
            "রুটিন", "ক্লাস", "routine", "class", "সময়", "পিরিয়ড",
            "কখন", "schedule", "তারিখ", "বার", "দিন"
        ]):
            rows = supabase.table("routines").select(
                "department,shift,semester,group_name,day,period,start_time,end_time,subject,teacher_short,room"
            ).limit(25).execute()
            if rows.data:
                data += "=== ক্লাস রুটিন ===\n"
                for r in rows.data:
                    data += f"{r['department']}|{r['shift']}|{r['semester']}|{r['group_name']}|{r['day']}|{r['period']}|{r['start_time']}-{r['end_time']}|{r['subject']}|{r['teacher_short']}|{r['room']}\n"

        # Teacher — expanded keywords + partial matching
        if any(w in q for w in [
            "শিক্ষক", "স্যার", "ম্যাম", "teacher", "instructor",
            "প্রভাষক", "অধ্যাপক", "শিক্ষিকা", "পড়ান", "পড়াচ্ছেন",
            "কে পড়া", "স্যারের", "ম্যামের", "কোন স্যার", "কোন শিক্ষক"
        ]):
            rows = supabase.table("teachers").select(
                "name,subject,short_name,designation"
            ).limit(20).execute()
            if rows.data:
                data += "=== শিক্ষক তালিকা ===\n"
                for t in rows.data:
                    data += f"{t['name']} | {t['subject']} | {t['short_name']} | {t['designation']}\n"

        # Notice
        if any(w in q for w in [
            "নোটিশ", "বিজ্ঞপ্তি", "notice", "circular",
            "ঘোষণা", "সর্বশেষ", "নতুন"
        ]):
            rows = supabase.table("notices").select(
                "title,content,date"
            ).order("created_at", desc=True).limit(5).execute()
            if rows.data:
                data += "=== সাম্প্রতিক নোটিশ ===\n"
                for n in rows.data:
                    content = (n.get("content") or "")[:200]
                    data += f"• {n['title']} ({n['date']}): {content}\n"

        # Location
        if any(w in q for w in [
            "কোথায়", "রুম", "ওয়াশরুম", "টয়লেট", "ক্যান্টিন",
            "লাইব্রেরি", "where", "room", "কক্ষ", "তলা", "floor",
            "lab", "laboratory", "center", "centre", "wiring",
            "hardware", "electrical", "ল্যাব", "কেন্দ্র"
        ]):
            rows = supabase.table("locations").select(
                "name,description,floor,building"
            ).limit(97).execute()
            if rows.data:
                data += "=== লোকেশন ===\n"
                for l in rows.data:
                    data += f"{l['name']}: {l['description']} | তলা: {l['floor']} | বিল্ডিং: {l['building']}\n"

        # Q&A always included
        qa = supabase.table("qa").select("question,answer").limit(100).execute()
        if qa.data:
            data += "\n=== প্রশ্নোত্তর ===\n"
            for item in qa.data:
                data += f"প্রশ্ন: {item['question']}\nউত্তর: {item['answer']}\n\n"

        if not data:
            data = "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫।"

        print(f"RAG data: {len(data)} chars")
        return data

    except Exception as e:
        print(f"Data fetch error: {e}")
        return "ডেটাবেজ সংযোগে সমস্যা।"


# ==============================
# SYSTEM PROMPT
# ==============================
def build_system_prompt(user_question=""):
    relevant_data = get_relevant_data(user_question)
    return f"""তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের AI সহকারী, নাম DPI Assistant। সবসময় বাংলায় উত্তর দাও।
কাউকে স্বাগত জানানোর সময় বা প্রথম বার্তায় সালাম দাও: "আসসালামু আলাইকুম" — কখনো "নমস্কার" বলবে না।

নিয়ম:
- শুধুমাত্র নিচের তথ্য থেকে উত্তর দাও
- তথ্য না থাকলে বলো: "দুঃখিত, এই তথ্যটি আমার কাছে নেই। টিমকে জানান।"
- রাজনীতি, ধর্মীয় বিতর্ক, অশ্লীল, প্রেম বা কলেজ-বহির্ভূত প্রশ্নে বলো: "আমি শুধু DPI সম্পর্কিত প্রশ্নের উত্তর দিতে পারি।"
- রুটিন জিজ্ঞেস করলে ধাপে ধাপে জিজ্ঞেস করো: বিভাগ → শিফট → সেমিস্টার ও গ্রুপ
- শিক্ষক সম্পর্কে জিজ্ঞেস করলে আগে জিজ্ঞেস করো: কোন বিভাগ? কোন শিফট? (একই নামে একাধিক শিক্ষক থাকতে পারেন)

=== তথ্য ===
{relevant_data}"""


# ==============================
# HELPERS
# ==============================
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
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "missing message"}), 400

        user_input = data["message"]
        history = data.get("history", [])[-4:]

        print(f"User input: {user_input[:80]}")

        system_prompt = build_system_prompt(user_input)
        reply = get_response(system_prompt, history, user_input)

        print(f"Reply: {reply[:80]}")

        save_conversation(user_input, reply)

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"CRITICAL /ask error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"reply": "দুঃখিত, সার্ভারে সমস্যা হয়েছে। একটু পরে চেষ্টা করুন।"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)