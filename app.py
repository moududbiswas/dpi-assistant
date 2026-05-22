import os
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from supabase import create_client
 
app = Flask(__name__)
 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set!", flush=True)
else:
    print("GEMINI_API_KEY loaded OK", flush=True)
 
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
 
 
# ==============================
# KEYWORD EXTRACTOR
# ==============================
def extract_context(q):
    """Extract department, shift, semester, day, floor, building from question"""
 
    dept_map = {
        "সিভিল": "civil", "civil": "civil",
        "ইলেকট্রিক্যাল": "electrical", "ইলেক্ট্রিক্যাল": "electrical", "electrical": "electrical",
        "কম্পিউটার": "computer", "computer": "computer",
        "মেকানিক্যাল": "mechanical", "mechanical": "mechanical",
        "আর্কিটেকচার": "architecture", "architecture": "architecture",
        "ইলেকট্রনিক্স": "electronics", "electronics": "electronics",
        "কেমিক্যাল": "chemical", "chemical": "chemical",
        "টেক্সটাইল": "textile", "textile": "textile",
    }
 
    shift_map = {
        "প্রথম": "1st", "1st": "1st", "first": "1st", "মর্নিং": "1st", "morning": "1st",
        "দ্বিতীয়": "2nd", "2nd": "2nd", "second": "2nd", "ডে": "2nd", "day": "2nd",
    }
 
    semester_map = {
        "১ম": "1st", "১": "1st", "প্রথম সেমিস্টার": "1st",
        "২য়": "2nd", "২": "2nd", "দ্বিতীয় সেমিস্টার": "2nd",
        "৩য়": "3rd", "৩": "3rd", "তৃতীয় সেমিস্টার": "3rd",
        "৪র্থ": "4th", "৪": "4th", "চতুর্থ সেমিস্টার": "4th",
        "৫ম": "5th", "৫": "5th", "পঞ্চম সেমিস্টার": "5th",
        "৬ষ্ঠ": "6th", "৬": "6th", "ষষ্ঠ সেমিস্টার": "6th",
        "৭ম": "7th", "৭": "7th", "সপ্তম সেমিস্টার": "7th",
        "৮ম": "8th", "৮": "8th", "অষ্টম সেমিস্টার": "8th",
    }
 
    day_map = {
        "রবিবার": "sunday", "sunday": "sunday",
        "সোমবার": "monday", "monday": "monday",
        "মঙ্গলবার": "tuesday", "tuesday": "tuesday",
        "বুধবার": "wednesday", "wednesday": "wednesday",
        "বৃহস্পতিবার": "thursday", "thursday": "thursday",
        "শুক্রবার": "friday", "friday": "friday",
        "শনিবার": "saturday", "saturday": "saturday",
    }
 
    floor_map = {
        "নিচ তলা": "ground", "গ্রাউন্ড": "ground", "ground": "ground",
        "১ম তলা": "1st", "দ্বিতীয় তলা": "2nd", "তৃতীয় তলা": "3rd",
        "চতুর্থ তলা": "4th", "পঞ্চম তলা": "5th",
        "1st floor": "1st", "2nd floor": "2nd", "3rd floor": "3rd",
    }
 
    ctx = {
        "department": None, "shift": None, "semester": None,
        "day": None, "floor": None,
        "short_names": re.findall(r'\b[A-Z]{2,5}\b', q),
    }
 
    for k, v in dept_map.items():
        if k in q:
            ctx["department"] = v
            break
    for k, v in shift_map.items():
        if k in q:
            ctx["shift"] = v
            break
    for k, v in semester_map.items():
        if k in q:
            ctx["semester"] = v
            break
    for k, v in day_map.items():
        if k in q:
            ctx["day"] = v
            break
    for k, v in floor_map.items():
        if k in q:
            ctx["floor"] = v
            break
 
    return ctx
 
 
# ==============================
# DYNAMIC TEACHER SEARCH
# ==============================
def search_teachers(q_raw, ctx):
    try:
        results = []
 
        # If short name found (e.g. SSS, MAA), search by that
        if ctx["short_names"]:
            for sn in ctx["short_names"]:
                query = supabase.table("teachers").select(
                    "name,subject,short_name,designation,department,shift"
                ).or_(
                    f"short_name.ilike.%{sn}%,"
                    f"name.ilike.%{sn}%"
                )
                if ctx["department"]:
                    query = query.ilike("department", f"%{ctx['department']}%")
                if ctx["shift"]:
                    r = query.eq("shift", ctx["shift"]).execute()
                    if r.data:
                        results.extend(r.data)
                        continue
                    # Fallback without shift
                r = query.execute()
                results.extend(r.data or [])
 
        # General search by department
        if not results:
            query = supabase.table("teachers").select(
                "name,subject,short_name,designation,department,shift"
            )
            if ctx["department"]:
                query = query.ilike("department", f"%{ctx['department']}%")
            if ctx["shift"]:
                r = query.eq("shift", ctx["shift"]).execute()
                if r.data:
                    results = r.data
                else:
                    results = query.execute().data or []
            else:
                results = query.limit(60).execute().data or []
 
        # Deduplicate
        seen = set()
        unique = []
        for t in results:
            if t["name"] not in seen:
                seen.add(t["name"])
                unique.append(t)
 
        return unique
 
    except Exception as e:
        print(f"Teacher search error: {e}", flush=True)
        return []
 
 
# ==============================
# DYNAMIC ROUTINE SEARCH
# ==============================
def search_routines(q_raw, ctx):
    try:
        query = supabase.table("routines").select(
            "department,shift,semester,group_name,day,period,start_time,end_time,subject,teacher_short,room"
        )
 
        if ctx["department"]:
            query = query.ilike("department", f"%{ctx['department']}%")
        if ctx["shift"]:
            query = query.eq("shift", ctx["shift"])
        if ctx["semester"]:
            query = query.ilike("semester", f"%{ctx['semester']}%")
        if ctx["day"]:
            query = query.ilike("day", f"%{ctx['day']}%")
 
        # Search by subject or teacher short name if present
        if ctx["short_names"]:
            for sn in ctx["short_names"]:
                r = query.ilike("teacher_short", f"%{sn}%").execute()
                if r.data:
                    return r.data
 
        result = query.limit(30).execute()
        return result.data or []
 
    except Exception as e:
        print(f"Routine search error: {e}", flush=True)
        return []
 
 
# ==============================
# DYNAMIC LOCATION SEARCH
# ==============================
def search_locations(q_raw, ctx):
    try:
        # Extract location keywords from raw question
        # Try ilike search on name and description
        keywords = []
 
        location_terms = [
            "ওয়াশরুম", "টয়লেট", "washroom", "toilet",
            "ক্যান্টিন", "canteen", "লাইব্রেরি", "library",
            "lab", "ল্যাব", "laboratory", "হলরুম", "hall",
            "অফিস", "office", "কক্ষ", "room", "gate", "গেট",
            "mosque", "মসজিদ", "field", "মাঠ", "parking", "পার্কিং",
            "wiring", "hardware", "electrical", "computer",
            "civil", "mechanical", "workshop", "ওয়ার্কশপ",
            "center", "centre", "কেন্দ্র",
        ]
 
        for term in location_terms:
            if term in q_raw.lower():
                keywords.append(term)
 
        if keywords:
            # Search by most specific keyword
            for kw in keywords:
                query = supabase.table("locations").select(
                    "name,description,floor,building"
                ).or_(
                    f"name.ilike.%{kw}%,"
                    f"description.ilike.%{kw}%"
                )
                if ctx["floor"]:
                    query = query.ilike("floor", f"%{ctx['floor']}%")
 
                result = query.limit(15).execute()
                if result.data:
                    return result.data
 
        # Floor-based search
        if ctx["floor"]:
            result = supabase.table("locations").select(
                "name,description,floor,building"
            ).ilike("floor", f"%{ctx['floor']}%").execute()
            if result.data:
                return result.data
 
        # Fallback: return all locations
        result = supabase.table("locations").select(
            "name,description,floor,building"
        ).limit(97).execute()
        return result.data or []
 
    except Exception as e:
        print(f"Location search error: {e}", flush=True)
        return []
 
 
# ==============================
# DYNAMIC QA SEARCH
# ==============================
def search_qa(q_raw):
    try:
        # First try to find matching question
        result = supabase.table("qa").select(
            "question,answer"
        ).ilike("question", f"%{q_raw[:50]}%").limit(10).execute()
 
        if result.data:
            return result.data
 
        # Fallback: get all QA
        result = supabase.table("qa").select(
            "question,answer"
        ).limit(100).execute()
        return result.data or []
 
    except Exception as e:
        print(f"QA search error: {e}", flush=True)
        return []
 
 
# ==============================
# SMART DATA FETCHING (RAG)
# ==============================
def get_relevant_data(user_question):
    q = user_question.lower()
    ctx = extract_context(q)
    data = ""
 
    try:
        # --- ROUTINE ---
        if any(w in q for w in [
            "রুটিন", "ক্লাস", "routine", "class", "সময়", "পিরিয়ড",
            "কখন", "schedule", "তারিখ", "বার", "দিন", "বিষয়", "subject"
        ]):
            rows = search_routines(user_question, ctx)
            if rows:
                data += "=== ক্লাস রুটিন ===\n"
                for r in rows:
                    data += (
                        f"{r['department']}|{r['shift']}|{r['semester']}|"
                        f"{r['group_name']}|{r['day']}|{r['period']}|"
                        f"{r['start_time']}-{r['end_time']}|{r['subject']}|"
                        f"{r['teacher_short']}|{r['room']}\n"
                    )
 
        # --- TEACHER ---
        if any(w in q for w in [
            "শিক্ষক", "স্যার", "ম্যাম", "teacher", "instructor",
            "প্রভাষক", "অধ্যাপক", "শিক্ষিকা", "পড়ান", "পড়াচ্ছেন",
            "কে পড়া", "স্যারের", "ম্যামের", "কোন স্যার", "কোন শিক্ষক",
            "chief", "head", "hod", "বিভাগীয়", "প্রধান", "ইন্সট্রাক্টর",
            "who is", "কে আছেন", "কে দায়িত্বে", "দায়িত্বপ্রাপ্ত"
        ]) or ctx["short_names"]:
            rows = search_teachers(user_question, ctx)
            if rows:
                data += "=== শিক্ষক তালিকা ===\n"
                for t in rows:
                    data += (
                        f"{t['name']} | {t['designation']} | "
                        f"{t['subject']} | {t['short_name']} | "
                        f"বিভাগ: {t.get('department','')} | "
                        f"শিফট: {t.get('shift','')}\n"
                    )
 
        # --- NOTICE ---
        if any(w in q for w in [
            "নোটিশ", "বিজ্ঞপ্তি", "notice", "circular", "ঘোষণা", "সর্বশেষ", "নতুন"
        ]):
            rows = supabase.table("notices").select(
                "title,content,date"
            ).order("created_at", desc=True).limit(5).execute()
            if rows.data:
                data += "=== সাম্প্রতিক নোটিশ ===\n"
                for n in rows.data:
                    content = (n.get("content") or "")[:200]
                    data += f"• {n['title']} ({n['date']}): {content}\n"
 
        # --- LOCATION ---
        if any(w in q for w in [
            "কোথায়", "রুম", "ওয়াশরুম", "টয়লেট", "ক্যান্টিন",
            "লাইব্রেরি", "where", "room", "কক্ষ", "তলা", "floor",
            "lab", "laboratory", "center", "centre", "wiring",
            "hardware", "ল্যাব", "কেন্দ্র", "gate", "গেট",
            "mosque", "মসজিদ", "field", "মাঠ", "অফিস", "office"
        ]):
            rows = search_locations(user_question, ctx)
            if rows:
                data += "=== লোকেশন ===\n"
                for l in rows:
                    data += (
                        f"{l['name']}: {l['description']} | "
                        f"তলা: {l['floor']} | বিল্ডিং: {l['building']}\n"
                    )
 
        # --- Q&A (dynamic match) ---
        qa_rows = search_qa(user_question)
        if qa_rows:
            data += "\n=== প্রশ্নোত্তর ===\n"
            for item in qa_rows:
                data += f"প্রশ্ন: {item['question']}\nউত্তর: {item['answer']}\n\n"
 
        if not data:
            data = "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫।"
 
        print(f"RAG data: {len(data)} chars", flush=True)
        return data
 
    except Exception as e:
        print(f"Data fetch error: {e}", flush=True)
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
# ERROR LOGGER → SUPABASE
# ==============================
def log_error(error_type, user_message, error_detail):
    try:
        supabase.table("error_logs").insert({
            "error_type": error_type,
            "user_message": user_message,
            "error_detail": str(error_detail)
        }).execute()
    except Exception as e:
        print(f"Error log save failed: {e}", flush=True)
 
 
# ==============================
# GEMINI RESPONSE
# ==============================
def get_response(system_prompt, history, user_input):
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
            generation_config={"max_output_tokens": 1000}
        )
 
        gemini_history = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
 
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_input)
        return response.text, None
 
    except Exception as e:
        print(f"CRITICAL GEMINI ERROR: {e}", flush=True)
        return None, str(e)
 
 
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
        print(f"Conversation save error: {e}", flush=True)
 
 
# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")
 
 
@app.route("/ask", methods=["POST"])
def ask():
    user_input = ""
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "missing message"}), 400
 
        user_input = data["message"]
        history = data.get("history", [])[-4:]
 
        print(f"User input: {user_input[:80]}", flush=True)
 
        system_prompt = build_system_prompt(user_input)
        reply, error = get_response(system_prompt, history, user_input)
 
        if error:
            log_error("GEMINI_ERROR", user_input, error)
            return jsonify({"reply": "দুঃখিত, এই মুহূর্তে উত্তর দিতে পারছি না। একটু পরে চেষ্টা করুন।"})
 
        print(f"Reply: {reply[:80]}", flush=True)
        save_conversation(user_input, reply)
        return jsonify({"reply": reply})
 
    except Exception as e:
        print(f"CRITICAL /ask error: {e}", flush=True)
        import traceback; traceback.print_exc()
        log_error("SERVER_ERROR", user_input, e)
        return jsonify({"reply": "দুঃখিত, সার্ভারে সমস্যা হয়েছে। একটু পরে চেষ্টা করুন।"})
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
 