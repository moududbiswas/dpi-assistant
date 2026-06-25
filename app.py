import os
import re
from gtts import gTTS
import io
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from supabase import create_client

app = Flask(__name__)


# CONFIG

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

MAX_CONTEXT_CHARS = 20000



def _find_earliest_match(mapping, text):
    best_value = None
    best_idx = None
    for key, value in mapping.items():
        idx = text.find(key)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_value = value
    return best_value



def extract_context(q_original, history=None):
    """Extract department, shift, semester, day, floor from question + history."""
    q = q_original          # keep original for short_names regex
    ql = q_original.lower() # lowercased for keyword matching

    history = history or []
    history_combined = " ".join(
        m.get("content", "") for m in history[-6:]
    ).lower()

    dept_map = {
        "সিভিল": "সিভিল", "civil": "সিভিল",
        "ইলেকট্রিক্যাল": "ইলেকট্রিক্যাল", "ইলেক্ট্রিক্যাল": "ইলেকট্রিক্যাল", "electrical": "ইলেকট্রিক্যাল",
        "কম্পিউটার": "কম্পিউটার", "computer": "কম্পিউটার",
        "মেকানিক্যাল": "মেকানিক্যাল", "mechanical": "মেকানিক্যাল",
        "আর্কিটেকচার": "আর্কিটেকচার", "architecture": "আর্কিটেকচার",
        "ইলেকট্রনিক্স": "ইলেকট্রনিক্স", "electronics": "ইলেকট্রনিক্স",
        "কেমিক্যাল": "কেমিক্যাল", "chemical": "কেমিক্যাল",
        "টেক্সটাইল": "টেক্সটাইল", "textile": "টেক্সটাইল",
    }


    shift_map = {
        "প্রথম শিফট": "১ম", "১ম শিফট": "১ম",
        "প্রথম": "১ম", "মর্নিং": "১ম", "morning": "১ম",
        "দ্বিতীয় শিফট": "২য়", "২য় শিফট": "২য়",
        "দ্বিতীয়": "২য়", "ডে": "২য়", "day shift": "২য়",
    }

   
    semester_map = {
        "১ম সেমিস্টার": "১ম", "প্রথম সেমিস্টার": "১ম",
        "২য় সেমিস্টার": "২য়", "দ্বিতীয় সেমিস্টার": "২য়",
        "৩য় সেমিস্টার": "৩য়", "তৃতীয় সেমিস্টার": "৩য়",
        "৪র্থ সেমিস্টার": "৪র্থ", "চতুর্থ সেমিস্টার": "৪র্থ",
        "৫ম সেমিস্টার": "৫ম", "পঞ্চম সেমিস্টার": "৫ম",
        "৬ষ্ঠ সেমিস্টার": "৬ষ্ঠ", "ষষ্ঠ সেমিস্টার": "৬ষ্ঠ",
        "৭ম সেমিস্টার": "৭ম", "সপ্তম সেমিস্টার": "৭ম",
        "৮ম সেমিস্টার": "৮ম", "অষ্টম সেমিস্টার": "৮ম",
        "১ম": "১ম", "২য়": "২য়", "৩য়": "৩য়", "৪র্থ": "৪র্থ",
        "৫ম": "৫ম", "৬ষ্ঠ": "৬ষ্ঠ", "৭ম": "৭ম", "৮ম": "৮ম",
        "1st": "১ম", "2nd": "২য়", "3rd": "৩য়", "4th": "৪র্থ",
        "5th": "৫ম", "6th": "৬ষ্ঠ", "7th": "৭ম", "8th": "৮ম",
    }

    day_map = {
        "রবিবার": "রবিবার", "sunday": "রবিবার",
        "সোমবার": "সোমবার", "monday": "সোমবার",
        "মঙ্গলবার": "মঙ্গলবার", "tuesday": "মঙ্গলবার",
        "বুধবার": "বুধবার", "wednesday": "বুধবার",
        "বৃহস্পতিবার": "বৃহস্পতিবার", "thursday": "বৃহস্পতিবার",
        "শুক্রবার": "শুক্রবার", "friday": "শুক্রবার",
        "শনিবার": "শনিবার", "saturday": "শনিবার",
    }

    floor_map = {
        "নিচ তলা": "ground", "গ্রাউন্ড": "ground", "ground": "ground",
        "১ম তলা": "1st", "দ্বিতীয় তলা": "2nd", "তৃতীয় তলা": "3rd",
        "চতুর্থ তলা": "4th", "পঞ্চম তলা": "5th",
        "1st floor": "1st", "2nd floor": "2nd", "3rd floor": "3rd",
    }

    group_match = re.findall(r'\b([A-D])\b', q)

    ctx = {
        "department": None, "shift": None, "semester": None,
        "day": None, "floor": None,
        "group": group_match[0] if group_match else None,
        "short_names": re.findall(r'\b[A-Z]{2,5}\b', q),
    }

    all_text = ql + " " + history_combined

    ctx["department"] = _find_earliest_match(dept_map, all_text)
    ctx["day"]        = _find_earliest_match(day_map, all_text)
    ctx["floor"]      = _find_earliest_match(floor_map, all_text)


    shift_pattern = re.search(
        r'(১ম|প্রথম|first|1st|morning|মর্নিং)\s*(শিফট|shift)?|'
        r'(২য়|দ্বিতীয়|second|2nd|day|ডে)\s*(শিফট|shift)',
        all_text
    )
    if shift_pattern:
        matched = shift_pattern.group(0).lower()
        if any(k in matched for k in ["১ম", "প্রথম", "first", "1st", "morning", "মর্নিং"]):
            ctx["shift"] = "১ম"
        else:
            ctx["shift"] = "২য়"
    else:
        ctx["shift"] = _find_earliest_match(shift_map, all_text)


    semester_from_map = _find_earliest_match(semester_map, all_text)

    # If the semester_map picked the same ordinal as shift (e.g. both
    # matched "1st"), we need to find the NEXT ordinal in the text.
    if semester_from_map and ctx["shift"]:
        shift_en_map = {"১ম": "1st", "২য়": "2nd"}
        shift_en = shift_en_map.get(ctx["shift"], "")

        shift_ordinals = {
            "১ম": ["১ম", "প্রথম", "1st", "first", "morning", "মর্নিং"],
            "২য়": ["২য়", "দ্বিতীয়", "2nd", "second", "day", "ডে"],
        }
        shift_keywords = shift_ordinals.get(ctx["shift"], [])

        earliest_sem_key = None
        earliest_sem_idx = None
        for key, val in semester_map.items():
            idx = all_text.find(key)
            if idx != -1 and (earliest_sem_idx is None or idx < earliest_sem_idx):
                earliest_sem_idx = idx
                earliest_sem_key = key

        if earliest_sem_key and earliest_sem_key in shift_keywords:
           
            sem_keyword_idx = all_text.find("semester")
            if sem_keyword_idx == -1:
                sem_keyword_idx = all_text.find("সেমিস্টার")

            best_key = None
            best_dist = None
            for key, val in semester_map.items():
                if key in shift_keywords:
                    continue  
                idx = all_text.find(key)
                if idx == -1:
                    continue
                if sem_keyword_idx != -1:
                    dist = abs(idx - sem_keyword_idx)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_key = key
                        semester_from_map = val
                else:
                    
                    if best_key is None:
                        best_key = key
                        semester_from_map = val

    ctx["semester"] = semester_from_map

    return ctx



# TEACHER SEARCH

def search_teachers(q_raw, ctx):
    try:
        results = []

        if ctx["short_names"]:
            for sn in ctx["short_names"]:
                query = supabase.table("teachers").select(
                    "name,subject,short_name,designation,department,shift,contact_number"
                ).or_(
                    f"short_name.ilike.%{sn}%,name.ilike.%{sn}%"
                )
                if ctx["department"]:
                    query = query.ilike("department", f"%{ctx['department']}%")
                if ctx["shift"]:
                    r = query.eq("shift", ctx["shift"]).execute()
                    if r.data:
                        results.extend(r.data)
                        continue
                r = query.execute()
                results.extend(r.data or [])

        if not results:
            query = supabase.table("teachers").select(
                "name,subject,short_name,designation,department,shift,contact_number"
            )
            if ctx["department"]:
                query = query.ilike("department", f"%{ctx['department']}%")
            if ctx["shift"]:
                r = query.eq("shift", ctx["shift"]).execute()
                results = r.data if r.data else query.execute().data or []
            else:
                results = query.limit(60).execute().data or []

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



# ROUTINE SEARCH

def search_routines(q_raw, ctx):
    try:
        query = supabase.table("routines").select(
            "department,shift,semester,group_name,day,period,start_time,end_time,subject,teacher_short,room"
        )

        has_any_filter = any([
            ctx["department"], ctx["shift"], ctx["semester"],
            ctx["day"], ctx["short_names"], ctx.get("group")
        ])

        if not has_any_filter:
            print("Routine: no filters detected, returning [] to let Gemini ask", flush=True)
            return []

        if ctx["department"]:
            query = query.ilike("department", f"%{ctx['department']}%")
        if ctx["shift"]:
            query = query.eq("shift", ctx["shift"])
        if ctx["semester"]:
            query = query.ilike("semester", f"%{ctx['semester']}%")
        if ctx["day"]:
            query = query.ilike("day", f"%{ctx['day']}%")

        if ctx.get("group"):
            query = query.ilike("group_name", f"%{ctx['group']}%")

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



# LOCATION SEARCH

def search_locations(q_raw, ctx):
    try:
        location_terms = [
            "ওয়াশরুম", "টয়লেট", "washroom", "toilet",
            "ক্যান্টিন", "canteen", "লাইব্রেরি", "library",
            "lab", "ল্যাব", "laboratory", "হলরুম", "hall",
            "অফিস", "office", "কক্ষ", "room", "gate", "গেট",
            "mosque", "মসজিদ", "field", "মাঠ", "parking", "পার্কিং",
            "wiring", "hardware", "electrical", "computer",
            "civil", "mechanical",  "workshop", "ওয়ার্কশপ",
            "center", "centre", "কেন্দ্র", "chemistry","physics", "wood shop",
        ]

        room_numbers = re.findall(r'\b\d{2,4}\b', q_raw)
        if room_numbers:
            for rn in room_numbers:
                result = supabase.table("locations").select(
                    "name,description,floor,building"
                ).or_(
                    f"name.ilike.%{rn}%,description.ilike.%{rn}%"
                ).limit(10).execute()
                if result.data:
                    print(f"Location: matched by room number '{rn}': {len(result.data)} rows", flush=True)
                    return result.data

        keywords = [term for term in location_terms if term in q_raw.lower()]

        if keywords:
            for kw in keywords:
                query = supabase.table("locations").select(
                    "name,description,floor,building"
                ).or_(
                    f"name.ilike.%{kw}%,description.ilike.%{kw}%"
                )
                if ctx["floor"]:
                    query = query.ilike("floor", f"%{ctx['floor']}%")
                result = query.limit(107).execute()
                if result.data:
                    return result.data

        if ctx["floor"]:
            result = supabase.table("locations").select(
                "name,description,floor,building"
            ).ilike("floor", f"%{ctx['floor']}%").execute()
            if result.data:
                return result.data

        print("Location: no keyword/floor match, returning [] to avoid context flood", flush=True)
        return []

    except Exception as e:
        print(f"Location search error: {e}", flush=True)
        return []



# QA SEARCH

def search_qa(q_raw):
    try:
        result = supabase.table("qa").select("question,answer") \
            .ilike("question", f"%{q_raw[:60]}%") \
            .not_.is_("answer", "null") \
            .limit(10).execute()

        if result.data:
            print(f"QA matched by chunk: {len(result.data)} rows", flush=True)
            return result.data

        bangla_stopwords = {"কি", "কে", "কোন", "কখন", "কত", "কার", "এর", "এই",
                            "আছে", "আছেন", "হয়", "কোথায়", "দেন", "দাও", "বলো"}
        words = [
            w.strip("?।,!\"'") for w in re.split(r'\s+', q_raw)
            if len(w.strip("?।,!\"'")) >= 3 and w.strip("?।,!\"'") not in bangla_stopwords
        ]

        seen = set()
        matches = []

        for word in words[:8]:
            r = supabase.table("qa").select("question,answer") \
                .ilike("question", f"%{word}%") \
                .not_.is_("answer", "null") \
                .limit(5).execute()

            for row in (r.data or []):
                key = row["question"]
                if key not in seen:
                    seen.add(key)
                    matches.append(row)

        if matches:
            print(f"QA matched by keywords: {len(matches)} rows", flush=True)
            return matches[:20]

        result = supabase.table("qa").select("question,answer") \
            .not_.is_("answer", "null") \
            .limit(20).execute()

        print(f"QA fallback: {len(result.data or [])} rows", flush=True)
        return result.data or []

    except Exception as e:
        print(f"QA search error: {e}", flush=True)
        return []



# SMART DATA FETCHING (RAG)

def get_relevant_data(user_question, history=None):
    ctx = extract_context(user_question, history)
    q = user_question.lower()
    data = ""

 
    print(f"CTX → dept={ctx['department']} shift={ctx['shift']} "
          f"sem={ctx['semester']} group={ctx['group']} day={ctx['day']}", flush=True)

    history = history or []
    history_text = " ".join(
        m.get("content", "") for m in history[-6:]
    ).lower()
    q_with_history = q + " " + history_text

    try:
        # --- ROUTINE ---
        if any(w in q_with_history for w in [
            "রুটিন", "ক্লাস", "routine", "class", "সময়", "পিরিয়ড",
            "কখন", "schedule", "তারিখ", "বার", "দিন", "বিষয়", "subject",
            "আজকে", "আজ", "কোন রুম", "পড়া", "ক্লাসরুম", "classroo"
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
        if any(w in q_with_history for w in [
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
                        f"বিভাগ: {t.get('department', '')} | "
                        f"শিফট: {t.get('shift', '')} | "
                        f"যোগাযোগ: {t.get('contact_number', '')}\n"
                    )

        # --- NOTICE ---
        if any(w in q_with_history for w in [
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
        if any(w in q_with_history for w in [
            "কোথায়", "রুম", "ওয়াশরুম", "টয়লেট", "ক্যান্টিন",
            "লাইব্রেরি", "where", "room", "কক্ষ", "তলা", "floor",
            "lab", "laboratory", "center", "centre", "wiring",
            "hardware", "ল্যাব", "কেন্দ্র", "gate", "গেট",
            "mosque", "মসজিদ", "field", "মাঠ", "অফিস", "office", "physics", "chemistry"
        ]):
            rows = search_locations(user_question, ctx)
            if rows:
                data += "=== লোকেশন ===\n"
                for l in rows:
                    data += (
                        f"{l['name']}: {l['description']} | "
                        f"তলা: {l['floor']} | বিল্ডিং: {l['building']}\n"
                    )

        # --- Q&A ---
        qa_rows = search_qa(user_question)
        if qa_rows:
            data += "\n=== প্রশ্নোত্তর ===\n"
            for item in qa_rows:
                answer = item.get("answer") or ""
                if answer.strip():
                    data += f"প্রশ্ন: {item['question']}\nউত্তর: {answer}\n\n"

        # --- Context size guard ---
        if len(data) > MAX_CONTEXT_CHARS:
            data = data[:MAX_CONTEXT_CHARS] + "\n[...তথ্য সংক্ষিপ্ত করা হয়েছে]"

        if not data.strip():
            data = "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫।"

        print(f"RAG data: {len(data)} chars", flush=True)
        return data

    except Exception as e:
        print(f"Data fetch error: {e}", flush=True)
        return "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫।"


# ==============================
# SYSTEM PROMPT
# ==============================
def build_system_prompt(user_question="", history=None):
    relevant_data = get_relevant_data(user_question, history or [])
    return f"""তুমি ঢাকা পলিটেকনিক ইনস্টিটিউটের AI সহকারী, নাম DPI Assistant। সবসময় বাংলায় উত্তর দাও।
কাউকে স্বাগত জানানোর সময় বা প্রথম বার্তায় সালাম দাও: "আসসালামু আলাইকুম" — কখনো "নমস্কার" বলবে না।

নিয়ম:
- শুধুমাত্র নিচের তথ্য থেকে উত্তর দাও
- তথ্য না থাকলে বলো: "দুঃখিত, এই তথ্যটি আমার কাছে নেই। টিমকে জানান।"
- রাজনীতি, ধর্মীয় বিতর্ক, অশ্লীল, প্রেম বা কলেজ-বহির্ভূত প্রশ্নে বলো: "আমি শুধু DPI সম্পর্কিত প্রশ্নের উত্তর দিতে পারি।"
- রুটিন জিজ্ঞেস করলে ধাপে ধাপে জিজ্ঞেস করো: বিভাগ → শিফট → সেমিস্টার ও গ্রুপ
- শিক্ষক সম্পর্কে জিজ্ঞেস করলে আগে জিজ্ঞেস করো: কোন বিভাগ? কোন শিফট?
- ব্যতিক্রম (Exception): তোমাকে কে তৈরি করেছে বা এই অ্যাপের ডেভেলপার কে, এই প্রশ্নের উত্তর দেওয়ার অনুমতি তোমাকে দেওয়া হলো। যদি কেউ এই ধরণের প্রশ্ন করে, ডাটাবেজ (qa) থেকে তথ্য নিয়ে উত্তর দাও
=== তথ্য ===
{relevant_data}"""



# ERROR LOGGER → SUPABASE

def log_error(error_type, user_message, error_detail):
    try:
        supabase.table("error_logs").insert({
            "error_type": error_type,
            "user_message": user_message,
            "error_detail": str(error_detail)
        }).execute()
    except Exception as e:
        print(f"Error log save failed: {e}", flush=True)



# GEMINI RESPONSE

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



# HELPERS

def save_conversation(user_message, bot_reply):
    try:
        supabase.table("conversations").insert({
            "user_message": user_message,
            "bot_reply": bot_reply
        }).execute()
    except Exception as e:
        print(f"Conversation save error: {e}", flush=True)



# ROUTES

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

        system_prompt = build_system_prompt(user_input, history)
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

 
@app.route("/tts", methods=["POST"])
def tts():
    """Convert text to speech using gTTS and return MP3 audio bytes."""
    try:
        data = request.json
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "no text"}), 400
 
        # Truncate long replies so TTS stays snappy
        if len(text) > 800:
            text = text[:800] + "..."
 
        tts_obj = gTTS(text=text, lang="bn", slow=False)
        mp3_fp = io.BytesIO()
        tts_obj.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
 
        from flask import send_file
        return send_file(
            mp3_fp,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="reply.mp3"
        )
 
    except Exception as e:
        print(f"TTS error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
