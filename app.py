import os
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from supabase import create_client

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================
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

MAX_CONTEXT_CHARS = 10000


# ==============================
# FIX 5: Position-based keyword matcher
# Old behavior: looped over a dict and broke on the first key found in
# DICT DEFINITION ORDER, not the order the keyword actually appears in
# the user's sentence. This caused wrong department/shift/semester/day/
# floor picks whenever a question mentioned more than one candidate
# (e.g. "মেকানিক্যাল না, ইলেকট্রিক্যাল বিভাগের রুটিন চাই").
# New behavior: scan ALL keys, find the one whose match starts at the
# LOWEST index in the text, and use that. Since `all_text` is built as
# (current message + history), the current message's keywords are
# checked first, naturally prioritizing the latest user intent over
# older history mentions.
# ==============================
def _find_earliest_match(mapping, text):
    best_value = None
    best_idx = None
    for key, value in mapping.items():
        idx = text.find(key)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_value = value
    return best_value


# ==============================
# KEYWORD EXTRACTOR
# FIX 1: Now receives original-case string so [A-Z] short_names regex works
# FIX 5: department/shift/semester/day/floor now picked by earliest
#        position in text instead of dict iteration order (see above)
# ==============================
def extract_context(q_original, history=None):
    """Extract department, shift, semester, day, floor from question + history.
    Must receive the ORIGINAL (non-lowercased) string so short_names
    regex r'\b[A-Z]{2,5}\b' can actually match uppercase abbreviations.
    history is scanned so multi-turn replies ("Electrical", "2nd") are caught.
    """
    q = q_original  # keep original for short_names regex
    ql = q_original.lower()  # use lowercased only for keyword matching

    # Combine history messages so context from previous turns is visible
    history = history or []
    history_combined = " ".join(
        m.get("content", "") for m in history[-6:]
    ).lower()

    # Values match exactly what is stored in Supabase (Bengali text)
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
        "প্রথম": "১ম", "1st": "১ম", "first": "১ম", "মর্নিং": "১ম", "morning": "১ম",
        "দ্বিতীয়": "২য়", "2nd": "২য়", "second": "২য়", "ডে": "২য়", "day": "২য়",
    }

    semester_map = {
        "১ম": "১ম", "প্রথম সেমিস্টার": "১ম",
        "২য়": "২য়", "দ্বিতীয় সেমিস্টার": "২য়",
        "৩য়": "৩য়", "তৃতীয় সেমিস্টার": "৩য়",
        "৪র্থ": "৪র্থ", "চতুর্থ সেমিস্টার": "৪র্থ",
        "৫ম": "৫ম", "পঞ্চম সেমিস্টার": "৫ম",
        "৬ষ্ঠ": "৬ষ্ঠ", "ষষ্ঠ সেমিস্টার": "৬ষ্ঠ",
        "৭ম": "৭ম", "সপ্তম সেমিস্টার": "৭ম",
        "৮ম": "৮ম", "অষ্টম সেমিস্টার": "৮ম",
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

    # Extract single-letter group (A/B/C/D) from original string
    group_match = re.findall(r'\b([A-D])\b', q)

    ctx = {
        "department": None, "shift": None, "semester": None,
        "day": None, "floor": None,
        "group": group_match[0] if group_match else None,
        # FIX 1: run regex on original string, not lowercased
        "short_names": re.findall(r'\b[A-Z]{2,5}\b', q),
    }

    # Scan current message first, fall back to history if not found.
    # This handles multi-turn: "routine" → "Electrical" → "2nd" → "5th,c"
    # Each follow-up reply alone has no dept/shift, but history does.
    all_text = ql + " " + history_combined

    # FIX 5: replaced dict-order "first match wins" loops with
    # position-based earliest-match lookup (see _find_earliest_match above)
    ctx["department"] = _find_earliest_match(dept_map, all_text)
    ctx["shift"] = _find_earliest_match(shift_map, all_text)
    ctx["semester"] = _find_earliest_match(semester_map, all_text)
    ctx["day"] = _find_earliest_match(day_map, all_text)
    ctx["floor"] = _find_earliest_match(floor_map, all_text)

    return ctx


# ==============================
# DYNAMIC TEACHER SEARCH
# ==============================
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
# FIX 2: Guard against unfiltered 100-row fallback flooding context
# ==============================
def search_routines(q_raw, ctx):
    try:
        query = supabase.table("routines").select(
            "department,shift,semester,group_name,day,period,start_time,end_time,subject,teacher_short,room"
        )

        has_any_filter = any([
            ctx["department"], ctx["shift"], ctx["semester"],
            ctx["day"], ctx["short_names"], ctx.get("group")
        ])

        # FIX 2: If no filters at all, return empty so Gemini asks the user
        # to clarify (dept → shift → semester) as instructed in system prompt
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

        # FIX 2: Cap at 30 rows max (was 100) to protect context budget
        result = query.limit(30).execute()
        return result.data or []

    except Exception as e:
        print(f"Routine search error: {e}", flush=True)
        return []


# ==============================
# DYNAMIC LOCATION SEARCH
# FIX 3: Remove blind 50-row fallback that floods Gemini's context
# ==============================
def search_locations(q_raw, ctx):
    try:
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

        # FIX: Extract room numbers directly from query (e.g. "where is 113 room")
        # Supabase has descriptions like "Room Number 113" — search by the number itself
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

        # Check against original q_raw (Bengali .lower() is a no-op anyway)
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
                result = query.limit(20).execute()
                if result.data:
                    return result.data

        if ctx["floor"]:
            result = supabase.table("locations").select(
                "name,description,floor,building"
            ).ilike("floor", f"%{ctx['floor']}%").execute()
            if result.data:
                return result.data

        # FIX 3: No keyword and no floor filter → return empty instead of
        # dumping 50 unrelated rows that confuse Gemini
        print("Location: no keyword/floor match, returning [] to avoid context flood", flush=True)
        return []

    except Exception as e:
        print(f"Location search error: {e}", flush=True)
        return []


# ==============================
# DYNAMIC QA SEARCH
# ==============================
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
            .limit(100).execute()

        print(f"QA fallback: {len(result.data or [])} rows", flush=True)
        return result.data or []

    except Exception as e:
        print(f"QA search error: {e}", flush=True)
        return []


# ==============================
# SMART DATA FETCHING (RAG)
# FIX 1 applied here: extract_context now receives user_question (original case)
# FIX 4: Expanded routine trigger keywords
# ==============================
def get_relevant_data(user_question, history=None):
    # FIX 1: pass original string — extract_context needs it for [A-Z] regex
    ctx = extract_context(user_question, history)
    q = user_question.lower()  # lowercased only for trigger keyword matching
    data = ""

    # Build a combined text from recent history to detect ongoing intent.
    # When user replies "Electrical" or "2nd" as follow-up, the original
    # intent (routine/location) only exists in previous messages.
    history = history or []
    history_text = " ".join(
        m.get("content", "") for m in history[-6:]
    ).lower()
    # Merge current message + history for trigger detection only
    q_with_history = q + " " + history_text

    try:
        # --- ROUTINE ---
        # FIX 4: check current message AND history for routine intent
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)