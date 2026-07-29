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

# NOTE: context-length truncation removed per request ("erase all token limits").
# If you ever hit real API context-window errors, reintroduce a guard here.



def _find_earliest_match(mapping, text):
    best_value = None
    best_idx = None
    for key, value in mapping.items():
        idx = text.find(key)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_value = value
    return best_value



# Ordinary 2-5 letter English words that must NEVER be treated as a teacher
# short-name/initials code (e.g. "who is farzana akter maam" must not turn
# into short_names ["WHO", "IS", "MAAM"]).
_SHORT_NAME_STOPWORDS = {
    "WHO", "IS", "ARE", "WAS", "WERE", "AM", "BE", "BEEN", "BEING",
    "THE", "A", "AN", "OF", "TO", "IN", "ON", "AT", "BY", "OR", "AND",
    "IT", "AS", "DO", "DOES", "DID", "IF", "MY", "ME", "HE", "SHE",
    "HIS", "HER", "HIM", "THIS", "THAT", "WHAT", "WHEN", "WHERE",
    "WHY", "HOW", "CAN", "WILL", "WOULD", "COULD", "SHOULD", "HAS",
    "HAVE", "HAD", "US", "OUR", "YOUR", "THEIR", "SIR", "MAAM", "MAM",
    "MADAM", "NAME", "ABOUT", "TELL", "PLEASE", "WITH", "FOR", "FROM",
    "SOME", "ANY", "ALL", "NOT", "NO", "YES", "OK", "OKAY",
    "CONTACT", "NUMBER", "PHONE", "EMAIL", "ADDRESS", "INFO",
    "INFORMATION", "DETAILS", "GIVE", "SHOW", "FIND", "SEARCH", "LIST",
    "TEACHER", "TEACHERS", "CLASS", "ROOM", "SEMESTER", "SHIFT", "GROUP",
    "SUBJECT", "DESIGNATION", "DEPARTMENT", "AND", "GET", "TELLS",
    "INSTRUCTOR", "SIRS", "TEACHES", "TEACHING", "PROFESSOR", "LECTURER",
}


# Common Bangla question/role words to exclude when extracting possible
# Bangla name tokens from a query (so we don't try to ilike-search the
# name column for words like "কে" or "শিক্ষক" themselves).
_BANGLA_NAME_STOPWORDS = {
    "কি", "কে", "কোন", "কখন", "কত", "কার", "এর", "এই", "আছে", "আছেন",
    "হয়", "কোথায়", "দেন", "দাও", "বলো", "স্যার", "ম্যাম", "শিক্ষক",
    "শিক্ষিকা", "নাম", "সম্পর্কে", "প্রভাষক", "অধ্যাপক", "ইনস্ট্রাক্টর",
}


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
        # Case-insensitive short-name matching: previously [A-Z]{2,5} only
        # matched all-caps codes (e.g. "MAH"); lowercase/mixed-case codes
        # like "mah" were silently ignored. Now we match either case and
        # normalize to uppercase so downstream ilike() lookups still work.
        # Common English function words are filtered out below (_SHORT_NAME_STOPWORDS)
        # so "who is ... maam" doesn't get parsed as teacher initials "WHO"/"IS"/"MAAM".
        "short_names": [
            s.upper() for s in re.findall(r'\b[A-Za-z]{2,5}\b', q)
            if s.upper() not in _SHORT_NAME_STOPWORDS
        ],
        # Full-length Latin name words (e.g. "farzana", "akter" — not capped
        # at 5 chars like short_names above) used to match against the
        # name_en (romanized) column, so full names typed in English/Banglish
        # can be matched precisely instead of relying on the full-directory
        # fallback + Gemini guesswork.
        "name_tokens": [
            w.upper() for w in re.findall(r'\b[A-Za-z]{3,}\b', q)
            if w.upper() not in _SHORT_NAME_STOPWORDS
        ],
        # Bangla-script full-name words (e.g. "ফারজানা", "আক্তার") so a
        # Bangla-typed full-name query actually gets searched against the
        # name column, instead of silently falling through to a generic
        # unfiltered department/shift browse when no Latin tokens exist.
        "bangla_name_tokens": [
            w for w in re.findall(r'[\u0980-\u09FF]{2,}', q)
            if w not in _BANGLA_NAME_STOPWORDS
        ],
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

_teachers_has_name_en = None  # lazily detected once per process


def _teacher_select_fields():
    """Whether the `teachers` table has a name_en (romanized) column.

    Detected once and cached. If you haven't added the column yet, this
    quietly falls back to the old field list so nothing breaks — but see
    the note in this conversation about why adding name_en is the real fix
    for the Bangla-vs-English name matching problem.
    """
    global _teachers_has_name_en
    if _teachers_has_name_en is None:
        try:
            supabase.table("teachers").select("name_en").limit(1).execute()
            _teachers_has_name_en = True
            print("teachers.name_en column detected — bilingual name matching enabled", flush=True)
        except Exception:
            _teachers_has_name_en = False
            print("teachers.name_en column NOT found — add it for reliable English full-name search", flush=True)

    if _teachers_has_name_en:
        return "name,name_en,subject,short_name,designation,department,shift,contact_number"
    return "name,subject,short_name,designation,department,shift,contact_number"


def _get_full_teacher_directory():
    """Fallback used only when structured ilike filters find no match at all.

    This should be rare now that name_en lets us match full English names
    precisely. It's kept as a last resort for typos/nicknames, but the
    system prompt is told explicitly not to fabricate details from it.

    FIX: this previously did NOT select `subject`, but the caller in
    get_relevant_data() formats every teacher row assuming a `subject` key
    exists -> KeyError: 'subject' whenever this fallback path was used
    (visible in logs as "Data fetch error: 'subject'"), which wiped out
    the entire RAG response for that turn. Now selects subject too, and
    the formatting code also uses .get() defensively as a second layer
    of protection.
    """
    try:
        fields = "name,short_name,department,shift,designation,contact_number,subject"
        if _teachers_has_name_en:
            fields = "name,name_en,short_name,department,shift,designation,contact_number,subject"
        result = supabase.table("teachers").select(fields).order("name").limit(300).execute()
        return result.data or []
    except Exception as e:
        print(f"Teacher directory fetch error: {e}", flush=True)
        return []


def _teacher_base_query(department=None):
    """Build a fresh teacher-select query builder.

    IMPORTANT: supabase-py / postgrest-py query builders mutate themselves
    in place when you call .eq()/.ilike()/etc and return `self`, not a new
    copy. Reusing one builder variable for both a "filtered" attempt and a
    "fallback" attempt means the fallback silently ends up with the same
    filters still applied (previously .eq("shift", ...) permanently
    "stuck" onto the base query). This helper always returns a brand new
    query object so filters never leak between attempts.
    """
    query = supabase.table("teachers").select(_teacher_select_fields())
    if department:
        query = query.ilike("department", f"%{department}%")
    return query


def _teacher_name_or_filter(token):
    """Build the .or_() filter string for a single search token, matching
    short_name, the Bangla name column, and — if present — the romanized
    name_en column. This is what actually lets "farzana akter" (English)
    and "ফারজানা আক্তার" (Bangla) both resolve to the same teacher, instead
    of only one script ever matching."""
    parts = [f"short_name.ilike.%{token}%", f"name.ilike.%{token}%"]
    if _teachers_has_name_en:
        parts.append(f"name_en.ilike.%{token}%")
    return ",".join(parts)


def search_teachers(q_raw, ctx):
    try:
        _teacher_select_fields()  # ensure name_en detection has run
        results = []

        # Search tokens: short codes (FA, MSI) AND full English name words
        # (farzana, akter) both get tried against name/short_name/name_en.
        search_tokens = list(dict.fromkeys(
            ctx["short_names"] + ctx.get("name_tokens", []) + ctx.get("bangla_name_tokens", [])
        ))

        if search_tokens:
            for token in search_tokens:
                # Fresh builder per attempt so filters don't leak.
                filtered_query = _teacher_base_query(ctx["department"]).or_(
                    _teacher_name_or_filter(token)
                )

                if ctx["shift"]:
                    r = filtered_query.eq("shift", ctx["shift"]).execute()
                    if r.data:
                        results.extend(r.data)
                        continue
                    # Real fallback: brand new query, no shift filter stuck on it.
                    fallback_query = _teacher_base_query(ctx["department"]).or_(
                        _teacher_name_or_filter(token)
                    )
                    r = fallback_query.execute()
                    results.extend(r.data or [])
                else:
                    r = filtered_query.execute()
                    results.extend(r.data or [])

        if not results and not search_tokens:
            # No name/short-name signal at all -> this is a genuine generic
            # department/shift browse query (e.g. "electrical department
            # 2nd shift teachers"), so an unfiltered/lightly-filtered
            # browse fetch is appropriate here.
            base_query = _teacher_base_query(ctx["department"])
            if ctx["shift"]:
                r = base_query.eq("shift", ctx["shift"]).execute()
                if r.data:
                    results = r.data
                else:
                    # Real fallback: fresh query without the shift filter.
                    fallback_query = _teacher_base_query(ctx["department"])
                    results = fallback_query.limit(60).execute().data or []
            else:
                results = base_query.limit(60).execute().data or []

        # NOTE: if search_tokens was non-empty but every attempt above found
        # nothing (e.g. a genuine typo, or name_en not populated yet for that
        # teacher), we deliberately do NOT fall back to an unfiltered browse
        # dump here -- that would silently return unrelated teachers and
        # mask the real problem. Instead we fall through to the
        # full-directory fallback below.

        seen = set()
        unique = []
        for t in results:
            if t["name"] not in seen:
                seen.add(t["name"])
                unique.append(t)

        if unique:
            return unique

        # Structured ilike matching found nothing — most likely a full
        # Bangla name was typed in Latin/Banglish. Fall back to the full
        # directory so Gemini can match it by reasoning instead of us
        # silently returning [] (see _get_full_teacher_directory docstring).
        print("Teacher search: no ilike match, falling back to full directory", flush=True)
        return _get_full_teacher_directory()

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

        result = query.limit(100).execute()
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
            "lab", "ল্যাব", "laboratory", "হলরুম", "hall", "auditorium",
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
            .limit(35).execute()

        print(f"QA fallback: {len(result.data or [])} rows", flush=True)
        return result.data or []

    except Exception as e:
        print(f"QA search error: {e}", flush=True)
        return []



# SMART DATA FETCHING (RAG)

# Keyword sets pulled out so we can check "does this query even look like a
# routine / location / notice question" BEFORE deciding whether a stray
# name_token (e.g. "auditorium", "canteen") should be allowed to force a
# teacher search. Previously ANY 3+ letter English word in the message
# became a name_token, and `ctx.get("name_tokens")` alone was enough to
# trigger search_teachers() -- so "where is auditorium" ran a teacher
# search for the word "AUDITORIUM", found no match, fell back to the full
# teacher directory (which was missing the `subject` column), threw a
# KeyError, and — because the whole function was one big try/except —
# wiped out every other section's data for that turn too. That's why only
# "pure" teacher questions ever seemed to work.

_ROUTINE_KEYWORDS = [
    "রুটিন", "ক্লাস", "routine", "class", "সময়", "পিরিয়ড",
    "কখন", "schedule", "তারিখ", "বার", "দিন", "বিষয়", "subject",
    "আজকে", "আজ", "কোন রুম", "পড়া", "ক্লাসরুম", "classroom"
]

_TEACHER_KEYWORDS = [
    "শিক্ষক", "স্যার", "ম্যাম", "teacher", "instructor",
    "প্রভাষক", "অধ্যাপক", "শিক্ষিকা", "পড়ান", "পড়াচ্ছেন",
    "কে পড়া", "স্যারের", "ম্যামের", "কোন স্যার", "কোন শিক্ষক",
    "chief", "head", "hod", "বিভাগীয়", "প্রধান", "ইন্সট্রাক্টর",
    "who is", "কে আছেন", "sir", "কে দায়িত্বে", "দায়িত্বপ্রাপ্ত"
]

_NOTICE_KEYWORDS = [
    "নোটিশ", "বিজ্ঞপ্তি", "notice", "circular", "ঘোষণা", "সর্বশেষ", "নতুন"
]

_LOCATION_KEYWORDS = [
    "কোথায়", "রুম", "ওয়াশরুম", "টয়লেট", "ক্যান্টিন",
    "লাইব্রেরি", "where", "room", "কক্ষ", "তলা", "floor",
    "lab", "laboratory", "center", "centre", "wiring",
    "hardware", "ল্যাব", "কেন্দ্র", "gate", "গেট",
    "mosque", "মসজিদ", "field", "মাঠ", "অফিস", "office", "physics", "chemistry",
    "auditorium", "hall"
]


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

    routine_hit = any(w in q_with_history for w in _ROUTINE_KEYWORDS)
    teacher_kw_hit = any(w in q_with_history for w in _TEACHER_KEYWORDS)
    notice_hit = any(w in q_with_history for w in _NOTICE_KEYWORDS)
    location_hit = any(w in q_with_history for w in _LOCATION_KEYWORDS)

    # FIX: a bare name_token/bangla_name_token no longer forces a teacher
    # search on its own if the query already clearly looks like a routine,
    # location, or notice question. short_names (2-5 letter ALL-CAPS style
    # codes) are a stronger signal and still trigger teacher search on
    # their own, since real words rarely look like "MSI" or "FA".
    name_signal = bool(ctx.get("name_tokens") or ctx.get("bangla_name_tokens"))
    teacher_hit = (
        teacher_kw_hit
        or bool(ctx["short_names"])
        or (name_signal and not (routine_hit or location_hit or notice_hit))
    )

    # --- ROUTINE ---
    if routine_hit:
        try:
            rows = search_routines(user_question, ctx)
            if rows:
                data += "=== ক্লাস রুটিন ===\n"
                for r in rows:
                    data += (
                        f"{r.get('department','')}|{r.get('shift','')}|{r.get('semester','')}|"
                        f"{r.get('group_name','')}|{r.get('day','')}|{r.get('period','')}|"
                        f"{r.get('start_time','')}-{r.get('end_time','')}|{r.get('subject','')}|"
                        f"{r.get('teacher_short','')}|{r.get('room','')}\n"
                    )
        except Exception as e:
            print(f"Data fetch error (routine section): {e}", flush=True)

    # --- TEACHER ---
    if teacher_hit:
        try:
            rows = search_teachers(user_question, ctx)
            if rows:
                data += "=== শিক্ষক তালিকা ===\n"
                for t in rows:
                    data += (
                        f"{t.get('name','')} | {t.get('designation','')} | "
                        f"{t.get('subject','')} | {t.get('short_name','')} | "
                        f"বিভাগ: {t.get('department', '')} | "
                        f"শিফট: {t.get('shift', '')} | "
                        f"যোগাযোগ: {t.get('contact_number', '')}\n"
                    )
        except Exception as e:
            print(f"Data fetch error (teacher section): {e}", flush=True)

    # --- NOTICE ---
    if notice_hit:
        try:
            rows = supabase.table("notices").select(
                "title,content,date"
            ).order("created_at", desc=True).limit(5).execute()
            if rows.data:
                data += "=== সাম্প্রতিক নোটিশ ===\n"
                for n in rows.data:
                    content = (n.get("content") or "")[:200]
                    data += f"• {n.get('title','')} ({n.get('date','')}): {content}\n"
        except Exception as e:
            print(f"Data fetch error (notice section): {e}", flush=True)

    # --- LOCATION ---
    if location_hit:
        try:
            rows = search_locations(user_question, ctx)
            if rows:
                data += "=== লোকেশন ===\n"
                for l in rows:
                    data += (
                        f"{l.get('name','')}: {l.get('description','')} | "
                        f"তলা: {l.get('floor','')} | বিল্ডিং: {l.get('building','')}\n"
                    )
        except Exception as e:
            print(f"Data fetch error (location section): {e}", flush=True)

    # --- Q&A ---
    try:
        qa_rows = search_qa(user_question)
        if qa_rows:
            data += "\n=== প্রশ্নোত্তর ===\n"
            for item in qa_rows:
                answer = item.get("answer") or ""
                if answer.strip():
                    data += f"প্রশ্ন: {item.get('question','')}\nউত্তর: {answer}\n\n"
    except Exception as e:
        print(f"Data fetch error (qa section): {e}", flush=True)

    if not data.strip():
        data = "ঢাকা পলিটেকনিক ইনস্টিটিউট, তেজগাঁও, ঢাকা। প্রতিষ্ঠাকাল: ১৯৫৫।"

    print(f"RAG data: {len(data)} chars", flush=True)
    return data


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
- রুটিন জিজ্ঞেস করলে, যদি নিচের তথ্যে ইতিমধ্যে বিভাগ/শিফট/সেমিস্টার অনুযায়ী উত্তর থাকে তাহলে সরাসরি উত্তর দাও; তথ্য না থাকলে বা অস্পষ্ট হলে ধাপে ধাপে জিজ্ঞেস করো: বিভাগ → শিফট → সেমিস্টার ও গ্রুপ
- শিক্ষক সম্পর্কে জিজ্ঞেস করলে, যদি নিচের "শিক্ষক তালিকা" তথ্যে উত্তর থাকে তাহলে সরাসরি সেখান থেকে উত্তর দাও; একাধিক ফলাফল থাকলে বা তথ্য না পেলে তখনই জিজ্ঞেস করো: কোন বিভাগ? কোন শিফট?
- শিক্ষক তালিকায় প্রশ্নে উল্লেখিত নামের সাথে হুবহু বা স্পষ্টভাবে কাছাকাছি মিল না পেলে অনুমান করে কোনো নাম, পদবি, বা যোগাযোগ নম্বর তৈরি/বানিয়ে বলবে না — এমতাবস্থায় বলো তথ্য নেই বা আরেকটু স্পষ্ট করতে বলো
- তোমাকে তৈরি করেছে ঢাকা পলিটেকনিক ইনস্টিটিউট এর ইলেকট্রিক্যাল টেকনোলজির ২০২৩ - ২০২৪ সেশনের ৩ জন ছাত্র। তারা হলো সাকিন আল আনফি, মুওদুদ বিশ্বাস,নাফিজুর রহমান নূর।
- ক্যাম্পাস অফিস টাইম ২ ভাগে বিভক্ত।প্রথম শিফটের জন্য সকাল ৭ টা থেকে ১:১৫ দ্বিতীয় শিফটের জন্য ১:১৫ থেকে ৫:৩০
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
        # NOTE: max_output_tokens limit removed per request ("erase all token limits").
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt
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

        text = re.sub(r'#{1,6}\s*', '', text)           # headers
        text = re.sub(r'\*{1,3}', '', text)              # bold / italic *
        text = re.sub(r'_{1,3}', '', text)               # bold / italic _
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)   # bullet lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)   # numbered lists
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE) # horizontal rules ---
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)        # blockquotes
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)                  # inline/block code
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)          # links → keep label
        text = re.sub(r'\|', ' ', text)                  # table pipes
        text = re.sub(r'\n{2,}', '\n', text)             # collapse blank lines
        text = re.sub(r'[ \t]{2,}', ' ', text)           # collapse spaces
        text = text.strip()

        # NOTE: 800-char truncation removed per request ("erase all token limits").
        # gTTS can handle longer text; this may just take a little longer to synthesize.

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