import os
import re
import hashlib
import base64
import requests
from datetime import datetime
from supabase import create_client
import google.generativeai as genai
from pdf2image import convert_from_bytes
import io

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# CONFIG
# ==============================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

TARGET_URL = "https://dhaka.polytech.gov.bd/pages/notices"
MAX_CHARS = 5000
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)


# ==============================
# HELPERS
# ==============================
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_{2,}|-{2,}", "", text)
    return text.strip()[:MAX_CHARS]


# ==============================
# STEP 1 — Get PDF links via Jina
# ==============================
def get_pdf_links():
    print("Fetching PDF links via Jina AI...")

    jina_url = f"https://r.jina.ai/{TARGET_URL}"
    headers = {
        "Accept": "text/plain",
        "X-Return-Format": "markdown"
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=60)
        if response.status_code != 200:
            print(f"Jina failed: {response.status_code}")
            return []

        text = response.text
        print(f"Jina fetched {len(text)} characters")

    except Exception as e:
        print(f"Jina error: {e}")
        return []

    pdf_links = []
    seen = set()

    # Markdown links with PDF URLs
    md_pattern = r'\[([^\]]+)\]\((https?://[^\)]+\.pdf)\)'
    md_matches = re.findall(md_pattern, text, re.IGNORECASE)

    for title, url in md_matches:
        if url not in seen:
            seen.add(url)
            pdf_links.append({"title": title.strip(), "url": url.strip()})

    # Raw PDF links
    raw_pattern = r'https?://[^\s\)\]]+\.pdf'
    raw_links = re.findall(raw_pattern, text, re.IGNORECASE)

    for url in raw_links:
        if url not in seen:
            seen.add(url)
            filename = url.split("/")[-1].replace(".pdf", "")
            filename = re.sub(r'[-_]', ' ', filename)
            pdf_links.append({"title": filename[:80], "url": url})

    print(f"Found {len(pdf_links)} PDFs")
    return pdf_links


# ==============================
# STEP 2 — Gemini OCR + Summarize (single call per page)
# ==============================
def ocr_and_summarize_with_gemini(title: str, pdf_url: str):
    """
    Downloads PDF → converts to images → sends to Gemini.
    OCR + summarization in ONE call per page. No separate summarize step needed.
    Returns (raw_text, summary)
    """
    try:
        print(f"Downloading PDF: {pdf_url[:60]}...")
        response = requests.get(pdf_url, timeout=30, verify=False)
        if response.status_code != 200:
            print(f"Download failed: {response.status_code}")
            return None, None

        pdf_bytes = response.content
        print(f"PDF size: {len(pdf_bytes)} bytes")

        print("Converting PDF to images...")
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=2)

        model = genai.GenerativeModel(GEMINI_MODEL)
        all_raw_text = []
        all_summaries = []

        for i, image in enumerate(images):
            print(f"Processing page {i+1} with Gemini...")

            # Convert image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            # Gemini vision — OCR + summarize in one shot
            response = model.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode("utf-8")
                },
                f"""এই নোটিশ পেজ থেকে দুইটি কাজ করো:

১. RAW_TEXT: সমস্ত বাংলা ও ইংরেজি টেক্সট হুবহু বের করো।
২. SUMMARY: ৩-৪ বাক্যে বাংলায় নোটিশের মূল বিষয় সংক্ষেপ করো।

এই ফরম্যাটে দাও:
RAW_TEXT:
[এখানে সব টেক্সট]

SUMMARY:
[এখানে সংক্ষেপ]

নোটিশের শিরোনাম: {title}"""
            ])

            result_text = response.text.strip()

            # Parse RAW_TEXT and SUMMARY from response
            raw_match = re.search(r'RAW_TEXT:\s*(.*?)(?=SUMMARY:|$)', result_text, re.DOTALL)
            summary_match = re.search(r'SUMMARY:\s*(.*?)$', result_text, re.DOTALL)

            raw = raw_match.group(1).strip() if raw_match else result_text
            summary = summary_match.group(1).strip() if summary_match else ""

            all_raw_text.append(raw)
            if summary:
                all_summaries.append(summary)

        full_raw = clean_text("\n".join(all_raw_text))
        full_summary = " ".join(all_summaries) if all_summaries else f"নোটিশ: {title}"

        print(f"OCR: {len(full_raw)} chars | Summary: {len(full_summary)} chars")
        return full_raw, full_summary

    except Exception as e:
        print(f"Gemini OCR error: {e}")
        return None, None


# ==============================
# STEP 3 — Database
# ==============================
def notice_exists(content_hash: str) -> bool:
    try:
        res = supabase.table("notices") \
            .select("id") \
            .eq("content_hash", content_hash) \
            .limit(1) \
            .execute()
        return bool(res.data)
    except Exception as e:
        print(f"DB check error: {e}")
        return False


def save_notice(title, raw_ocr, summary, source, content_hash):
    try:
        supabase.table("notices").insert({
            "title": title,
            "content": summary,
            "raw_content": raw_ocr,
            "source": source,
            "content_hash": content_hash,
            "date": datetime.now().strftime("%Y-%m-%d")
        }).execute()
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False


# ==============================
# MAIN
# ==============================
def run_scraper():
    print("=" * 50)
    print("DPI Notice Scraper Starting... (Gemini)")
    print("=" * 50)

    pdf_links = get_pdf_links()

    if not pdf_links:
        print("No PDFs found!")
        return 0

    latest = pdf_links[:5]
    print(f"\nProcessing latest {len(latest)} PDFs...")
    saved = 0

    for i, item in enumerate(latest):
        title = item["title"]
        url = item["url"]

        print(f"\n[{i+1}] {title[:60]}...")

        # Hash check before downloading (saves API calls)
        quick_hash = sha256(title + url)
        if notice_exists(quick_hash):
            print("Already exists, skipping...")
            continue

        # Gemini OCR + Summarize
        raw_text, summary = ocr_and_summarize_with_gemini(title, url)

        if not raw_text:
            print("OCR failed, saving title + link only...")
            raw_text = ""
            summary = f"নোটিশ: {title}\nবিস্তারিত দেখুন: {url}"

        content_hash = sha256((raw_text or title) + url)

        if notice_exists(content_hash):
            print("Already exists (content match), skipping...")
            continue

        if save_notice(title, raw_text, summary, url, content_hash):
            saved += 1
            print(f"Saved! Summary: {summary[:80]}...")
        else:
            print("Failed to save!")

    print(f"\n{'='*50}")
    print(f"Done! Saved {saved} new notices!")
    print(f"{'='*50}")
    return saved


if __name__ == "__main__":
    run_scraper()