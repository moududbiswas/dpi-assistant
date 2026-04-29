import os
import re
import hashlib
import requests
from datetime import datetime
from supabase import create_client
from groq import Groq

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# CONFIG
# ==============================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OCR_API_KEY = os.environ.get("OCR_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TARGET_URL = "https://dhaka.polytech.gov.bd/pages/notices"
MAX_OCR_CHARS = 5000

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


# ==============================
# HELPERS
# ==============================
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.I)
    text = re.sub(r"_{2,}|-{2,}", "", text)
    return text.strip()[:MAX_OCR_CHARS]


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

    # Extract markdown links with PDF URLs — [Title](URL.pdf)
    md_pattern = r'\[([^\]]+)\]\((https?://[^\)]+\.pdf)\)'
    md_matches = re.findall(md_pattern, text, re.IGNORECASE)

    for title, url in md_matches:
        if url not in seen:
            seen.add(url)
            pdf_links.append({
                "title": title.strip(),
                "url": url.strip()
            })

    # Also grab raw PDF links not in markdown format
    raw_pattern = r'https?://[^\s\)\]]+\.pdf'
    raw_links = re.findall(raw_pattern, text, re.IGNORECASE)

    for url in raw_links:
        if url not in seen:
            seen.add(url)
            filename = url.split("/")[-1].replace(".pdf", "")
            filename = re.sub(r'[-_]', ' ', filename)
            pdf_links.append({
                "title": filename[:80],
                "url": url
            })

    print(f"Found {len(pdf_links)} PDF links")
    return pdf_links


# ==============================
# STEP 2 — OCR the PDF
# ==============================
def ocr_pdf(pdf_url: str):
    if not OCR_API_KEY:
        print("No OCR API key!")
        return None

    try:
        print(f"Running OCR...")
        payload = {
            "url": pdf_url,
            "apikey": OCR_API_KEY,
            "OCREngine": 2,
            "scale": True,
            "detectOrientation": True,
            "isOverlayRequired": False,
            # No language — auto detect works best for Bengali!
        }
        response = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            timeout=60,
        )
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            print(f"OCR error: {result.get('ErrorMessage')}")
            return None

        parsed = result.get("ParsedResults", [])
        if not parsed:
            print("No OCR results!")
            return None

        text = parsed[0].get("ParsedText", "")
        cleaned = clean_ocr_text(text)
        print(f"OCR extracted {len(cleaned)} characters")
        return cleaned

    except Exception as e:
        print(f"OCR error: {e}")
        return None


# ==============================
# STEP 3 — AI Summarize (once!)
# ==============================
def summarize_with_ai(title: str, ocr_text: str) -> str:
    if not ocr_text or len(ocr_text) < 20:
        return f"নোটিশ: {title}"

    try:
        print("Summarizing with AI...")
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "তুমি একটি নোটিশ সারসংক্ষেপকারী। বাংলায় ৩-৪ বাক্যে নোটিশের মূল বিষয় সংক্ষেপ করো। শুধু গুরুত্বপূর্ণ তথ্য রাখো।"
                },
                {
                    "role": "user",
                    "content": f"এই নোটিশটি সংক্ষেপ করো:\n\nশিরোনাম: {title}\n\nবিষয়বস্তু:\n{ocr_text[:2000]}"
                }
            ],
            max_tokens=300
        )
        summary = response.choices[0].message.content.strip()
        print(f"Summary: {summary[:80]}...")
        return summary

    except Exception as e:
        print(f"AI summarize error: {e}")
        return f"নোটিশ: {title}\n\n{ocr_text[:300]}"


# ==============================
# STEP 4 — Database
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
    print("DPI Notice Scraper Starting...")
    print("=" * 50)

    pdf_links = get_pdf_links()

    if not pdf_links:
        print("No PDFs found!")
        return 0

    # Only latest 5
    latest = pdf_links[:5]
    print(f"\nProcessing latest {len(latest)} PDFs...")
    saved = 0

    for i, item in enumerate(latest):
        title = item["title"]
        url = item["url"]

        print(f"\n[{i+1}] {title[:60]}...")
        print(f"URL: {url[:70]}...")

        # Step 2 — OCR
        ocr_text = ocr_pdf(url)

        if not ocr_text:
            print("OCR failed, saving with link only...")
            summary = f"নোটিশ: {title}\nবিস্তারিত দেখুন: {url}"
        else:
            # Step 3 — AI summarize once
            summary = summarize_with_ai(title, ocr_text)

        # Check duplicate
        content_hash = sha256((ocr_text or title) + url)

        if notice_exists(content_hash):
            print("Already exists, skipping...")
            continue

        # Step 4 — Save
        if save_notice(title, ocr_text, summary, url, content_hash):
            saved += 1
            print(f"Saved!")
        else:
            print("Failed to save!")

    print(f"\n{'='*50}")
    print(f"Done! Saved {saved} new notices!")
    print(f"{'='*50}")
    return saved


if __name__ == "__main__":
    run_scraper()