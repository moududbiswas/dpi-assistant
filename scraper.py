import os
import re
import hashlib
import base64
import requests
from datetime import datetime
from supabase import create_client
from mistralai.client import MistralClient
from groq import Groq
from pdf2image import convert_from_bytes
import io

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# CONFIG
# ==============================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TARGET_URL = "https://dhaka.polytech.gov.bd/pages/notices"
MAX_CHARS = 3000

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


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
            pdf_links.append({
                "title": title.strip(),
                "url": url.strip()
            })

    # Raw PDF links
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

    print(f"Found {len(pdf_links)} PDFs")
    return pdf_links


# ==============================
# STEP 2 — Mistral OCR
# ==============================
def ocr_with_mistral(pdf_url: str):
    try:
        print(f"Downloading PDF: {pdf_url[:60]}...")
        response = requests.get(pdf_url, timeout=30, verify=False)
        if response.status_code != 200:
            return None

        pdf_bytes = response.content
        print(f"PDF size: {len(pdf_bytes)} bytes")

        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=2)

        all_text = []

        for i, image in enumerate(images):
            print(f"Processing page {i+1}...")

            # Convert image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            result = mistral_client.chat(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{img_base64}"
                            },
                            {
                                "type": "text",
                                "text": "এই ছবি থেকে সমস্ত বাংলা ও ইংরেজি টেক্সট বের করো। শুধু টেক্সট দাও।"
                            }
                        ]
                    }
                ]
            )

            text = result.choices[0].message.content.strip()
            all_text.append(text)

        full_text = "\n".join(all_text)
        cleaned = clean_text(full_text)
        print(f"Extracted {len(cleaned)} characters")
        return cleaned

    except Exception as e:
        print(f"Mistral OCR error: {e}")
        return None


# ==============================
# STEP 3 — AI Summarize
# ==============================
def summarize_with_ai(title: str, ocr_text: str) -> str:
    if not ocr_text or len(ocr_text) < 20:
        return f"নোটিশ: {title}"

    try:
        print("Summarizing with Groq...")
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
        print(f"Summary done: {summary[:60]}...")
        return summary

    except Exception as e:
        print(f"Summarize error: {e}")
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

    latest = pdf_links[:5]
    print(f"\nProcessing latest {len(latest)} PDFs...")
    saved = 0

    for i, item in enumerate(latest):
        title = item["title"]
        url = item["url"]

        print(f"\n[{i+1}] {title[:60]}...")

        # Step 2 — Mistral OCR
        ocr_text = ocr_with_mistral(url)

        if not ocr_text:
            print("OCR failed, saving with link only...")
            summary = f"নোটিশ: {title}\nবিস্তারিত দেখুন: {url}"
        else:
            # Step 3 — Summarize
            summary = summarize_with_ai(title, ocr_text)

        content_hash = sha256((ocr_text or title) + url)

        if notice_exists(content_hash):
            print("Already exists, skipping...")
            continue

        if save_notice(title, ocr_text, summary, url, content_hash):
            saved += 1
            print("Saved!")
        else:
            print("Failed to save!")

    print(f"\n{'='*50}")
    print(f"Done! Saved {saved} new notices!")
    print(f"{'='*50}")
    return saved


if __name__ == "__main__":
    run_scraper()