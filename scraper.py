import os
import re
import requests
from datetime import datetime
from supabase import create_client
from bs4 import BeautifulSoup

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

OCR_API_KEY = os.environ.get("OCR_API_KEY")
TARGET_URL = "https://dhaka.polytech.gov.bd/pages/notices"
BASE_URL = "https://dhaka.polytech.gov.bd"

def scrape_with_jina(url):
    """Method 1 - Jina AI Reader (free, no signup needed!)"""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "text"
        }
        response = requests.get(jina_url, headers=headers, timeout=30)
        if response.status_code == 200:
            print("Jina AI scraping successful!")
            return response.text
        print(f"Jina failed with status: {response.status_code}")
        return None
    except Exception as e:
        print(f"Jina error: {e}")
        return None

def scrape_with_bs4(url):
    """Method 2 - BeautifulSoup fallback"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except Exception as e:
        print(f"BS4 error: {e}")
        return None

def extract_pdf_links_from_text(text):
    """Extract PDF links from Jina text output"""
    pdf_pattern = r'https?://[^\s\)]+\.pdf'
    links = re.findall(pdf_pattern, text)
    return list(set(links))  # remove duplicates

def extract_notice_links_from_text(text):
    """Extract all links and titles from Jina text output"""
    notices = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Jina formats links like: [Title](URL)
        link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        matches = re.findall(link_pattern, line)

        for title, url in matches:
            if len(title) > 5:
                notices.append({
                    "title": title.strip(),
                    "link": url.strip(),
                    "is_pdf": url.lower().endswith(".pdf")
                })

        # Also check plain text lines for notice-like content
        if any(word in line for word in ["নোটিশ", "বিজ্ঞপ্তি", "Notice", "Circular", "notice"]):
            if len(line) > 10 and "http" not in line:
                notices.append({
                    "title": line,
                    "link": TARGET_URL,
                    "is_pdf": False
                })

    return notices

def extract_links_from_bs4(soup):
    """Extract links from BeautifulSoup"""
    notices = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)

        if not text or len(text) < 5:
            continue

        if href.startswith("/"):
            href = BASE_URL + href
        elif not href.startswith("http"):
            continue

        notices.append({
            "title": text,
            "link": href,
            "is_pdf": href.lower().endswith(".pdf")
        })

    return notices

def ocr_pdf(pdf_url):
    """Extract text from scanned PDF using OCR.space"""
    try:
        print(f"Running OCR on: {pdf_url}")
        payload = {
            "url": pdf_url,
            "apikey": OCR_API_KEY,
            "language": "bng",
            "isOverlayRequired": False,
            "detectOrientation": True,
            "scale": True,
            "OCREngine": 2
        }
        response = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            timeout=60
        )
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            print(f"OCR error: {result.get('ErrorMessage')}")
            return None

        parsed = result.get("ParsedResults", [])
        if parsed:
            text = parsed[0].get("ParsedText", "").strip()
            print(f"OCR extracted {len(text)} characters")
            return text
        return None

    except Exception as e:
        print(f"OCR error: {e}")
        return None

def save_notice(title, content, source):
    """Save to Supabase if not already exists"""
    try:
        existing = supabase.table("notices")\
            .select("id")\
            .eq("title", title)\
            .execute()

        if existing.data:
            print(f"Already exists: {title}")
            return False

        supabase.table("notices").insert({
            "title": title,
            "content": content,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": source
        }).execute()

        print(f"Saved: {title}")
        return True

    except Exception as e:
        print(f"Save error: {e}")
        return False

def run_scraper():
    print("=" * 50)
    print("Starting DPI Notice Scraper...")
    print("=" * 50)

    notices = []

    # Step 1 — Try Jina AI first
    print("\nStep 1: Trying Jina AI Reader...")
    jina_text = scrape_with_jina(TARGET_URL)

    if jina_text:
        # Extract links from Jina output
        notices = extract_notice_links_from_text(jina_text)
        pdf_links = extract_pdf_links_from_text(jina_text)

        # Add standalone PDF links
        for pdf in pdf_links:
            title = pdf.split('/')[-1].replace('.pdf', '').replace('-', ' ').replace('_', ' ')
            if not any(n['link'] == pdf for n in notices):
                notices.append({
                    "title": title,
                    "link": pdf,
                    "is_pdf": True
                })
    else:
        # Step 2 — Fallback to BeautifulSoup
        print("\nStep 2: Jina failed, trying BeautifulSoup...")
        soup = scrape_with_bs4(TARGET_URL)
        if soup:
            notices = extract_links_from_bs4(soup)

    print(f"\nFound {len(notices)} notices total")

    # Step 3 — Process and save each notice
    saved = 0
    for i, notice in enumerate(notices[:20]):  # limit 20 per run
        title = notice["title"]
        link = notice["link"]

        print(f"\n[{i+1}] Processing: {title[:50]}...")

        if notice["is_pdf"] and OCR_API_KEY:
            content = ocr_pdf(link)
            if not content:
                content = f"PDF নোটিশ। বিস্তারিত দেখুন: {link}"
        else:
            content = f"বিস্তারিত দেখুন: {link}"

        if save_notice(title, content, link):
            saved += 1

    print(f"\nDone! Saved {saved} new notices to Supabase!")
    return saved

if __name__ == "__main__":
    run_scraper()