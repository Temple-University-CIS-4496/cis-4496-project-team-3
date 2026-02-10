import os
import re
import time
import json
import random
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ----------------------------
# CONFIG (fill these in)
# ----------------------------
BASE_URL = "https://example.com/"
LOGIN_URL = urljoin(BASE_URL, "/login")
PDF_LIST_PAGES = [
    urljoin(BASE_URL, "/scripts"),
    # add more listing pages if needed
]

USERNAME = "your_username"
PASSWORD = "your_password"

OUT_DIR = "downloaded_pdfs"
MANIFEST_PATH = "download_manifest.json"

# “human-like” policy knobs (keep conservative)
MIN_DELAY_S = 10
MAX_DELAY_S = 30
TIMEOUT_S = 30
MAX_RETRIES = 3


# ----------------------------
# Helpers
# ----------------------------
def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path) or "file.pdf"
    # sanitize
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded": {}}  # url -> {filename, sha256}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def polite_sleep() -> None:
    time.sleep(random.uniform(MIN_DELAY_S, MAX_DELAY_S))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------
# Login (you may need to adapt)
# ----------------------------
def login(session: requests.Session) -> None:
    """
    Generic login flow:
    1) GET login page (to pick up cookies + maybe CSRF)
    2) POST credentials + CSRF (if any)

    You MUST adjust field names to match the site.
    """
    r = session.get(LOGIN_URL, timeout=TIMEOUT_S)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Common CSRF patterns — adjust as needed
    csrf_token = None
    csrf_input = soup.select_one("input[name=csrf_token]") or soup.select_one("input[name=csrfmiddlewaretoken]")
    if csrf_input and csrf_input.get("value"):
        csrf_token = csrf_input["value"]

    payload = {
        "username": USERNAME,   # maybe "email" on some sites
        "password": PASSWORD,
    }
    if csrf_token:
        # choose correct key based on what the site uses
        if soup.select_one("input[name=csrfmiddlewaretoken]"):
            payload["csrfmiddlewaretoken"] = csrf_token
        else:
            payload["csrf_token"] = csrf_token

    # Some sites require a Referer header
    headers = {"Referer": LOGIN_URL}

    pr = session.post(LOGIN_URL, data=payload, headers=headers, timeout=TIMEOUT_S)
    pr.raise_for_status()

    # Basic sanity check: if login failed, you might still be on login page
    if "login" in pr.url.lower() and ("password" in pr.text.lower() or "invalid" in pr.text.lower()):
        raise RuntimeError("Login likely failed. Check field names / credentials / CSRF handling.")


# ----------------------------
# Crawl PDF links (listing pages)
# ----------------------------
def extract_pdf_links(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        abs_url = urljoin(page_url, href)
        if abs_url.lower().endswith(".pdf"):
            links.append(abs_url)
    # dedupe while preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_pdf_urls(session: requests.Session) -> list[str]:
    all_urls = []
    for page in PDF_LIST_PAGES:
        r = session.get(page, timeout=TIMEOUT_S)
        r.raise_for_status()
        pdfs = extract_pdf_links(r.text, page)
        all_urls.extend(pdfs)
        polite_sleep()
    # dedupe
    seen = set()
    out = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------
# Download with retries + resume
# ----------------------------
def download_pdf(session: requests.Session, url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = safe_filename_from_url(url)
    path = os.path.join(out_dir, filename)

    # stream download
    with session.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        # quick content-type sanity check (some sites mislabel; don’t hard fail)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ct and not url.lower().endswith(".pdf"):
            print(f"Warning: unexpected Content-Type={ct} for {url}")

        tmp_path = path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        os.replace(tmp_path, path)

    return path


def main():
    manifest = load_manifest()

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; polite-downloader/1.0; +contact: you@example.com)"
        })

        # login (if the PDFs require it)
        login(session)
        print("Logged in.")

        pdf_urls = collect_all_pdf_urls(session)
        print(f"Found {len(pdf_urls)} PDF links.")

        for idx, url in enumerate(pdf_urls, start=1):
            if url in manifest["downloaded"]:
                print(f"[{idx}/{len(pdf_urls)}] Skip (already downloaded): {url}")
                continue

            print(f"[{idx}/{len(pdf_urls)}] Download: {url}")

            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    path = download_pdf(session, url, OUT_DIR)
                    digest = sha256_file(path)
                    manifest["downloaded"][url] = {"filename": os.path.basename(path), "sha256": digest}
                    save_manifest(manifest)
                    print(f"  Saved: {path} (sha256={digest[:12]}...)")
                    break
                except Exception as e:
                    last_err = e
                    print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    # backoff + jitter
                    time.sleep((2 ** attempt) + random.uniform(0, 2))
            else:
                print(f"FAILED permanently: {url}\nLast error: {last_err}")

            polite_sleep()


if __name__ == "__main__":
    main()
