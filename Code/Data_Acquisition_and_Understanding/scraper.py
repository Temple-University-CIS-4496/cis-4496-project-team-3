import os
import re
import time
import json
import random
import hashlib
import requests


# CONFIG

SUPABASE_URL = "https://ceyhlqhetmcpzpzzrppv.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNleWhscWhldG1jcHpwenpycHB2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI5MjE5MzIsImV4cCI6MjA1ODQ5NzkzMn0.oMs4gqPFi2P70khV2TvsKo-9eRN4Lf_EopB_oABKY6E"

# ⚠ Refresh this from DevTools > Network > any request > Authorization header (expires hourly)
SESSION_TOKEN = "your-session-token" # Expires hourly
CATEGORY_FILTER = "Film"   # Set to None to download all categories
LIMIT = 100                # Set to None for full run

OUT_DIR = "downloaded_pdfs"
MANIFEST_PATH = "download_manifest.json"

MIN_DELAY_S = 2
MAX_DELAY_S = 5
MAX_RETRIES = 3

SUPABASE_HEADERS = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {SESSION_TOKEN}",
    "Content-Type": "application/json",
}


# Supabase fetch

def get_all_records(category=None) -> list[dict]:
    all_records = []
    limit = 1000
    offset = 0
    filter_str = f"&Category=eq.{category}" if category else ""
    print(f"\n📡 Fetching records from Supabase{f' (category: {category})' if category else ''}...")

    while True:
        url = (
            f"{SUPABASE_URL}/rest/v1/files"
            f"?select=*"
            f"{filter_str}"
            f"&record_deleted=eq.false"
            f"&limit={limit}&offset={offset}"
        )
        r = requests.get(url, headers=SUPABASE_HEADERS)

        if r.status_code == 401:
            print("✗ 401 Unauthorized — SESSION_TOKEN has expired.")
            print("  Refresh: DevTools > Network > any request > Authorization header.")
            return []

        if r.status_code != 200:
            print(f"✗ API error {r.status_code}: {r.text}")
            return []

        batch = r.json()
        if not batch:
            break

        all_records.extend(batch)
        print(f"  {len(all_records)} records fetched...")

        if len(batch) < limit:
            break
        offset += limit

    print(f"✓ Total records: {len(all_records)}")
    return all_records



# Supabase signed URL + download

def get_signed_url(filename: str, expires_in: int = 300) -> str | None:
    """
    Request a signed download URL from Supabase storage.
    POST /storage/v1/object/sign/files/<filename>
    Returns the signed URL string or None on failure.
    """
    endpoint = f"{SUPABASE_URL}/storage/v1/object/sign/files/{filename}"
    payload = {"expiresIn": expires_in}

    r = requests.post(endpoint, headers=SUPABASE_HEADERS, json=payload, timeout=15)

    if r.status_code == 401:
        print("    ✗ 401 — SESSION_TOKEN expired. Update it at the top of the script.")
        return None
    if r.status_code not in (200, 201):
        print(f"    ✗ Signing failed ({r.status_code}): {r.text[:200]}")
        return None

    data = r.json()

    # Response contains signedURL or signedUrl
    signed = data.get("signedURL") or data.get("signedUrl") or data.get("signed_url")
    if not signed:
        print(f"    ✗ No signed URL in response: {data}")
        return None

    # Make sure it's a full URL
    if signed.startswith("/object"):
        signed = SUPABASE_URL + "/storage/v1" + signed

    return signed


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:200]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded": {}}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def download_pdf(session: requests.Session, signed_url: str, filename: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, safe_filename(filename))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.scripthive.com/",
    }

    with session.get(signed_url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()

        # Verify it's actually a PDF
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type:
            raise ValueError(f"Got HTML instead of PDF — signed URL may have expired")

        size = r.headers.get("Content-Length")
        if size:
            print(f"    Size: {int(size)/1024:.1f} KB")

        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)

    return path



# Main

def main():
    print("=" * 60)
    print("SCRIPTHIVE → SUPABASE STORAGE PDF DOWNLOADER")
    print("=" * 60)

    manifest = load_manifest()
    print(f"Manifest: {len(manifest['downloaded'])} previously downloaded")

    records = get_all_records(category=CATEGORY_FILTER)
    if not records:
        return

    # Build task list — Title field is the Supabase storage filename
    tasks = []
    for rec in records:
        filename = rec.get("Title")   
        if not rec.get("storage_hash"):
            continue
        if not filename.lower().endswith(".pdf"):
            continue
        tasks.append((filename, rec))

    # Apply limit for test runs
    if LIMIT:
        tasks = tasks[:LIMIT]

    print(f"\n📊 {len(tasks)} PDFs to download")
    if not tasks:
        return

    print(f"📥 Saving to ./{OUT_DIR}/\n")
    success, failed = 0, 0

    with requests.Session() as session:
        for idx, (filename, record) in enumerate(tasks, 1):
            file_id = record.get("file_id") or filename

            if file_id in manifest["downloaded"]:
                print(f"[{idx}/{len(tasks)}] ⏭  Skip: {filename[:60]}")
                success += 1
                continue

            print(f"[{idx}/{len(tasks)}] 📄 {filename[:70]}")

            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    #  Get a fresh signed URL
                    signed_url = get_signed_url(filename)
                    if not signed_url:
                        raise ValueError("Could not get signed URL")

                    #  Download using the signed URL
                    path = download_pdf(session, signed_url, filename)
                    digest = sha256_file(path)

                    manifest["downloaded"][file_id] = {
                        "filename": os.path.basename(path),
                        "sha256": digest,
                        "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "category": record.get("Category"),
                        "writer": record.get("Writer(s)"),
                        "year": record.get("Draft Date"),
                    }
                    save_manifest(manifest)
                    print(f"    ✓ Saved: {os.path.basename(path)}")
                    success += 1
                    break

                except Exception as e:
                    last_err = e
                    print(f"    ✗ Attempt {attempt}/{MAX_RETRIES}: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep((2 ** attempt) + random.uniform(0, 2))
            else:
                print(f"    ✗ FAILED permanently: {last_err}")
                failed += 1

            if idx < len(tasks):
                time.sleep(random.uniform(MIN_DELAY_S, MAX_DELAY_S))

    print("\n" + "=" * 60)
    print(f"DONE — ✓ {success} downloaded, ✗ {failed} failed")
    print(f"Files: ./{OUT_DIR}/  |  Manifest: {MANIFEST_PATH}")
    print("=" * 60)


main()