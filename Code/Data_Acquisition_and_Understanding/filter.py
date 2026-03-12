

import subprocess, sys
for pkg in ["PyMuPDF"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("✓ Dependencies ready")


import os
import re
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF

BASE = "/content/drive/MyDrive/data sci capstone files"

PDF_DIRS = [
    "/content/drive/MyDrive/data sci capstone files/downloaded_pdfs_raw"
]

OUTPUT_DIR = BASE + "/pipeline_outputs"
MIN_WORDS   = 2000
YEAR_MIN    = 1980
YEAR_MAX    = 2024

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Config set")
print(f"  Input folders : {len(PDF_DIRS)}")
print(f"  Output dir    : {OUTPUT_DIR}")


# Helper Functions
def extract_text(path: str) -> str:
    """Extract raw text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def parse_metadata(filename: str) -> dict:
    """
    Extract title, writer, year from filename.
    Handles both formats:
      - Clean:  House-of-Dracula-(1945).pdf
      - Legacy: The Purge_James DeMonaco_Film_7 6 2011_____.pdf
    """
    stem = Path(filename).stem

    # Extract year from anywhere in filename
    year_match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})', stem)
    year = int(year_match.group()) if year_match else None

    # Try legacy underscore format
    parts = stem.split("_")
    if len(parts) >= 3:
        title  = parts[0].strip()
        writer = parts[1].strip()
    else:
        # Clean format — derive title from stem
        title  = re.sub(r'\(\d{4}\)', '', stem)
        title  = re.sub(r'[-_]', ' ', title).strip()
        writer = "Unknown"

    return {
        "filename": filename,
        "title":    title,
        "writer":   writer,
        "year":     year,
    }


def era_label(year) -> str:
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return "Unknown"
    year = int(year)
    if year <= 1979:   return "Pre-1980"
    if year <= 1995:   return "1980-1995 (Baseline)"
    if year <= 2009:   return "1996-2009"
    return "2010-Present"


def is_usable(text: str, filename: str) -> tuple[bool, str]:
    """
    Returns (True, 'ok') if script passes all quality checks.
    Returns (False, reason) if it should be filtered out.
    """
    # 1. Word count
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False, f"too short ({word_count} words)"

    # 2. Year in filename
    year_match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})', filename)
    if not year_match:
        return False, "no year in filename"

    # 3. Year in scope
    year = int(year_match.group())
    if year < YEAR_MIN or year > YEAR_MAX:
        return False, f"year {year} outside {YEAR_MIN}-{YEAR_MAX}"

    # 4. Script formatting indicators
    script_indicators = [
        'INT.', 'EXT.', 'FADE IN', 'FADE OUT',
        'CUT TO', 'DISSOLVE', 'INT ', 'EXT ',
        'SMASH CUT', 'MATCH CUT'
    ]
    has_format = any(ind in text.upper() for ind in script_indicators)
    is_outline  = any(w in filename.lower() for w in
                      ['outline', 'beat', 'treatment', 'beatsheet'])

    if not has_format and not is_outline:
        return False, "no script formatting detected"

    # 5. OCR quality check
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False, f"bad OCR (alpha ratio {alpha_ratio:.2f})"

    return True, "ok"


# Collect all PDF paths 

all_pdfs = []
for folder in PDF_DIRS:
    if not os.path.exists(folder):
        print(f"⚠ Folder not found: {folder}")
        continue
    found = [os.path.join(folder, f)
             for f in os.listdir(folder)
             if f.lower().endswith(".pdf")]
    print(f"  {len(found):4d} PDFs  ←  {folder.split('/')[-1]}")
    all_pdfs.extend(found)

total_gb = sum(os.path.getsize(p) for p in all_pdfs) / 1e9
print(f"\n✓ Total: {len(all_pdfs)} PDFs ({total_gb:.1f} GB)")


all_pdfs = all_pdfs[:1000]
records  = []
skipped  = {}

for fpath in tqdm(all_pdfs, desc="Filtering"):
    fname = os.path.basename(fpath)
    text  = extract_text(fpath)
    usable, reason = is_usable(text, fname)

    if not usable:
        skipped[fname] = reason
        continue

    meta = parse_metadata(fname)
    meta["full_path"]   = fpath
    meta["word_count"]  = len(text.split())
    meta["era"]         = era_label(meta["year"])
    meta["source_folder"] = Path(fpath).parent.name
    records.append(meta)


# Save outputs
usable_df  = pd.DataFrame(records)
filter_log = pd.DataFrame([
    {"filename": k, "reason": v} for k, v in skipped.items()
])

usable_path = os.path.join(OUTPUT_DIR, "usable_scripts.csv")
filter_path = os.path.join(OUTPUT_DIR, "filter_log.csv")

usable_df.to_csv(usable_path, index=False)
filter_log.to_csv(filter_path, index=False)

print(f"✓ Saved usable_scripts.csv  ({len(usable_df)} scripts)")
print(f"✓ Saved filter_log.csv      ({len(filter_log)} filtered out)")



print("\n" + "=" * 55)
print("FILTER SUMMARY")
print("=" * 55)
print(f"  Total PDFs scanned : {len(all_pdfs)}")
print(f"  Usable scripts     : {len(usable_df)}")
print(f"  Filtered out       : {len(skipped)}")
print(f"  Pass rate          : {len(usable_df)/max(len(all_pdfs),1)*100:.1f}%")

print(f"\nTop filter reasons:")
for reason, count in Counter(skipped.values()).most_common(8):
    print(f"  {count:4d} × {reason}")

print(f"\nEra distribution (usable scripts):")
print(usable_df["era"].value_counts().to_string())

print(f"\nYear range : {usable_df['year'].min():.0f} – {usable_df['year'].max():.0f}")
print(f"Avg words  : {usable_df['word_count'].mean():.0f}")

print(f"\nSource folder breakdown:")
print(usable_df["source_folder"].value_counts().to_string())

print(f"\nOutput files:")
print(f"  {usable_path}")
print(f"  {filter_path}")
print("=" * 55)
print("\n✓ Ready to run narrative_pipeline.py")