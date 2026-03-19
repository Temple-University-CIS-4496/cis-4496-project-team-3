import os
import re
import fitz
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_FOLDER = r"G:\.shortcut-targets-by-id\1zMZhOhWAJ4OREe9g3XIkRb34qNHbX6bw\data sci capstone files\downloaded_pdfs_raw"
OUTPUT_FOLDER = r"C:\Users\vishr\Downloads\cleaned_scripts"

# make each worker write its own result instead of one shared log file
SKIPPED_LOG = r"C:\Users\vishr\Downloads\skipped_scripts_parallel.txt"
PROGRESS_LOG = r"C:\Users\vishr\Downloads\current_progress_parallel.txt"

# ========= SETTINGS =========
SAMPLE_PAGES = 2
MIN_SAMPLE_CHARS = 50
MIN_SAMPLE_WORDS = 20
PRINT_EVERY = 100
PROGRESS_EVERY = 200

# number of parallel workers
MAX_WORKERS = 4

# optional chunking if you want to split the full dataset across separate runs
TOTAL_CHUNKS = 1
CHUNK_INDEX = 0
# ============================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PAGE_NUM_RE1 = re.compile(r"^\d+$")
PAGE_NUM_RE2 = re.compile(r"^-?\s*\d+\s*-?$")
PAGE_NUM_RE3 = re.compile(r"^page\s+\d+$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]{2,}")
SPACE_RE = re.compile(r"[ \t]+")
BLANKLINES_RE = re.compile(r"\n{3,}")


def remove_page_number_lines(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        s = line.strip()
        if PAGE_NUM_RE1.fullmatch(s):
            continue
        if PAGE_NUM_RE2.fullmatch(s):
            continue
        if PAGE_NUM_RE3.fullmatch(s):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_text(text):
    text = text.lower()
    text = remove_page_number_lines(text)
    text = SPACE_RE.sub(" ", text)
    text = BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def process_one(file_name):
    pdf_path = os.path.join(INPUT_FOLDER, file_name)
    output_name = os.path.splitext(file_name)[0] + ".txt"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # skip if already done
    if os.path.exists(output_path):
        return ("already_done", file_name, "")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return ("skipped", file_name, f"open failed: {str(e)}")

    sample_parts = []
    full_parts = []

    try:
        for page_index, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
            except Exception:
                page_text = ""

            if page_index < SAMPLE_PAGES:
                sample_parts.append(page_text)

            full_parts.append(page_text)

        doc.close()
    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass
        return ("skipped", file_name, f"read failed: {str(e)}")

    sample_text = "\n".join(sample_parts).strip()

    if len(sample_text) < MIN_SAMPLE_CHARS:
        return ("skipped", file_name, "not selectable or unreadable")

    if len(WORD_RE.findall(sample_text)) < MIN_SAMPLE_WORDS:
        return ("skipped", file_name, "not selectable or unreadable")

    full_text = "\n".join(full_parts)
    cleaned = clean_text(full_text)

    if len(cleaned) <= 50:
        return ("skipped", file_name, "extracted text too short")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
    except Exception as e:
        return ("skipped", file_name, f"write failed: {str(e)}")

    return ("processed", file_name, "")


def main():
    pdf_files = sorted(
        entry.name for entry in os.scandir(INPUT_FOLDER)
        if entry.is_file() and entry.name.lower().endswith(".pdf")
    )

    if TOTAL_CHUNKS > 1:
        pdf_files = pdf_files[CHUNK_INDEX::TOTAL_CHUNKS]

    total_files = len(pdf_files)
    completed = 0
    processed = 0
    skipped = 0
    already_done = 0

    skipped_buffer = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, file_name): file_name for file_name in pdf_files}

        for future in as_completed(futures):
            completed += 1

            try:
                status, file_name, message = future.result()
            except Exception as e:
                status = "skipped"
                file_name = futures[future]
                message = f"worker crashed: {str(e)}"

            if status == "processed":
                processed += 1
            elif status == "already_done":
                already_done += 1
            else:
                skipped += 1
                skipped_buffer.append(f"{file_name} | {message}\n")

            if len(skipped_buffer) >= 100:
                with open(SKIPPED_LOG, "a", encoding="utf-8") as log_file:
                    log_file.writelines(skipped_buffer)
                skipped_buffer = []

            if completed % PRINT_EVERY == 0:
                print(
                    f"Completed {completed}/{total_files} | "
                    f"Processed: {processed} | "
                    f"Skipped: {skipped} | "
                    f"Already done: {already_done}"
                )

            if completed % PROGRESS_EVERY == 0:
                with open(PROGRESS_LOG, "w", encoding="utf-8") as prog:
                    prog.write(f"Chunk index: {CHUNK_INDEX}\n")
                    prog.write(f"Total chunks: {TOTAL_CHUNKS}\n")
                    prog.write(f"Completed in this chunk: {completed}/{total_files}\n")
                    prog.write(f"Processed this run: {processed}\n")
                    prog.write(f"Skipped this run: {skipped}\n")
                    prog.write(f"Already done skipped: {already_done}\n")

    if skipped_buffer:
        with open(SKIPPED_LOG, "a", encoding="utf-8") as log_file:
            log_file.writelines(skipped_buffer)

    print("\nDONE")
    print("Chunk index:", CHUNK_INDEX)
    print("Total chunks:", TOTAL_CHUNKS)
    print("Processed this run:", processed)
    print("Skipped this run:", skipped)
    print("Already done skipped:", already_done)
    print("Cleaned files saved to:", OUTPUT_FOLDER)
    print("Skipped log saved to:", SKIPPED_LOG)
    print("Current progress log:", PROGRESS_LOG)


if __name__ == "__main__":
    main()