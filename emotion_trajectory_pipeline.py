import json
import logging
import re
import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.signal import savgol_filter

# =========================
# CONFIG
# =========================
INPUT_DIR = Path(r"G:\.shortcut-targets-by-id\1zMZhOhWAJ4OREe9g3XIkRb34qNHbX6bw\data sci capstone files\processed_scripts\cleaned_scripts")

OUTPUT_DIR = Path(r"G:\.shortcut-targets-by-id\1zMZhOhWAJ4OREe9g3XIkRb34qNHbX6bw\data sci capstone files\processed_scripts\cleaned_scripts\sorted_by_duplicates")

WINDOW_WORDS = 180
WINDOW_OVERLAP = 150

SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2

MIN_WORDS = 800

# =========================
# SETUP
# =========================
def setup():
    logging.basicConfig(level=logging.INFO)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except:
        nltk.download("vader_lexicon")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT UTILS
# =========================
def tokenize(text):
    return re.findall(r"\b[\w']+\b", text)

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# WINDOWS
# =========================
def make_windows(text):
    words = tokenize(text)
    step = WINDOW_WORDS - WINDOW_OVERLAP

    windows = []
    for i in range(0, len(words), step):
        chunk = words[i:i+WINDOW_WORDS]
        if not chunk:
            break
        windows.append(" ".join(chunk))
    return windows

# =========================
# EMOTION LOGIC
# =========================
def label_emotion(v, a):
    if v > 0.3 and a > 0.5:
        return "excited_positive"
    elif v > 0.3:
        return "calm_positive"
    elif v < -0.3 and a > 0.5:
        return "angry_distressed"
    elif v < -0.3:
        return "sad"
    else:
        return "neutral"

def compute_emotions(windows, analyzer):
    rows = []

    for i, w in enumerate(windows):
        s = analyzer.polarity_scores(w)

        valence = s["compound"]
        arousal = abs(s["compound"]) + 0.5*(s["pos"]+s["neg"])

        label = label_emotion(valence, arousal)

        rows.append({
            "idx": i,
            "text": w,
            "valence": valence,
            "arousal": arousal,
            "emotion": label
        })

    return pd.DataFrame(rows)

# =========================
# SMOOTHING
# =========================
def smooth(series):
    if len(series) < 3:
        return series
    return savgol_filter(series, min(len(series)//2*2-1, SMOOTH_WINDOW), SMOOTH_POLYORDER)

# =========================
# TRANSITIONS
# =========================
def get_transitions(df):
    transitions = []

    for i in range(1, len(df)):
        if df.iloc[i]["emotion"] != df.iloc[i-1]["emotion"]:
            transitions.append({
                "from": df.iloc[i-1]["emotion"],
                "to": df.iloc[i]["emotion"],
                "position": i
            })

    return transitions

# =========================
# PROCESS FILE
# =========================
def process_file(path, analyzer):
    text = clean_text(path.read_text(errors="ignore"))

    if len(tokenize(text)) < MIN_WORDS:
        return None

    windows = make_windows(text)
    df = compute_emotions(windows, analyzer)

    df["valence_smooth"] = smooth(df["valence"])
    df["arousal_smooth"] = smooth(df["arousal"])

    transitions = get_transitions(df)

    summary = {
        "file": path.name,
        "avg_valence": float(df["valence_smooth"].mean()),
        "avg_arousal": float(df["arousal_smooth"].mean()),
        "transitions": len(transitions)
    }

    return {
        "df": df,
        "summary": summary,
        "transitions": transitions
    }

# =========================
# MAIN
# =========================
def main():
    setup()
    analyzer = SentimentIntensityAnalyzer()

    summaries = []

    for file in INPUT_DIR.glob("*.txt"):
        logging.info(f"processing {file.name}")

        result = process_file(file, analyzer)
        if not result:
            continue

        summaries.append(result["summary"])

        # save per-script curve
        result["df"].to_json(OUTPUT_DIR / f"{file.stem}_curve.json", orient="records")

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "emotion_summary.csv", index=False)

    print("DONE")

if __name__ == "__main__":
    main()