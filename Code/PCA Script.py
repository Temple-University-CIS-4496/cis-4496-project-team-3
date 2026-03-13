import json
import logging
import re
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pypdf import PdfReader
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# config
INPUT_DIR = Path(r"C:\Data\Downloads\pdfs\Joinable")
OUTPUT_DIR = INPUT_DIR / "_spectral_output"
NUM_SEGMENTS = 20
MIN_WORDS_PER_SCRIPT = 400
K_MIN = 2
K_MAX = 8
RANDOM_STATE = 42


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_vader():
    logging.info("checking vader sentiment lexicon")
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        logging.info("vader lexicon already present")
    except LookupError:
        logging.warning("vader lexicon not found, downloading now")
        nltk.download("vader_lexicon", quiet=False)
        logging.info("vader lexicon download complete")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("output directory ready: %s", OUTPUT_DIR)


def find_pdfs(input_dir: Path):
    logging.info("scanning for pdf files in: %s", input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))
    logging.info("found %d pdf file(s)", len(pdfs))
    for i, pdf in enumerate(pdfs, start=1):
        logging.debug("pdf %d: %s", i, pdf.name)
    return pdfs


def clean_text(text: str):
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(pdf_path: Path):
    logging.info("reading pdf: %s", pdf_path.name)
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    logging.debug("[%s] total pages: %d", pdf_path.name, page_count)

    page_texts = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            logging.debug(
                "[%s] page %d/%d extracted chars: %d",
                pdf_path.name,
                page_idx,
                page_count,
                len(text),
            )
            page_texts.append(text)
        except Exception as exc:
            logging.exception(
                "[%s] failed to extract page %d: %s",
                pdf_path.name,
                page_idx,
                exc,
            )
            page_texts.append("")

    raw_text = "\n".join(page_texts)
    cleaned = clean_text(raw_text)
    logging.info(
        "[%s] extraction complete, raw chars=%d, cleaned chars=%d",
        pdf_path.name,
        len(raw_text),
        len(cleaned),
    )
    return cleaned, page_count


def tokenize_words(text: str):
    return re.findall(r"\b[\w']+\b", text)


def split_sentences(text: str):
    flat = re.sub(r"\s+", " ", text.strip())
    if not flat:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def extract_year_from_name(name: str):
    match = re.search(r"(19\d{2}|20\d{2})", name)
    if match:
        return int(match.group(1))
    return None


def is_scene_heading(line: str):
    upper = line.upper()
    if len(upper) > 120:
        return False
    return bool(
        re.match(
            r"^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.|I/E\.|EST\.|INT -|EXT -)",
            upper,
        )
    )


def is_character_cue(line: str):
    stripped = line.strip()
    if not stripped:
        return False
    if is_scene_heading(stripped):
        return False
    if len(stripped) > 35:
        return False
    if re.search(r"[a-z]", stripped):
        return False
    words = stripped.replace(".", " ").replace("-", " ").split()
    if len(words) == 0 or len(words) > 5:
        return False
    alpha_chars = re.findall(r"[A-Z]", stripped)
    return len(alpha_chars) >= 2


def count_dialogue_lines(lines):
    dialogue_lines = 0
    cue_count = 0
    in_dialogue = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            in_dialogue = False
            continue

        if is_scene_heading(stripped):
            in_dialogue = False
            continue

        if is_character_cue(stripped):
            cue_count += 1
            in_dialogue = True
            continue

        if in_dialogue:
            dialogue_lines += 1

    return dialogue_lines, cue_count


def safe_div(a, b):
    return a / b if b else 0.0


def split_into_segments(text: str, num_segments: int):
    lines = text.splitlines()
    if not lines:
        return [""] * num_segments

    chunks = np.array_split(np.array(lines, dtype=object), num_segments)
    segments = ["\n".join(chunk.tolist()).strip() for chunk in chunks]

    logging.debug("split script into %d segment(s)", len(segments))
    for i, seg in enumerate(segments, start=1):
        logging.debug(
            "segment %02d chars=%d words=%d",
            i,
            len(seg),
            len(tokenize_words(seg)),
        )
    return segments


def segment_features(segment_text: str, analyzer: SentimentIntensityAnalyzer):
    lines = [line.strip() for line in segment_text.splitlines() if line.strip()]
    words = tokenize_words(segment_text)
    sentences = split_sentences(segment_text)

    total_lines = len(lines)
    total_words = len(words)
    total_sentences = len(sentences)

    scene_count = sum(is_scene_heading(line) for line in lines)
    dialogue_lines, cue_count = count_dialogue_lines(lines)
    uppercase_lines = sum(
        1 for line in lines if re.search(r"[A-Z]", line) and line == line.upper()
    )

    avg_sentence_words = 0.0
    if total_sentences:
        avg_sentence_words = float(
            np.mean([len(tokenize_words(sentence)) for sentence in sentences])
        )

    sentiment = analyzer.polarity_scores(segment_text[:12000] if segment_text else "")

    features = [
        sentiment["compound"],
        sentiment["pos"],
        sentiment["neg"],
        sentiment["neu"],
        safe_div(scene_count, total_lines),
        safe_div(dialogue_lines, total_lines),
        safe_div(cue_count, total_lines),
        safe_div(uppercase_lines, total_lines),
        safe_div(segment_text.count("?"), max(total_words, 1)),
        safe_div(segment_text.count("!"), max(total_words, 1)),
        avg_sentence_words,
        float(total_words),
    ]

    return {
        "total_lines": total_lines,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "scene_count": scene_count,
        "dialogue_lines": dialogue_lines,
        "cue_count": cue_count,
        "avg_sentence_words": avg_sentence_words,
        "sentiment_compound": sentiment["compound"],
        "vector": features,
    }


def global_script_metrics(text: str, page_count: int):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = tokenize_words(text)
    sentences = split_sentences(text)

    total_lines = len(lines)
    total_words = len(words)
    total_sentences = len(sentences)
    total_scenes = sum(is_scene_heading(line) for line in lines)
    dialogue_lines, cue_count = count_dialogue_lines(lines)

    avg_sentence_words = 0.0
    if total_sentences:
        avg_sentence_words = float(
            np.mean([len(tokenize_words(sentence)) for sentence in sentences])
        )

    metrics = {
        "pages": page_count,
        "lines": total_lines,
        "words": total_words,
        "sentences": total_sentences,
        "scenes": total_scenes,
        "dialogue_lines": dialogue_lines,
        "character_cues": cue_count,
        "dialogue_ratio": safe_div(dialogue_lines, total_lines),
        "scene_ratio": safe_div(total_scenes, total_lines),
        "avg_sentence_words": avg_sentence_words,
    }

    global_vector = [
        np.log1p(page_count),
        np.log1p(total_words),
        np.log1p(total_lines),
        safe_div(total_scenes * 1000.0, max(total_words, 1)),
        safe_div(dialogue_lines, total_lines),
        safe_div(cue_count, total_lines),
        avg_sentence_words,
    ]

    return metrics, global_vector


def build_feature_vector(pdf_path: Path, analyzer: SentimentIntensityAnalyzer):
    text, page_count = extract_text_from_pdf(pdf_path)

    if not text.strip():
        logging.warning("[%s] empty text after extraction", pdf_path.name)
        return None

    metrics, global_vector = global_script_metrics(text, page_count)
    logging.info(
        "[%s] global metrics: pages=%d words=%d lines=%d scenes=%d dialogue_ratio=%.4f",
        pdf_path.name,
        metrics["pages"],
        metrics["words"],
        metrics["lines"],
        metrics["scenes"],
        metrics["dialogue_ratio"],
    )

    if metrics["words"] < MIN_WORDS_PER_SCRIPT:
        logging.warning(
            "[%s] skipped because word count %d is below minimum %d",
            pdf_path.name,
            metrics["words"],
            MIN_WORDS_PER_SCRIPT,
        )
        return None

    segments = split_into_segments(text, NUM_SEGMENTS)

    segment_vectors = []
    for idx, segment_text in enumerate(segments, start=1):
        features = segment_features(segment_text, analyzer)
        segment_vectors.extend(features["vector"])
        logging.debug(
            "[%s] segment %02d summary: words=%d scenes=%d dialogue_lines=%d sentiment=%.4f",
            pdf_path.name,
            idx,
            features["total_words"],
            features["scene_count"],
            features["dialogue_lines"],
            features["sentiment_compound"],
        )

    full_vector = np.array(segment_vectors + global_vector, dtype=float)

    result = {
        "file_name": pdf_path.name,
        "file_stem": pdf_path.stem,
        "year": extract_year_from_name(pdf_path.stem),
        "feature_vector": full_vector,
        **metrics,
    }

    logging.info(
        "[%s] feature vector built, length=%d, extracted_year=%s",
        pdf_path.name,
        len(full_vector),
        result["year"],
    )
    return result


def choose_best_k(x_scaled: np.ndarray):
    n_samples = x_scaled.shape[0]
    if n_samples < 3:
        raise ValueError("need at least 3 usable scripts for clustering")

    max_k = min(K_MAX, n_samples - 1)
    if max_k < K_MIN:
        return 2

    n_neighbors = max(2, min(10, n_samples - 1))
    logging.info(
        "selecting cluster count, candidate range=%d..%d, n_neighbors=%d",
        K_MIN,
        max_k,
        n_neighbors,
    )

    scores = []
    for k in range(K_MIN, max_k + 1):
        try:
            model = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                assign_labels="kmeans",
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(x_scaled)
            unique = np.unique(labels)

            if len(unique) < 2:
                logging.warning("k=%d produced only one cluster, skipping", k)
                continue

            score = silhouette_score(x_scaled, labels)
            scores.append((k, score))
            logging.info("k=%d silhouette=%.6f", k, score)
        except Exception as exc:
            logging.exception("k=%d failed during evaluation: %s", k, exc)

    if not scores:
        fallback_k = min(4, max_k)
        logging.warning("all silhouette evaluations failed, fallback k=%d", fallback_k)
        return fallback_k

    best_k, best_score = max(scores, key=lambda item: item[1])
    logging.info("selected k=%d with best silhouette=%.6f", best_k, best_score)
    return best_k


def representative_script_indices(x_scaled: np.ndarray, labels: np.ndarray):
    reps = {}
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        cluster_vectors = x_scaled[idx]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        best_local = int(np.argmin(distances))
        reps[int(cluster_id)] = int(idx[best_local])
    return reps


def save_cluster_plot(df: pd.DataFrame):
    if len(df) < 2:
        logging.warning("not enough rows for pca plot")
        return

    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(df["cluster"].unique()):
        chunk = df[df["cluster"] == cluster_id]
        plt.scatter(chunk["pca_1"], chunk["pca_2"], label=f"cluster {cluster_id}", s=60)

        for _, row in chunk.iterrows():
            plt.annotate(
                row["file_stem"][:25],
                (row["pca_1"], row["pca_2"]),
                fontsize=8,
                alpha=0.8,
            )

    plt.title("spectral clustering of movie scripts")
    plt.xlabel("pca 1")
    plt.ylabel("pca 2")
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "pca_clusters.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logging.info("saved pca cluster plot: %s", plot_path)


def save_year_trend_outputs(df: pd.DataFrame):
    usable = df.dropna(subset=["year"]).copy()
    if usable.empty:
        logging.warning("no years detected in filenames, skipping trend outputs")
        return

    usable["year"] = usable["year"].astype(int)
    trend = usable.groupby(["year", "cluster"]).size().unstack(fill_value=0).sort_index()

    trend_csv = OUTPUT_DIR / "cluster_year_counts.csv"
    trend.to_csv(trend_csv)
    logging.info("saved year trend table: %s", trend_csv)

    if len(trend.index) >= 2:
        plt.figure(figsize=(11, 7))
        for cluster_id in trend.columns:
            plt.plot(trend.index, trend[cluster_id], marker="o", label=f"cluster {cluster_id}")

        plt.title("cluster frequency over time")
        plt.xlabel("year")
        plt.ylabel("number of scripts")
        plt.legend()
        plt.tight_layout()

        trend_plot = OUTPUT_DIR / "cluster_year_trends.png"
        plt.savefig(trend_plot, dpi=180)
        plt.close()
        logging.info("saved year trend plot: %s", trend_plot)
    else:
        logging.warning("only one distinct year found, skipping trend plot")


def save_run_summary(summary: dict):
    path = OUTPUT_DIR / "run_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("saved run summary: %s", path)


def main():
    setup_logging()

    logging.info("starting script clustering pipeline")
    logging.info("input dir: %s", INPUT_DIR)
    logging.info("output dir: %s", OUTPUT_DIR)
    logging.info("num segments per script: %d", NUM_SEGMENTS)
    logging.info("minimum words per script: %d", MIN_WORDS_PER_SCRIPT)

    if not INPUT_DIR.exists():
        logging.error("input directory does not exist: %s", INPUT_DIR)
        sys.exit(1)

    ensure_output_dir()
    ensure_vader()

    analyzer = SentimentIntensityAnalyzer()
    pdfs = find_pdfs(INPUT_DIR)

    if not pdfs:
        logging.error("no pdf files found")
        sys.exit(1)

    records = []
    for i, pdf_path in enumerate(pdfs, start=1):
        logging.info("processing %d/%d: %s", i, len(pdfs), pdf_path.name)
        try:
            record = build_feature_vector(pdf_path, analyzer)
            if record is not None:
                records.append(record)
                logging.info("accepted: %s", pdf_path.name)
            else:
                logging.warning("rejected: %s", pdf_path.name)
        except Exception as exc:
            logging.error("failed on %s: %s", pdf_path.name, exc)
            logging.debug("full traceback:\n%s", traceback.format_exc())

    logging.info("usable scripts after preprocessing: %d", len(records))

    if len(records) < 3:
        logging.error("need at least 3 usable scripts, got %d", len(records))
        sys.exit(1)

    feature_matrix = np.vstack([record["feature_vector"] for record in records])
    logging.info(
        "feature matrix shape: rows=%d cols=%d",
        feature_matrix.shape[0],
        feature_matrix.shape[1],
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_matrix)
    logging.info("feature scaling complete")

    best_k = choose_best_k(x_scaled)
    n_neighbors = max(2, min(10, len(records) - 1))

    logging.info(
        "running final spectral clustering with k=%d n_neighbors=%d",
        best_k,
        n_neighbors,
    )

    model = SpectralClustering(
        n_clusters=best_k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(x_scaled)
    logging.info("final clustering complete")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x_scaled)
    logging.info(
        "pca complete, explained variance ratio=%.4f, %.4f",
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
    )

    for record, label, coord in zip(records, labels, coords):
        record["cluster"] = int(label)
        record["pca_1"] = float(coord[0])
        record["pca_2"] = float(coord[1])
        record.pop("feature_vector", None)

    df = pd.DataFrame(records)
    df = df.sort_values(["cluster", "file_name"]).reset_index(drop=True)

    cluster_csv = OUTPUT_DIR / "script_clusters.csv"
    df.to_csv(cluster_csv, index=False)
    logging.info("saved per-script cluster table: %s", cluster_csv)

    reps = representative_script_indices(x_scaled, labels)
    cluster_summary_rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        chunk = df[df["cluster"] == cluster_id]
        rep_idx = reps[int(cluster_id)]
        rep_name = records[rep_idx]["file_name"]

        cluster_summary_rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(len(chunk)),
                "representative_script": rep_name,
                "avg_pages": float(chunk["pages"].mean()),
                "avg_words": float(chunk["words"].mean()),
                "avg_scenes": float(chunk["scenes"].mean()),
                "avg_dialogue_ratio": float(chunk["dialogue_ratio"].mean()),
                "avg_sentence_words": float(chunk["avg_sentence_words"].mean()),
            }
        )

    summary_df = pd.DataFrame(cluster_summary_rows).sort_values("cluster")
    summary_csv = OUTPUT_DIR / "cluster_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info("saved cluster summary: %s", summary_csv)

    save_cluster_plot(df)
    save_year_trend_outputs(df)

    run_summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "num_input_pdfs": len(pdfs),
        "num_usable_scripts": len(records),
        "num_segments": NUM_SEGMENTS,
        "min_words_per_script": MIN_WORDS_PER_SCRIPT,
        "selected_k": int(best_k),
        "n_neighbors": int(n_neighbors),
        "pca_explained_variance_ratio": [
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        ],
    }
    save_run_summary(run_summary)

    logging.info("run finished successfully")
    logging.info("cluster counts:")
    for _, row in summary_df.iterrows():
        logging.info(
            "cluster %d -> count=%d representative=%s avg_words=%.1f avg_scenes=%.1f",
            int(row["cluster"]),
            int(row["count"]),
            row["representative_script"],
            row["avg_words"],
            row["avg_scenes"],
        )

    print()
    print("=" * 80)
    print("done")
    print(f"input folder:  {INPUT_DIR}")
    print(f"output folder: {OUTPUT_DIR}")
    print(f"usable scripts: {len(records)}")
    print(f"selected clusters: {best_k}")
    print(f"per-script table: {cluster_csv}")
    print(f"cluster summary:  {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()