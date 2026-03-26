import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample, find_peaks
from scipy.fftpack import dct

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PATHS
# =========================
OUTPUT_DIR = Path(r"G:\.shortcut-targets-by-id\1zMZhOhWAJ4OREe9g3XIkRb34qNHbX6bw\data sci capstone files\processed_scripts\emotion_analysis_final")

CURVE_DIR = OUTPUT_DIR / "curves"
PLOT_DIR = OUTPUT_DIR / "plots"
VECTOR_FILE = OUTPUT_DIR / "all_vectors.csv"

# =========================
# SETUP
# =========================
def setup():
    logging.basicConfig(level=logging.INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

# =========================
# FEATURE ENGINEERING
# =========================
def resample_curve(v, a, target_len=100):
    return resample(v, target_len), resample(a, target_len)

def basic_stats(x):
    return [np.mean(x), np.std(x), np.min(x), np.max(x)]

def slope_features(x):
    return [x[-1] - x[0], np.mean(np.diff(x))]

def peak_features(x):
    peaks, _ = find_peaks(x)
    valleys, _ = find_peaks(-x)

    return [
        len(peaks),
        len(valleys),
        np.mean(x[peaks]) if len(peaks) > 0 else 0,
        np.mean(x[valleys]) if len(valleys) > 0 else 0
    ]

def area_feature(x):
    from numpy import trapezoid
    return [trapezoid(x)]

def dct_features(x, k=20):
    return list(dct(x, norm="ortho")[:k])

def build_feature_vector(valence, arousal):
    v_res, a_res = resample_curve(valence, arousal)

    features = []
    for signal in [v_res, a_res]:
        features += basic_stats(signal)
        features += slope_features(signal)
        features += peak_features(signal)
        features += area_feature(signal)
        features += dct_features(signal, k=20)

    return np.array(features)

# =========================
# LOAD EXISTING VECTORS
# =========================
def load_existing_vectors():
    if not VECTOR_FILE.exists():
        return {}

    df = pd.read_csv(VECTOR_FILE)
    names = df["script"].tolist()
    vectors = df.drop(columns=["script"]).values

    return dict(zip(names, vectors))

# =========================
# SAVE ALL VECTORS
# =========================
def save_all_vectors(vector_dict):
    rows = []
    for name, vec in vector_dict.items():
        row = {"script": name}
        for i, val in enumerate(vec):
            row[f"f_{i}"] = val
        rows.append(row)

    pd.DataFrame(rows).to_csv(VECTOR_FILE, index=False)

# =========================
# MAIN ANALYSIS
# =========================
def run_analysis():
    vector_dict = load_existing_vectors()

    new_count = 0

    for file in CURVE_DIR.glob("*.json"):
        if file.stem in vector_dict:
            continue

        print(f"processing {file.stem}")

        df = pd.read_json(file)

        valence = df["valence"].values
        arousal = df["arousal"].values

        vec = build_feature_vector(valence, arousal)

        vector_dict[file.stem] = vec
        new_count += 1

    if new_count > 0:
        print(f"Added {new_count} new scripts.")
        save_all_vectors(vector_dict)
    else:
        print("No new scripts. Using existing vectors.")

    if len(vector_dict) == 0:
        print("No data available.")
        return

    # =========================
    # FULL SIMILARITY (ALWAYS RUNS)
    # =========================
    names = list(vector_dict.keys())
    X = np.vstack(list(vector_dict.values()))

    X_scaled = StandardScaler().fit_transform(X)

    sim_matrix = cosine_similarity(X_scaled)

    pd.DataFrame(sim_matrix, index=names, columns=names)\
        .to_csv(OUTPUT_DIR / "cosine_similarity_FULL.csv")

    print("FULL similarity computed.")

    # =========================
    # CLUSTERING
    # =========================
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    pd.DataFrame({
        "script": names,
        "cluster": labels
    }).to_csv(OUTPUT_DIR / "clusters_FULL.csv", index=False)

    # =========================
    # PCA PLOT
    # =========================
    coords = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure()
    for i in range(4):
        idx = labels == i
        plt.scatter(coords[idx, 0], coords[idx, 1], label=f"cluster {i}")

    plt.legend()
    plt.title("Emotional Clusters (FULL)")
    plt.savefig(PLOT_DIR / "clusters_FULL.png")
    plt.close()

# =========================
# MAIN
# =========================
def main():
    setup()

    print("VECTOR + SIMILARITY (FINAL VERSION)")
    run_analysis()

    print("DONE")

if __name__ == "__main__":
    main()