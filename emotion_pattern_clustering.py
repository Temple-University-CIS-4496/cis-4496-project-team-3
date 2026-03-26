import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_DIR = Path(r"G:\.shortcut-targets-by-id\1zMZhOhWAJ4OREe9g3XIkRb34qNHbX6bw\data sci capstone files\processed_scripts\cleaned_scripts\sorted_by_duplicates")

OUTPUT_DIR = INPUT_DIR

# =========================
# LOAD EMOTION CURVES
# =========================
def load_curves():
    sequences = []
    names = []

    for file in INPUT_DIR.glob("*_curve.json"):
        df = pd.read_json(file)

        max_len = 200

        seq = np.vstack([
            df["valence_smooth"],
            df["arousal_smooth"]
        ]).T

        # pad or trim
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            pad = np.zeros((max_len - len(seq), 2))
            seq = np.vstack([seq, pad])

        sequences.append(seq.flatten())
        names.append(file.stem)

    return names, sequences

# =========================
# MAIN
# =========================
def main():
    names, sequences = load_curves()

    X = np.array(sequences)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k = 4
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    rows = []
    for name, label in zip(names, labels):
        rows.append({
            "script": name,
            "cluster": int(label)
        })

    pd.DataFrame(rows).to_csv(INPUT_DIR / "emotional_clusters.csv", index=False)

    print("CLUSTERING DONE")

if __name__ == "__main__":
    main()