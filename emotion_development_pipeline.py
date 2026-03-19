import os
import re
import numpy as np
import fitz
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_FOLDER = "cleaned_scripts"
OUTPUT_EMBEDDINGS = "emotion_development_embeddings.npy"
OUTPUT_CLUSTERS = "emotion_development_clusters.txt"

WINDOW_SIZE = 200
STEP_SIZE = 100

EMOTION_WORDS = {
    "joy": ["happy", "joy", "smile", "laugh"],
    "sadness": ["sad", "cry", "tear", "grief"],
    "anger": ["angry", "rage", "furious"],
    "fear": ["fear", "scared", "terrified"],
    "surprise": ["surprised", "shock"],
}

def emotion_vector(text):
    text = text.lower()
    vec = []
    for emo in EMOTION_WORDS:
        count = sum(text.count(w) for w in EMOTION_WORDS[emo])
        vec.append(count)
    return np.array(vec)

def get_windows(text):
    words = text.split()
    windows = []
    for i in range(0, len(words) - WINDOW_SIZE, STEP_SIZE):
        chunk = " ".join(words[i:i+WINDOW_SIZE])
        windows.append(chunk)
    return windows

def process_script(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    windows = get_windows(text)
    features = []

    for w in windows:
        vec = emotion_vector(w)
        features.append(vec)

    if len(features) == 0:
        return None

    features = np.array(features)

    # emotional development = mean + variance + change
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    diffs = np.diff(features, axis=0)
    volatility = np.mean(np.abs(diffs)) if len(diffs) > 0 else 0

    return np.concatenate([mean, std, [volatility]])

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

    embeddings = []
    names = []

    for f in files:
        path = os.path.join(INPUT_FOLDER, f)
        emb = process_script(path)
        if emb is not None:
            embeddings.append(emb)
            names.append(f)

    embeddings = np.array(embeddings)

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=5, random_state=0)
    labels = kmeans.fit_predict(X)

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    with open(OUTPUT_CLUSTERS, "w") as f:
        for name, label in zip(names, labels):
            f.write(f"{name} -> cluster {label}\n")

    print("done")

if __name__ == "__main__":
    main()