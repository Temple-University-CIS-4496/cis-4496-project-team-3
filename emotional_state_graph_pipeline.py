import os
import re
import networkx as nx
import numpy as np

INPUT_FOLDER = "cleaned_scripts"

EMOTION_WORDS = {
    "joy": ["happy", "joy", "smile", "laugh"],
    "sadness": ["sad", "cry", "tear", "grief"],
    "anger": ["angry", "rage", "furious"],
    "fear": ["fear", "scared", "terrified"],
}

def detect_emotion(sentence):
    sentence = sentence.lower()
    scores = {}

    for emo in EMOTION_WORDS:
        scores[emo] = sum(sentence.count(w) for w in EMOTION_WORDS[emo])

    if sum(scores.values()) == 0:
        return "neutral", 0

    emo = max(scores, key=scores.get)
    return emo, scores[emo]

def split_sentences(text):
    return re.split(r"[.!?]", text)

def build_graph(text):
    sentences = split_sentences(text)

    G = nx.DiGraph()

    prev_node = None

    for i, s in enumerate(sentences):
        emo, score = detect_emotion(s)

        node_id = f"{i}"
        G.add_node(node_id, emotion=emo, intensity=score)

        if prev_node is not None:
            G.add_edge(prev_node, node_id, type="temporal")

            # emotion shift
            prev_emo = G.nodes[prev_node]["emotion"]
            if prev_emo != emo:
                G.add_edge(prev_node, node_id, type="emotion_shift")

        prev_node = node_id

    return G

def graph_features(G):
    emotions = nx.get_node_attributes(G, "emotion")
    counts = {}

    for e in emotions.values():
        counts[e] = counts.get(e, 0) + 1

    total = sum(counts.values())
    vec = [counts.get(e, 0)/total for e in ["joy", "sadness", "anger", "fear", "neutral"]]

    return np.array(vec)

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

    all_features = []

    for f in files:
        path = os.path.join(INPUT_FOLDER, f)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        G = build_graph(text)
        feat = graph_features(G)

        all_features.append(feat)

    print("processed graphs:", len(all_features))

if __name__ == "__main__":
    main()