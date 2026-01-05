import os
import numpy as np
from scipy.spatial.distance import cdist

FEATURES_DIR = "../features"

def load_chroma(path):
    return np.load(path)

def qmax_like_score(X, Y):
    D = cdist(X.T, Y.T, metric="cosine")
    S = 1 - D

    scores = []
    for k in range(-min(S.shape) + 1, min(S.shape)):
        diag = np.diagonal(S, offset=k)
        if len(diag) > 10:
            scores.append(np.mean(diag))

    return max(scores)

# Load clean database
clean_dir = os.path.join(FEATURES_DIR, "clean")
clean_files = [f for f in os.listdir(clean_dir) if f.endswith("_chroma.npy")]

clean_features = {
    f.replace("_chroma.npy", ""): load_chroma(os.path.join(clean_dir, f))
    for f in clean_files
}

def evaluate_distortion(distortion):
    query_dir = os.path.join(FEATURES_DIR, distortion)
    queries = [f for f in os.listdir(query_dir) if f.endswith("_chroma.npy")]

    ranks = []

    for q in queries:
        query_name = q.replace("_chroma.npy", "")
        true_match = query_name.split("_")[0]  # videoplayback_ps1 -> videoplayback

        query_feat = load_chroma(os.path.join(query_dir, q))

        scores = {}
        for name, feat in clean_features.items():
            scores[name] = qmax_like_score(query_feat, feat)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_names = [name for name, _ in ranked]

        rank = ranked_names.index(true_match) + 1
        ranks.append(rank)

    # Metrics
    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    p10 = sum(1 for r in ranks if r <= 10) / (10 * len(ranks))
    map_score = np.mean([1 / r for r in ranks])

    return top1, p10, map_score

distortions = ["pitch", "tempo", "noise", "reverb", "volume"]

for d in distortions:
    top1, p10, map_score = evaluate_distortion(d)
    print(f"\n{d.upper()}")
    print(f"  Top-1 Accuracy: {top1:.3f}")
    print(f"  Precision@10:   {p10:.3f}")
    print(f"  MAP:            {map_score:.3f}")