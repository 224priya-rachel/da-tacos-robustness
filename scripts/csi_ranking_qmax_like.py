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

# Load all clean chroma features
clean_dir = os.path.join(FEATURES_DIR, "clean")
clean_files = [f for f in os.listdir(clean_dir) if f.endswith("_chroma.npy")]

clean_features = {}
for f in clean_files:
    clean_features[f.replace("_chroma.npy", "")] = load_chroma(
        os.path.join(clean_dir, f)
    )

# Function to rank one query
def rank_query(query_chroma, clean_features):
    scores = {}
    for name, feat in clean_features.items():
        scores[name] = qmax_like_score(query_chroma, feat)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# Run ranking for each distortion type
distortions = ["pitch", "tempo", "noise", "reverb", "volume"]

for dist in distortions:
    print(f"\n=== {dist.upper()} QUERIES ===")
    query_dir = os.path.join(FEATURES_DIR, dist)

    for f in os.listdir(query_dir):
        if f.endswith("_chroma.npy"):
            query_name = f.replace("_chroma.npy", "")
            query_feat = load_chroma(os.path.join(query_dir, f))

            ranking = rank_query(query_feat, clean_features)

            print(f"\nQuery: {query_name}")
            for rank, (name, score) in enumerate(ranking[:5], start=1):
                print(f"  Rank {rank}: {name} (score={score:.3f})")