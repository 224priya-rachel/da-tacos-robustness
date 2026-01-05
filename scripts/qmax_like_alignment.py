import os
import numpy as np
from scipy.spatial.distance import cdist

def load_chroma(path):
    return np.load(path)

def qmax_like_score(X, Y):
    """
    Simplified Qmax-style score:
    - Compute frame-to-frame cosine distance
    - Convert to similarity
    - Take maximum mean diagonal score
    """
    D = cdist(X.T, Y.T, metric="cosine")
    S = 1 - D  # similarity

    diag_scores = []
    for k in range(-min(S.shape)+1, min(S.shape)):
        diag = np.diagonal(S, offset=k)
        if len(diag) > 10:
            diag_scores.append(np.mean(diag))

    return max(diag_scores)

# Load clean chroma
clean = load_chroma("../features/clean/clean_chroma.npy")

print("=== Qmax-like alignment scores (Chroma) ===")

def compare(folder, label):
    for f in os.listdir(folder):
        if f.endswith("_chroma.npy"):
            distorted = load_chroma(f"{folder}/{f}")
            score = qmax_like_score(clean, distorted)
            print(f"{label} | {f.replace('_chroma.npy','')} : {score:.3f}")

compare("../features/pitch", "PITCH")
compare("../features/tempo", "TEMPO")
compare("../features/noise", "NOISE")
compare("../features/reverb", "REVERB")
compare("../features/volume", "VOLUME")