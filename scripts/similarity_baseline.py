import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def load_feature(path):
    feat = np.load(path)
    return np.mean(feat, axis=1).reshape(1, -1)  # time-averaged

# Load clean features
clean_chroma = load_feature("../features/clean/clean_chroma.npy")
clean_mfcc = load_feature("../features/clean/clean_mfcc.npy")
clean_cens = load_feature("../features/clean/clean_cens.npy")

print("=== Similarity to CLEAN version ===")

def compare(folder, label):
    for f in os.listdir(folder):
        if f.endswith("_chroma.npy"):
            base = f.replace("_chroma.npy", "")
            chroma = load_feature(os.path.join(folder, f))
            mfcc = load_feature(os.path.join(folder, base + "_mfcc.npy"))
            cens = load_feature(os.path.join(folder, base + "_cens.npy"))

            c_sim = cosine_similarity(clean_chroma, chroma)[0][0]
            m_sim = cosine_similarity(clean_mfcc, mfcc)[0][0]
            cen_sim = cosine_similarity(clean_cens, cens)[0][0]

            print(f"{label} | {base}")
            print(f"  Chroma similarity: {c_sim:.3f}")
            print(f"  MFCC similarity:   {m_sim:.3f}")
            print(f"  CENS similarity:   {cen_sim:.3f}")

# Compare all distortions
compare("../features/pitch", "PITCH")
compare("../features/tempo", "TEMPO")
compare("../features/noise", "NOISE")
compare("../features/reverb", "REVERB")
compare("../features/volume", "VOLUME")