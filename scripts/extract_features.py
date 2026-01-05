import librosa
import numpy as np
import os

# Input audio (start with clean cover)
INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"

# Output directory
OUTPUT_DIR = "../features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load audio
y, sr = librosa.load(INPUT_AUDIO, sr=None)

# -------- CHROMA --------
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# -------- MFCC --------
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# -------- CENS --------
cens = librosa.feature.chroma_cens(y=y, sr=sr)

# Save features
np.save(os.path.join(OUTPUT_DIR, "chroma.npy"), chroma)
np.save(os.path.join(OUTPUT_DIR, "mfcc.npy"), mfcc)
np.save(os.path.join(OUTPUT_DIR, "cens.npy"), cens)

print("Features extracted and saved.")
print("Chroma shape:", chroma.shape)
print("MFCC shape:", mfcc.shape)
print("CENS shape:", cens.shape)