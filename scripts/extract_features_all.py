import librosa
import numpy as np
import os

DATA_DIR = "../data"
FEATURE_DIR = "../features"

folders = {
    "clean": "covers_clean",
    "pitch": "covers_distorted/pitch_shift",
    "tempo": "covers_distorted/tempo_change",
    "noise": "covers_distorted/noise",
    "reverb": "covers_distorted/reverb",
    "volume": "covers_distorted/volume"
}

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    return chroma, mfcc, cens

for label, folder in folders.items():
    audio_dir = os.path.join(DATA_DIR, folder)
    out_dir = os.path.join(FEATURE_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for f in files:
        audio_path = os.path.join(audio_dir, f)
        chroma, mfcc, cens = extract_features(audio_path)

        base = f.replace(".wav", "")
        np.save(os.path.join(out_dir, base + "_chroma.npy"), chroma)
        np.save(os.path.join(out_dir, base + "_mfcc.npy"), mfcc)
        np.save(os.path.join(out_dir, base + "_cens.npy"), cens)

        print(f"[{label}] features extracted for {f}")

print("All features extracted for all files.")