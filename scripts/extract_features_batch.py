import librosa
import numpy as np
import os

def extract_and_save(audio_path, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=None)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    cens = librosa.feature.chroma_cens(y=y, sr=sr)

    np.save(os.path.join(out_dir, f"{prefix}_chroma.npy"), chroma)
    np.save(os.path.join(out_dir, f"{prefix}_mfcc.npy"), mfcc)
    np.save(os.path.join(out_dir, f"{prefix}_cens.npy"), cens)

    print(f"Features saved for {prefix}")

# ---------- CLEAN ----------
extract_and_save(
    "../data/covers_clean/videoplayback.wav",
    "../features/clean",
    "clean"
)

# ---------- PITCH ----------
pitch_dir = "../data/covers_distorted/pitch_shift/"
for f in os.listdir(pitch_dir):
    extract_and_save(
        os.path.join(pitch_dir, f),
        "../features/pitch",
        f.replace(".wav", "")
    )

# ---------- TEMPO ----------
tempo_dir = "../data/covers_distorted/tempo_change/"
for f in os.listdir(tempo_dir):
    extract_and_save(
        os.path.join(tempo_dir, f),
        "../features/tempo",
        f.replace(".wav", "")
    )

# ---------- NOISE ----------
noise_dir = "../data/covers_distorted/noise/"
for f in os.listdir(noise_dir):
    extract_and_save(
        os.path.join(noise_dir, f),
        "../features/noise",
        f.replace(".wav", "")
    )

# ---------- REVERB ----------
reverb_dir = "../data/covers_distorted/reverb/"
for f in os.listdir(reverb_dir):
    extract_and_save(
        os.path.join(reverb_dir, f),
        "../features/reverb",
        f.replace(".wav", "")
    )

# ---------- VOLUME ----------
volume_dir = "../data/covers_distorted/volume/"
for f in os.listdir(volume_dir):
    extract_and_save(
        os.path.join(volume_dir, f),
        "../features/volume",
        f.replace(".wav", "")
    )

print("All features extracted.")