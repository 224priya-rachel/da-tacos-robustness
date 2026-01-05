import librosa
import soundfile as sf
import os

INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"
OUTPUT_DIR = "../data/covers_distorted/tempo_change/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

y, sr = librosa.load(INPUT_AUDIO, sr=None)

tempo_factors = [0.9, 1.1]  # slower, faster

for t in tempo_factors:
    y_tempo = librosa.effects.time_stretch(y, rate=t)
    out_name = f"videoplayback_tempo{t}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, y_tempo, sr)

print("Tempo-changed files created.")