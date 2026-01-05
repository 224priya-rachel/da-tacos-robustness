import librosa
import soundfile as sf
import os

INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"
OUTPUT_DIR = "../data/covers_distorted/pitch_shift/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

y, sr = librosa.load(INPUT_AUDIO, sr=None)

semitones = [1, -1, 2, -2]

for n in semitones:
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n)
    out_name = f"videoplayback_ps{n}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, y_shifted, sr)

print("Pitch-shifted files created.")