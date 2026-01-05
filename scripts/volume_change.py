from pydub import AudioSegment
import os

INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"
OUTPUT_DIR = "../data/covers_distorted/volume/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio = AudioSegment.from_wav(INPUT_AUDIO)

# Volume changes in dB
volume_changes = [-5, 5]  # quieter, louder

for v in volume_changes:
    changed = audio + v
    out_name = f"videoplayback_vol{v}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    changed.export(out_path, format="wav")

print("Volume-changed files created.")