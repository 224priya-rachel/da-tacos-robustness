from pydub import AudioSegment
import os

INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"
OUTPUT_DIR = "../data/covers_distorted/reverb/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio = AudioSegment.from_wav(INPUT_AUDIO)

# Create a simple reverb by overlaying delayed, quieter copies
reverb = audio
delay_ms = 60        # small room effect
decay_db = -8        # how quiet the echo is

for i in range(1, 4):
    reverb = reverb.overlay(audio - (i * abs(decay_db)), position=i * delay_ms)

out_path = os.path.join(OUTPUT_DIR, "videoplayback_reverb.wav")
reverb.export(out_path, format="wav")

print("Reverb-added file created.")