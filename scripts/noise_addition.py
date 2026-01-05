from pydub import AudioSegment
from pydub.generators import WhiteNoise
import os

INPUT_AUDIO = "../data/covers_clean/videoplayback.wav"
OUTPUT_DIR = "../data/covers_distorted/noise/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio = AudioSegment.from_wav(INPUT_AUDIO)

# Light background noise
noise = WhiteNoise().to_audio_segment(
    duration=len(audio),
    volume=-20  # dB, realistic background noise
)

noisy_audio = audio.overlay(noise)

out_path = os.path.join(OUTPUT_DIR, "videoplayback_noise.wav")
noisy_audio.export(out_path, format="wav")

print("Noise-added file created.")