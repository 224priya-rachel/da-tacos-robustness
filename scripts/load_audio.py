import librosa

audio_path = "../data/covers_clean/videoplayback.wav"

y, sr = librosa.load(audio_path, sr=None)

duration = librosa.get_duration(y=y, sr=sr)

print("Sample rate:", sr)
print("Duration (seconds):", duration)
print("Number of samples:", len(y))