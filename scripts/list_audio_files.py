import os

BASE_DIR = "../data"

folders = {
    "clean": "covers_clean",
    "pitch": "covers_distorted/pitch_shift",
    "tempo": "covers_distorted/tempo_change",
    "noise": "covers_distorted/noise",
    "reverb": "covers_distorted/reverb",
    "volume": "covers_distorted/volume"
}

for label, path in folders.items():
    full_path = os.path.join(BASE_DIR, path)
    files = [f for f in os.listdir(full_path) if f.endswith(".wav")]
    print(f"\n{label.upper()} FILES ({len(files)}):")
    for f in files:
        print("  ", f)