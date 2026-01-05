import numpy as np

# Example similarity scores from Step 15 (replace with your real outputs later)
# Higher score = more similar
similarities = {
    "clean": 1.0,               # correct match
    "pitch_ps1": 0.72,
    "pitch_ps2": 0.65,
    "tempo_0.9": 0.85,
    "noise": 0.88,
    "reverb": 0.82,
    "volume": 0.90
}

# Sort by similarity (descending)
ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# --- Top-1 Accuracy ---
top1 = 1 if ranked[0][0] == "clean" else 0

# --- Precision@10 ---
# Only one relevant item: "clean"
relevant_found = 0
for i, (name, _) in enumerate(ranked[:10]):
    if name == "clean":
        relevant_found += 1

precision_at_10 = relevant_found / min(10, len(ranked))

# --- Mean Average Precision (MAP) ---
# Only one relevant item → AP = 1 / rank
for idx, (name, _) in enumerate(ranked):
    if name == "clean":
        ap = 1 / (idx + 1)
        break

# Print results
print("Ranking:", ranked)
print("Top-1 Accuracy:", top1)
print("Precision@10:", precision_at_10)
print("MAP:", ap)