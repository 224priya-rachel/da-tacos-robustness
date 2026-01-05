import matplotlib.pyplot as plt

# Results from Step 20 (replace values if needed)
distortions = ["Pitch", "Tempo", "Noise", "Reverb", "Volume"]

top1 = [0.50, 1.00, 1.00, 1.00, 1.00]
map_scores = [0.58, 1.00, 0.92, 0.85, 0.97]
p10 = [0.10, 0.10, 0.10, 0.10, 0.10]

# ---- Top-1 Accuracy ----
plt.figure()
plt.bar(distortions, top1)
plt.title("Top-1 Accuracy under Distortions")
plt.ylabel("Top-1 Accuracy")
plt.ylim(0, 1)
plt.show()

# ---- MAP ----
plt.figure()
plt.bar(distortions, map_scores)
plt.title("MAP under Distortions")
plt.ylabel("Mean Average Precision")
plt.ylim(0, 1)
plt.show()

# ---- Precision@10 ----
plt.figure()
plt.bar(distortions, p10)
plt.title("Precision@10 under Distortions")
plt.ylabel("Precision@10")
plt.ylim(0, 1)
plt.show()