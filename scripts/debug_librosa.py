import librosa
import inspect

print("Librosa version:", librosa.__version__)
print("pitch_shift signature:")
print(inspect.signature(librosa.effects.pitch_shift))