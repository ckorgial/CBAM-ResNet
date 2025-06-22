import os
import numpy as np
from tqdm import tqdm

# Parameters
DURATION = 860
SHIFT = 430
BASE_DIR = "./POLIPHONE_mel_44100"
OUTPUT_DIR = "./Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Devices = subfolders in BASE_DIR
device_folders = sorted(os.listdir(BASE_DIR))
device_to_label = {dev: idx for idx, dev in enumerate(device_folders)}

def segment_spectrogram(spec, duration=DURATION, shift=SHIFT):
    return [spec[:, i:i + duration][np.newaxis, ...]
            for i in range(0, spec.shape[1] - duration + 1, shift)
            if spec[:, i:i + duration].shape[1] == duration]

X, y, sources = [], [], []

for device in tqdm(device_folders, desc="Devices"):
    label = device_to_label[device]
    mel_dir = os.path.join(BASE_DIR, device)
    if not os.path.isdir(mel_dir): continue

    for fname in sorted(os.listdir(mel_dir)):
        if not fname.endswith("_mel.npy"): continue
        path = os.path.join(mel_dir, fname)
        try:
            logmel = np.load(path)
            segments = segment_spectrogram(logmel)
            if not segments:
                continue
            X.extend(segments)
            y.extend([label] * len(segments))
            sources.extend([f"{device}_{fname}"] * len(segments))
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")

if X:
    X = np.concatenate(X)
    y = np.array(y)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "sources.npy"), np.array(sources))
    print("✅ Saved:", X.shape, y.shape)
else:
    print("❌ No valid spectrograms were found. Please check the mel directory structure.")
