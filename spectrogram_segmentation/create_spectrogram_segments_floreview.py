import os
import numpy as np
import re

# Parameters
DURATION = 860
SHIFT = 430
BASE_DIR = "./Floreview_mel_flat_44100/"
OUTPUT_DIR = "./Data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_device_id(dirname):
    match = re.match(r'D(\d+)', dirname)
    return int(match.group(1)) if match else None

def segment_spectrogram(spectrogram, duration=DURATION, shift=SHIFT):
    segments = []
    for start in range(0, spectrogram.shape[1] - duration + 1, shift):
        segment = spectrogram[:, start:start + duration]
        if segment.shape[1] == duration:
            segments.append(segment[np.newaxis, ...])
    return segments

X, y, sources = [], [], []

for device_folder in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, device_folder)
    if not os.path.isdir(folder_path):
        continue

    label = extract_device_id(device_folder)
    if label is None:
        continue

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            spectrogram = np.load(fpath)
            segments = segment_spectrogram(spectrogram)
            X.extend(segments)
            y.extend([label - 1] * len(segments))  # 0-based indexing
            rec_id = f"{device_folder}_{fname}"
            sources.extend([rec_id] * len(segments))
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

# Save
np.save(os.path.join(OUTPUT_DIR, "X.npy"), np.array(X))
np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(y))
np.save(os.path.join(OUTPUT_DIR, "sources.npy"), np.array(sources))
print("Saved /Data/ with shape:", np.array(X).shape)
