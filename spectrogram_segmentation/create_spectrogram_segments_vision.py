import os
import numpy as np
import re

# Parameters
DURATION = 860
SHIFT = 430
BASE_DIR = "./VISION_mel_all_44100/"
SCENES = ['flat', 'indoor', 'outdoor']
OUTPUT_DIR = "./Data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_device_id(dirname):
    match = re.match(r'D(\d+)_', dirname)
    return int(match.group(1)) if match else None

def segment_spectrogram(spectrogram, duration=DURATION, shift=SHIFT):
    segments = []
    for start in range(0, spectrogram.shape[1] - duration + 1, shift):
        segment = spectrogram[:, start:start + duration]
        if segment.shape[1] == duration:
            segments.append(segment[np.newaxis, ...])
    return segments

def process_scene(scene):
    X_scene, y_scene, sources_scene = [], [], []
    for device_folder in sorted(os.listdir(BASE_DIR)):
        label = extract_device_id(device_folder)
        if label is None:
            continue
        scene_path = os.path.join(BASE_DIR, device_folder, scene)
        if not os.path.isdir(scene_path):
            continue
        for fname in sorted(os.listdir(scene_path)):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(scene_path, fname)
            try:
                spectrogram = np.load(fpath)
                segments = segment_spectrogram(spectrogram)
                X_scene.extend(segments)
                y_scene.extend([label] * len(segments))
                rec_id = f"{scene}_{device_folder}_{fname}"
                sources_scene.extend([rec_id] * len(segments))
            except Exception as e:
                print(f"Error: {fpath} - {e}")
    return np.array(X_scene), np.array(y_scene), np.array(sources_scene)

# Process all scenes
X_total, y_total, sources_total = [], [], []
for scene in SCENES:
    X_scene, y_scene, sources_scene = process_scene(scene)
    X_total.append(X_scene)
    y_total.append(y_scene)
    sources_total.append(sources_scene)

# Save combined
X_all = np.concatenate(X_total)
y_all = np.concatenate(y_total)
sources_all = np.concatenate(sources_total)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_all)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_all)
np.save(os.path.join(OUTPUT_DIR, "sources.npy"), sources_all)

print("Saved full dataset with shape:", X_all.shape)

import numpy as np
X = np.load('./Data/X.npy')
y = np.load('./Data/y.npy')
print(X.shape, y.shape, "min label:", y.min(), "max label:", y.max())
print("Unique labels:", np.unique(y, return_counts=True))

