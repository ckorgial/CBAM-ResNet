import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.utils import shuffle

# === Paths and Settings ===
DATA_PATH = "./Data"
SAVE_DIR = "./Splits"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)

# === Load dataset ===
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))
sources = np.load(os.path.join(DATA_PATH, "sources.npy"))

# === Ensure labels are 0-based
if y.min() == 1 and y.max() == 35:
    y = y - 1

NUM_CLASSES = 35
split_ratios = {"train": 0.55, "val": 0.15, "test": 0.30}
print("✅ Dataset loaded")

# === Parse source info: (scene, device, recording)
def parse_source(src):
    parts = src.split('_')
    scene = parts[0]
    device = parts[1]
    recording = '_'.join(parts[2:]).replace('.npy', '')
    return scene, device, recording

recording_groups = defaultdict(list)
parsed_info = []

for idx, src in enumerate(sources):
    scene, device, recording = parse_source(src)
    dev_idx = int(device.replace('D', '')) - 1  # D01 → 0
    key = (dev_idx, scene, recording)
    recording_groups[key].append(idx)
    parsed_info.append((scene, dev_idx, recording))

print(f"✅ Found {len(recording_groups)} unique (device, scene, recording) groups")

# === Group recordings per device
device_to_recordings = defaultdict(list)
for (dev, scene, rec), indices in recording_groups.items():
    device_to_recordings[dev].append((scene, rec, indices))

# === Stratified disjoint split ensuring each class appears in all splits
splits = {'train': [], 'val': [], 'test': []}
split_labels = np.empty(len(sources), dtype=object)
split_labels[:] = "none"

for dev in range(NUM_CLASSES):
    recs = device_to_recordings[dev]
    recs = shuffle(recs, random_state=SEED)

    total = sum(len(indices) for _, _, indices in recs)
    target = {
        "train": int(total * split_ratios["train"]),
        "val": int(total * split_ratios["val"]),
        "test": total - int(total * split_ratios["train"]) - int(total * split_ratios["val"])
    }

    counts = {"train": 0, "val": 0, "test": 0}

    for scene, rec, indices in recs:
        assigned = False
        for split in ["train", "val", "test"]:
            if counts[split] + len(indices) <= target[split]:
                splits[split].extend(indices)
                split_labels[indices] = split
                counts[split] += len(indices)
                assigned = True
                break
        if not assigned:
            # assign to split with most remaining room
            remaining = {s: target[s] - counts[s] for s in ["train", "val", "test"]}
            best_split = max(remaining, key=remaining.get)
            splits[best_split].extend(indices)
            split_labels[indices] = best_split
            counts[best_split] += len(indices)

print("✅ Split complete")

# === Save split arrays
for split in ['train', 'val', 'test']:
    idxs = np.array(splits[split])
    np.save(os.path.join(SAVE_DIR, f"X_{split}.npy"), X[idxs])
    np.save(os.path.join(SAVE_DIR, f"y_{split}.npy"), y[idxs])
    print(f"✅ Saved: X_{split}.npy with {len(idxs)} samples")

# === Save per-scene-per-split arrays
unique_scenes = sorted(set([parse_source(s)[0] for s in sources]))
for split in ['train', 'val', 'test']:
    for scene in unique_scenes:
        idxs = [i for i in splits[split] if parse_source(sources[i])[0] == scene]
        np.save(os.path.join(SAVE_DIR, f"X_{scene}_{split}.npy"), X[idxs])
        np.save(os.path.join(SAVE_DIR, f"y_{scene}_{split}.npy"), y[idxs])
        print(f"✅ Saved: X_{scene}_{split}.npy with {len(idxs)} samples")

# === Save metadata
metadata = pd.DataFrame({
    "filename": sources,
    "device": [parse_source(s)[1] for s in sources],
    "scene": [parse_source(s)[0] for s in sources],
    "split": split_labels
})
metadata.to_csv(os.path.join(SAVE_DIR, "all_split_metadata_disjoint.csv"), index=False)
print("📄 Saved metadata CSV")

# === Check: Print unique class count per split
for split in ['train', 'val', 'test']:
    labels = y[np.array(splits[split])]
    unique_labels = np.unique(labels)
    print(f"🔎 {split.upper()} contains {len(unique_labels)} unique classes: {sorted(unique_labels)}")
