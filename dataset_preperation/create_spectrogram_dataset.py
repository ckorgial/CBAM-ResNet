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

# === Ensure labels are 0-based ===
if y.min() == 1 and y.max() == 35:
    y = y - 1

# === Parse source info ===
def parse_source(src):
    parts = src.split('_')
    scene = parts[0]
    device = parts[1]
    recording = '_'.join(parts[2:]).replace('.npy', '')
    return scene, device, recording

source_info = [parse_source(src) for src in sources]

# === Create groups by (scene, device, recording) ===
indices_per_group = defaultdict(list)
for idx, (scene, device, recording) in enumerate(source_info):
    indices_per_group[(scene, device, recording)].append(idx)

# === Shuffle groups and split ===
splits = {'train': [], 'val': [], 'test': []}
groups = list(indices_per_group.items())
groups = shuffle(groups, random_state=SEED)

# Calculate exact counts based on desired split ratios
num_total = len(X)
num_train = int(num_total * 0.55)
num_val = int(num_total * 0.15)
num_test = num_total - num_train - num_val

split_limits = {'train': num_train, 'val': num_val, 'test': num_test}
split_counts = {'train': 0, 'val': 0, 'test': 0}

for (scene, device, recording), indices in groups:
    chosen_split = None
    for split in ['train', 'val', 'test']:
        if split_counts[split] + len(indices) <= split_limits[split]:
            splits[split].extend(indices)
            split_counts[split] += len(indices)
            chosen_split = split
            break
    if chosen_split is None:
        remaining_splits = {k: v - split_counts[k] for k, v in split_limits.items()}
        best_split = max(remaining_splits, key=remaining_splits.get)
        splits[best_split].extend(indices)
        split_counts[best_split] += len(indices)

# === Save split arrays ===
for split in ['train', 'val', 'test']:
    idxs = np.array(splits[split])
    np.save(os.path.join(SAVE_DIR, f"X_{split}.npy"), X[idxs])
    np.save(os.path.join(SAVE_DIR, f"y_{split}.npy"), y[idxs])
    print(f"✅ Saved: X_{split}.npy with {len(idxs)} samples")

# === Save per-scene-per-split arrays ===
splits_scene = defaultdict(lambda: defaultdict(list))
for split in ['train', 'val', 'test']:
    for idx in splits[split]:
        scene = source_info[idx][0]
        splits_scene[scene][split].append(idx)

for scene in splits_scene:
    for split in ['train', 'val', 'test']:
        idxs = np.array(splits_scene[scene][split])
        np.save(os.path.join(SAVE_DIR, f"X_{scene}_{split}.npy"), X[idxs])
        np.save(os.path.join(SAVE_DIR, f"y_{scene}_{split}.npy"), y[idxs])
        print(f"✅ Saved: X_{scene}_{split}.npy with {len(idxs)} samples")

# === Save metadata ===
exact_split_labels = np.empty(len(sources), dtype=object)
exact_split_labels[:] = "none"
for split, indices in splits.items():
    exact_split_labels[indices] = split

exact_metadata_final = pd.DataFrame({
    "filename": sources,
    "device": [info[1] for info in source_info],
    "scene": [info[0] for info in source_info],
    "split": exact_split_labels
})
exact_metadata_final.to_csv(os.path.join(SAVE_DIR, "exact_split_metadata.csv"), index=False)
print("📄 Saved exact metadata CSV")
