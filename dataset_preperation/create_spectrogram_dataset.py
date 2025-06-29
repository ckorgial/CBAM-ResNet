import os
import numpy as np
import pandas as pd
from collections import defaultdict

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

if y.min() == 1 and y.max() == 35:
    y = y - 1

# === Define target ratios and exact counts ===
total = len(sources)
target_counts = {
    "train": int(total * 0.55),
    "val": int(total * 0.15),
    "test": total - int(total * 0.55) - int(total * 0.15),
}
print("✅ Target counts:", target_counts)

# === Parse source and group by recording ===
def parse_source(src):
    parts = src.split('_')
    scene = parts[0]
    device = parts[1]
    recording = '_'.join(parts[2:]).replace('.npy', '')
    return scene, device, recording

recording_to_indices = defaultdict(list)
scenes = []

for idx, src in enumerate(sources):
    scene, device, recording = parse_source(src)
    recording_to_indices[(device, scene, recording)].append(idx)
    scenes.append(scene)

unique_scenes = sorted(set(scenes))

# === Build per-scene recording groups ===
scene_recordings = defaultdict(list)
scene_counts = {scene: 0 for scene in unique_scenes}
for rec_key, indices in recording_to_indices.items():
    scene = rec_key[1]
    scene_recordings[scene].append((rec_key, indices))
    scene_counts[scene] += len(indices)

# === Compute per-scene target counts ===
scene_targets = {
    scene: {
        split: int(target_counts[split] * (scene_counts[scene] / sum(scene_counts.values())))
        for split in ["train", "val", "test"]
    }
    for scene in unique_scenes
}

# === Initialize splits ===
splits = {'train': [], 'val': [], 'test': []}
split_labels = np.empty(len(sources), dtype=object)
split_labels[:] = "none"
counters = {'train': 0, 'val': 0, 'test': 0}

# === Allocate recordings per scene ===
for scene in unique_scenes:
    recs = scene_recordings[scene]
    recs.sort(key=lambda item: len(item[1]), reverse=True)
    local_counts = {'train': 0, 'val': 0, 'test': 0}
    local_target = scene_targets[scene]

    for rec_key, indices in recs:
        rec_len = len(indices)
        assigned = False
        for split in ['train', 'val', 'test']:
            if local_counts[split] + rec_len <= local_target[split]:
                splits[split].extend(indices)
                split_labels[indices] = split
                local_counts[split] += rec_len
                counters[split] += rec_len
                assigned = True
                break
        if not assigned:
            remain = {k: local_target[k] - local_counts[k] for k in ['train', 'val', 'test']}
            best_split = max(remain, key=remain.get)
            splits[best_split].extend(indices)
            split_labels[indices] = best_split
            local_counts[best_split] += rec_len
            counters[best_split] += rec_len

# === Save split arrays ===
for split in ['train', 'val', 'test']:
    idxs = np.array(splits[split])
    np.save(os.path.join(SAVE_DIR, f"X_{split}.npy"), X[idxs])
    np.save(os.path.join(SAVE_DIR, f"y_{split}.npy"), y[idxs])
    print(f"✅ [Disjoint] Saved {split} split with {len(idxs)} samples")

# === Save per-scene-per-split arrays ===
for split in ['train', 'val', 'test']:
    for scene in unique_scenes:
        idxs = [i for i in splits[split] if parse_source(sources[i])[0] == scene]
        np.save(os.path.join(SAVE_DIR, f"X_{scene}_{split}.npy"), X[idxs])
        np.save(os.path.join(SAVE_DIR, f"y_{scene}_{split}.npy"), y[idxs])
        print(f"✅ [Disjoint] Saved: X_{scene}_{split}.npy with {len(idxs)} samples")

# === Save metadata ===
metadata = pd.DataFrame({
    "filename": sources,
    "device": [parse_source(src)[1] for src in sources],
    "scene": [parse_source(src)[0] for src in sources],
    "split": split_labels
})
metadata.to_csv(os.path.join(SAVE_DIR, "all_split_metadata_disjoint.csv"), index=False)
print("📄 [Disjoint] Saved metadata for all samples")
