import os
import numpy as np
import pandas as pd
from collections import defaultdict

# CONFIG
DATA_PATH = "./Data"
SAVE_DIR = "./Splits_Disjoint"
os.makedirs(SAVE_DIR, exist_ok=True)
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train, val, test
SEED = 42
np.random.seed(SEED)

# Load arrays
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))
sources = np.load(os.path.join(DATA_PATH, "sources.npy"))

def parse_source(src):
    parts = src.replace(".npy", "").split("_")
    try:
        speaker_idx = parts.index("speaker")
        speaker = f"{parts[speaker_idx]}_{parts[speaker_idx + 1]}"  # e.g., speaker_10
        device = "_".join(parts[:speaker_idx])                      # e.g., huawei_nova_9
        recording = "_".join(parts)                                 # full unique name
        return device, speaker, recording
    except (ValueError, IndexError):
        raise ValueError(f"❌ Could not parse speaker/device from filename: {src}")


# Build per-recording groups
recording_groups = defaultdict(list)
devices, speakers = [], []
for idx, src in enumerate(sources):
    device, speaker, recording = parse_source(src)
    recording_groups[(device, speaker, recording)].append(idx)
    devices.append(device)
    speakers.append(speaker)

unique_devices = sorted(set(devices))
unique_speakers = sorted(set(speakers))

# Split indices
splits = {'train': [], 'val': [], 'test': []}
split_labels = np.empty(len(sources), dtype=object)
split_labels[:] = "none"

for device in unique_devices:
    spks_for_dev = [s for (d, s, _) in recording_groups if d == device]
    spks_for_dev = sorted(set(spks_for_dev))
    np.random.shuffle(spks_for_dev)

    n = len(spks_for_dev)
    n_train = max(1, int(n * SPLIT_RATIOS[0]))
    n_val = max(1, int(n * SPLIT_RATIOS[1]))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n - n_train - n_test)

    train_spk = spks_for_dev[:n_train]
    val_spk = spks_for_dev[n_train:n_train + n_val]
    test_spk = spks_for_dev[n_train + n_val:]

    for (dev, spk, rec), idxs in recording_groups.items():
        if dev == device:
            if spk in train_spk:
                splits['train'].extend(idxs)
                split_labels[idxs] = "train"
            elif spk in val_spk:
                splits['val'].extend(idxs)
                split_labels[idxs] = "val"
            elif spk in test_spk:
                splits['test'].extend(idxs)
                split_labels[idxs] = "test"

# Save split arrays
for split in ['train', 'val', 'test']:
    idxs = np.array(splits[split])
    np.save(os.path.join(SAVE_DIR, f"X_{split}.npy"), X[idxs])
    np.save(os.path.join(SAVE_DIR, f"y_{split}.npy"), y[idxs])
    np.save(os.path.join(SAVE_DIR, f"sources_{split}.npy"), sources[idxs])
    print(f"✅ Saved {split} split: {len(idxs)} samples")

# Diagnostics and metadata
metadata = pd.DataFrame({
    "filename": sources,
    "device": [parse_source(src)[0] for src in sources],
    "speaker": [parse_source(src)[1] for src in sources],
    "split": split_labels
})
metadata.to_csv(os.path.join(SAVE_DIR, "all_split_metadata.csv"), index=False)
print("📄 Saved metadata CSV for all samples.")

# Report missing device-split combinations
for split in ['train', 'val', 'test']:
    present = set((parse_source(sources[i])[0]) for i in splits[split])
    missing = []
    for d in unique_devices:
        if d not in present:
            missing.append(d)
    if missing:
        print(f"\n⚠️ {split.upper()} split is missing devices: {missing}")
    else:
        print(f"\n✅ {split.upper()} split contains all devices.")

print("🚦 Splitting complete and reviewer-proof!")
