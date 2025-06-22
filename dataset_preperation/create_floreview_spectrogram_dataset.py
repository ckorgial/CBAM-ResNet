# === create_train_val_test_data_brand_disjoint_balanced.py ===
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = "./Data"
SAVE_DIR = "./Splits"
VIS_DIR = os.path.join(SAVE_DIR, "visualizations")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

SPLIT_RATIOS = (0.6, 0.2, 0.2)
SEED = 42
MAX_SAMPLES_PER_BRAND = 3000  # Cap total samples per brand to avoid Apple domination
np.random.seed(SEED)

# --- Load Data ---
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))
sources = np.load(os.path.join(DATA_PATH, "sources.npy"))
brands = np.load(os.path.join(DATA_PATH, "brands.npy"))

# --- Helper Functions ---
def parse_source(src):
    parts = src.replace('.npy', '').split('_')
    return parts[0], parts[1], parts[2]  # brand, device, recording

def plot_brand_distribution(metadata, title, filename):
    plt.figure(figsize=(12, 6))
    brand_counts = metadata.groupby(['brand', 'split']).size().unstack()
    brand_counts.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.xlabel("Brand")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()

# --- Group by Recording (for disjoint split) ---
recording_groups = defaultdict(list)
metadata_records = []

for idx, src in enumerate(sources):
    brand, device, recording = parse_source(src)
    key = (brand, device, recording)
    recording_groups[key].append(idx)
    metadata_records.append((src, brand, device, recording, y[idx]))

metadata_df = pd.DataFrame(metadata_records, columns=["filename", "brand", "device", "recording", "class"])

# --- Split per Brand ---
splits = {'train': [], 'val': [], 'test': []}
split_labels = np.empty(len(sources), dtype=object)
split_labels[:] = "none"

brand_recording_keys = defaultdict(list)
for key in recording_groups:
    brand = key[0]
    brand_recording_keys[brand].append(key)

for brand, rec_keys in brand_recording_keys.items():
    np.random.shuffle(rec_keys)

    if len(rec_keys) < 3:
        print(f"⚠️ Brand {brand} has only {len(rec_keys)} recordings – skipping.")
        continue

    n_total = len(rec_keys)
    n_train = max(1, int(n_total * SPLIT_RATIOS[0]))
    n_val = max(1, int(n_total * SPLIT_RATIOS[1]))
    n_test = max(1, n_total - n_train - n_val)

    if n_test == 0:
        n_train -= 1
        n_test = 1

    train_keys = rec_keys[:n_train]
    val_keys = rec_keys[n_train:n_train + n_val]
    test_keys = rec_keys[n_train + n_val:]

    for split_name, split_keys in zip(['train', 'val', 'test'], [train_keys, val_keys, test_keys]):
        indices = []
        for key in split_keys:
            indices.extend(recording_groups[key])
        np.random.shuffle(indices)
        indices = indices[:MAX_SAMPLES_PER_BRAND // 3]  # limit per brand per split
        splits[split_name].extend(indices)
        split_labels[indices] = split_name

# --- Save Splits ---
for split_name in ['train', 'val', 'test']:
    indices = np.array(splits[split_name])
    np.save(os.path.join(SAVE_DIR, f"X_{split_name}.npy"), X[indices])
    np.save(os.path.join(SAVE_DIR, f"y_{split_name}.npy"), y[indices])
    print(f"✅ Saved {split_name} split with {len(indices)} samples")

# --- Save Metadata ---
metadata_df["split"] = split_labels
metadata_df.to_csv(os.path.join(SAVE_DIR, "all_split_metadata.csv"), index=False)
print("📄 Saved metadata with split labels")

# --- Visualization ---
plot_brand_distribution(metadata_df, "Brand Distribution Across Splits", "brand_distribution.png")
