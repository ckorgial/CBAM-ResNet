import os
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_disjoint_splits(data_dir, save_dir, merged_map, excluded_devices,
                           test_size=0.2, val_size=0.15, min_samples_per_class=5,
                           seed=12345):
    """
    Creates strictly disjoint splits where recordings don't overlap between train/val/test.
    Ensures:
    1. No recording appears in more than one split
    2. Balanced class distribution across splits
    3. Minimum samples per class are maintained
    """

    os.makedirs(save_dir, exist_ok=True)
    analysis_dir = os.path.join(save_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Load data
    try:
        X = np.load(os.path.join(data_dir, "X.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
        sources = np.load(os.path.join(data_dir, "sources.npy"))
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        raise

    # Parse source information
    records = []
    class_rec_counts = defaultdict(int)

    for i, source in enumerate(sources):
        parts = source.split('_')
        if len(parts) < 3:
            continue

        scene = parts[0]
        device_folder = parts[1]
        rec_id = '_'.join(parts[2:]).replace('.npy', '')

        device_match = re.match(r'D(\d+)', device_folder)
        if not device_match:
            continue

        device = f"D{int(device_match.group(1)):02d}"
        if device in excluded_devices:
            continue

        group = merged_map.get(device, device)
        records.append({
            'index': i,
            'device': device,
            'group': group,
            'scene': scene,
            'rec_id': rec_id,
            'label': y[i]
        })
        class_rec_counts[group] += 1

    # Filter out classes with too few recordings
    valid_classes = [g for g, cnt in class_rec_counts.items() if cnt >= min_samples_per_class]
    records = [r for r in records if r['group'] in valid_classes]
    df = pd.DataFrame(records)

    # Get unique recordings per class
    class_recordings = df.groupby('group')['rec_id'].unique().to_dict()

    # Split recordings per class
    splits = {}
    for class_name, recordings in class_recordings.items():
        # First split: test set
        train_val, test = train_test_split(
            recordings,
            test_size=test_size,
            random_state=seed
        )

        # Second split: val from remaining
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            random_state=seed
        )

        splits[class_name] = {
            'train': train,
            'val': val,
            'test': test
        }

    # Assign splits
    df['split'] = 'train'  # Default
    for class_name, class_splits in splits.items():
        for split_name, recs in class_splits.items():
            df.loc[(df['group'] == class_name) & (df['rec_id'].isin(recs)), 'split'] = split_name

    # Verify no recording overlap
    rec_assignments = df.groupby('rec_id')['split'].nunique()
    if any(rec_assignments > 1):
        raise ValueError("Recording leakage detected - some recordings appear in multiple splits!")

    # Create label mapping
    classes = sorted(df['group'].unique())
    class_to_label = {c: i for i, c in enumerate(classes)}
    df['label'] = df['group'].map(class_to_label)

    # Save splits
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        indices = split_df['index'].values
        np.save(os.path.join(save_dir, f"X_{split}.npy"), X[indices])
        np.save(os.path.join(save_dir, f"y_{split}.npy"), split_df['label'].values)
        logger.info(f"Saved {split} split with {len(indices)} samples")

    # Save metadata
    df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

    # Plot class distributions
    plt.figure(figsize=(16, 8))
    sns.countplot(data=df, x='group', hue='split',
                  order=sorted(df['group'].unique()))
    plt.title('Class Distribution Across Splits (Disjoint)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "class_distribution_disjoint.png"))
    plt.close()

    logger.info(f"Disjoint splits created successfully in {save_dir}")


# Configuration
if __name__ == "__main__":
    DATA_DIR = "./Data/"
    SAVE_DIR = "./Splits_Merged"

    merged_map = {
        'D01': 'D01-D26', 'D26': 'D01-D26',
        'D02': 'D02-D10', 'D10': 'D02-D10',
        'D03': 'D03',
        'D05': 'D05-D14-D18', 'D14': 'D05-D14-D18', 'D18': 'D05-D14-D18',
        'D06': 'D06-D15', 'D15': 'D06-D15',
        'D07': 'D07', 'D08': 'D08', 'D09': 'D09',
        'D11': 'D11', 'D13': 'D13', 'D16': 'D16', 'D19': 'D19',
        'D20': 'D20', 'D21': 'D21', 'D23': 'D23', 'D24': 'D24', 'D25': 'D25',
        'D27': 'D27', 'D28': 'D28',
        'D29': 'D29-D34', 'D34': 'D29-D34',
        'D30': 'D30', 'D31': 'D31', 'D32': 'D32', 'D33': 'D33',
        'D35': 'D35'
    }

    excluded_devices = {'D04', 'D12', 'D17', 'D22'}

    create_disjoint_splits(
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        merged_map=merged_map,
        excluded_devices=excluded_devices,
        test_size=0.2,
        val_size=0.15,
        min_samples_per_class=5,
        seed=12345
    )
