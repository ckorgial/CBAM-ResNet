import os
import re
import numpy as np

# Parameters for spectrogram segmentation
HOP_LEN = 128
SAMPLING_RATE = 44100
DURATION = 300  # Duration in frames for each patch
SHIFT = 200  # Shift in frames between patches

spectro_dataset_path = ''

# Define device labels by brand
brand_to_label = {
    "Apple": 0,
    "Google": 1,
    "Huawei": 2,
    "LG": 3,
    "Motorola": 4,
    "OnePlus": 5,
    "Samsung": 6,
    "Xiaomi": 7
}

def iterate_dataset(path, brand_labels):
    for brand, label in brand_labels.items():
        folder_path = os.path.join(path, brand)

        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".npy"):
                        spectro_file_path = os.path.join(root, file)
                        print(f"Processing spectrogram file: {spectro_file_path}")
                        create_patches(spectro_file_path, label)
        else:
            print(f"Folder not found: {folder_path}")

def segment_spectrogram(spectrogram, duration, shift):
    """
    Separates a spectrogram file into smaller parts and returns the parts.

    Parameters
    ---------------
    spectrogram: np.ndarray
        The input spectrogram.
    duration: int
        The duration of each part (in frames).
    shift: int
        How much to shift to start creating the next fragment (in frames).

    Returns
    ---------------
    np.ndarray
        An array containing the segmented spectrogram patches.
    """
    parts = []

    start_sample = 0
    end_sample = duration

    while end_sample < spectrogram.shape[1]:
        part = spectrogram[:, start_sample:end_sample]
        parts.append(part)

        start_sample += shift
        end_sample += shift

    if start_sample < spectrogram.shape[1]:
        end_sample = spectrogram.shape[1]
        start_sample = max(0, end_sample - duration)
        part = spectrogram[:, start_sample:end_sample]
        parts.append(part)

    return np.array(parts)

def create_patches(spectro_path, label):
    x = np.load(spectro_path)
    x_parts = segment_spectrogram(x, DURATION, SHIFT)

    X.extend(x_parts)
    y.extend([label] * len(x_parts))

if __name__ == "__main__":
    X, y = [], []
    iterate_dataset(spectro_dataset_path, brand_to_label)

    # Save the segmented spectrogram dataset and labels
    np.save('X_FloreView_brand.npy', X)
    np.save('y_FloreView_brand.npy', y)
    print("Dataset creation complete.")

