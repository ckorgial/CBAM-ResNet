import os
import re
import numpy as np

# Parameters for spectrogram segmentation
HOP_LEN = 128
SAMPLING_RATE = 44100
DURATION = 300  # Duration in frames for each patch
SHIFT = 200  # Shift in frames between patches

spectro_dataset_path = ''

# Define device classes
device_ids = [f'D{str(i).zfill(2)}' for i in range(1, 47)]


def iterate_dataset(path, devices):
    for device_code in devices:
        # Find folders that match the device code
        matching_folders = [folder for folder in os.listdir(path) if device_code in folder]

        for matching_folder in matching_folders:
            folder_path = os.path.join(path, matching_folder)

            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(".npy"):
                            spectro_file_path = os.path.join(root, file)
                            print(f"Processing spectrogram file: {spectro_file_path}")
                            create_patches(spectro_file_path)
            else:
                print(f"Folder not found: {folder_path}")

def find_label(spectro_path):
    # Define a regular expression pattern to match the device label
    pattern = r"D([\d]+)_"

    # Search for the pattern in the input string
    match = re.search(pattern, spectro_path)

    if match:
        return int(match.group(1))
    else:
        return None

def segment_spectrogram(spectrogram, duration, shift):
    """
    Separates a spectrogram file into smaller parts and returns the parts

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

def create_patches(spectro_path):
    label = find_label(spectro_path)
    if label is None:
        print(f"No label found for {spectro_path}, skipping.")
        return

    x = np.load(spectro_path)
    x_parts = segment_spectrogram(x, DURATION, SHIFT)

    X.extend(x_parts)
    y.extend([label] * len(x_parts))

if __name__ == "__main__":
    X, y = [], []
    iterate_dataset(spectro_dataset_path, device_ids)

    # Save the segmented spectrogram dataset and labels
    np.save('X_FloreView.npy', X)
    np.save('y_FloreView.npy', y)
    print("Dataset creation complete.")

