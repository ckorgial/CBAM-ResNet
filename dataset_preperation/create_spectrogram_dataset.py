import os
import re
import numpy as np


HOP_LEN = 128
SAMPLING_RATE = 44100
DURATION = 300
SHIFT = 200

spectro_dataset_path = '/media/blue/ckorgial/XAI_VISION/VISION_mel_all_44100_bandpass/'
recording_type = ['flat', 'indoor', 'outdoor']
recording_source = ['', 'YT', 'WA']
device_ids = ['D{:01}'.format(i) for i in range(1, 35)]


def iterate_dataset(path, devices, recording_types):
    for device_code in devices:
        for recording_type in recording_types:
            # Find folders that contain the device code
            matching_folders = [folder for folder in os.listdir(path) if device_code in folder]

            for matching_folder in matching_folders:
                folder_path = os.path.join(path, matching_folder, recording_type)

                if os.path.exists(folder_path):

                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            if file.endswith(".npy"):
                                spectro_file_path = os.path.join(root, file)
                                print(f"Processing WAV file: {spectro_file_path}")
                                # TODO Add processing logic here
                                create_patches(spectro_file_path)

                else:
                    print(f"Folder not found: {folder_path}")


def find_label(spectro_path):
    # Define a regular expression pattern to match "X" in the string
    pattern = r"D([\d]+)_"

    # Search for the pattern in the input string
    match = re.search(pattern, spectro_path)

    # If a match is found, extract and return the value of "X"
    if match:
        return int(match.group(1))
    else:
        # If no match is found, you can handle it accordingly, e.g., return None
        return None


def segment_spectrogram(spectrogram, duration, shift):
    """
    Separates a spectrogram file into smaller parts and returns the parts

    Parameters
    ---------------
    duration: int
        The duration of each part (in frames)
    shift:  int
        How much to shift to start creating the next fragment (in frames)
    wav_file_path: str
    """

    # Create the array to store th parts
    parts = []

    # Convert M and F from seconds to nof_samples
    '''duration_samples = duration * fs
    shift_samples = (duration - overlap) * fs'''

    start_sample = 0
    end_sample = duration
    i = 0

    while end_sample < spectrogram.shape[1]:
        # Extract the part from the audio
        part = spectrogram[:, start_sample:end_sample]

        # Store the part
        parts.append(part)

        start_sample += shift
        end_sample += shift
        i += 1

    # Create the last part containing the last minutes of the audio
    if start_sample < spectrogram.shape[1]:
        end_sample = spectrogram.shape[1]
        start_sample = end_sample - duration

        # Extract the part from the audio
        part = spectrogram[:, start_sample:end_sample]

        # Store the part
        parts.append(part)

    return np.array(parts)

def create_patches(spectro_path):
    # Find the spectrogram's device's labels
    label = find_label(spectro_path)

    # Read spectrogram
    x = np.load(spectro_path)

    # Create the patches
    x_parts = segment_spectrogram(x, DURATION, SHIFT)

    # Add parts and labels in X and y
    X.extend(x_parts)
    y.extend(np.array([label for i in range(len(x_parts))]))

    pass


# Example usage
devices_list = ['D01','D02','D03','D04','D05','D06','D07','D08','D09','D10','D11','D12','D13','D14','D15','D16',
                'D17','D18','D19','D20','D21','D22','D23','D24','D25','D26','D27','D28','D29','D30','D31','D32',
                'D33','D34','D35']
recording_types_list = ['flat'] # Changefor flat, indoor, and outdoor

X, y = [], []
iterate_dataset(spectro_dataset_path, devices_list, recording_types_list)
# Perform the process for flat, indoor and outdoor
np.save('X_flat.npy', X)
np.save('y_flat.npy', y)
