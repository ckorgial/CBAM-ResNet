import os
import glob
import librosa
import librosa.display
import numpy as np

def process_audio(wav_file_path, output_directory):
    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=44100)  # Ensuring consistent sampling rate

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # log

    # Save the Mel spectrogram as a NumPy file
    output_filename = os.path.splitext(os.path.basename(wav_file_path))[0] + '_mel.npy'
    output_path = os.path.join(output_directory, output_filename)
    np.save(output_path, mel_spectrogram_db)

def process_device_folder(base_directory, output_base_directory):
    # Navigate through each device folder
    for device_folder in glob.glob(os.path.join(base_directory, '*')):
        device_name = os.path.basename(device_folder)
        output_device_path = os.path.join(output_base_directory, device_name)

        # Create the output directory for the device
        os.makedirs(output_device_path, exist_ok=True)

        # Process each WAV file in the device folder
        for wav_file in glob.glob(os.path.join(device_folder, '*.wav')):
            print(f"Processing file: {wav_file}")
            # Call the process_audio function to handle the processing and saving of the audio
            process_audio(wav_file, output_device_path)

# Set the base directory where the device folders are located
base_directory = ''
# Set the base output directory where the .npy files will be saved
output_base_directory = ''

# Process audio for each device folder
process_device_folder(base_directory, output_base_directory)

print("Processing complete. All Mel spectrograms have been saved.")
