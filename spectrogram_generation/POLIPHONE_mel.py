import os
import glob
import librosa
import librosa.display
import numpy as np

def process_audio(wav_file_path, output_directory):
    # Load the audio at 44.1 kHz
    y, sr = librosa.load(wav_file_path, sr=44100)

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

    # Convert to decibels (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Save the Mel spectrogram as a NumPy file
    output_filename = os.path.splitext(os.path.basename(wav_file_path))[0] + '_mel.npy'
    output_path = os.path.join(output_directory, output_filename)
    np.save(output_path, mel_spectrogram_db)

def process_device_folder(base_directory, output_base_directory):
    # Iterate through each device folder
    for device_folder in glob.glob(os.path.join(base_directory, '*')):
        device_name = os.path.basename(device_folder)
        speech_clean_path = os.path.join(device_folder, 'speech_clean')

        # Skip if speech_clean doesn't exist
        if not os.path.isdir(speech_clean_path):
            continue

        # Output path: one per device
        output_device_path = os.path.join(output_base_directory, device_name)
        os.makedirs(output_device_path, exist_ok=True)

        for wav_file in glob.glob(os.path.join(speech_clean_path, '*.wav')):
            print(f"Processing {wav_file}")
            process_audio(wav_file, output_device_path)

# === Paths ===
base_directory = './POLIPHONE_44100/'
output_base_directory = './POLIPHONE_mel_44100/'

# === Start processing ===
process_device_folder(base_directory, output_base_directory)

