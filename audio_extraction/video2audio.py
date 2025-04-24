import os
import subprocess

def extract_audio_from_video(video_path, output_directory):
    try:
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        audio_output_filename = video_name + ".wav"
        audio_output_path = os.path.join(output_directory, audio_output_filename)

        # Use subprocess to call ffmpeg for audio extraction
        subprocess.run(['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '44100', audio_output_path],
                       check=True)
        print(f"Extracted audio: {audio_output_path}")
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def process_videos_in_classes(class_folder_path, output_directory):
    for class_folder in os.listdir(class_folder_path):
        class_path = os.path.join(class_folder_path, class_folder)
        output_class_folder = os.path.join(output_directory, class_folder)

        if os.path.isdir(class_path):
            os.makedirs(output_class_folder, exist_ok=True)

            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.mov')):
                    video_path = os.path.join(class_path, video_file)
                    extract_audio_from_video(video_path, output_class_folder)

if __name__ == "__main__":
    dataset_path = ""
    output_directory = ""

    os.makedirs(output_directory, exist_ok=True)
    process_videos_in_classes(dataset_path, output_directory)

