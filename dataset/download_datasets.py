import os
import requests
from tqdm import tqdm

# Set root directory for downloads
root_download = ''

def download_and_organize(urls, dataset_name):
    print(f"Downloading files for {dataset_name}...")
    pbar = tqdm(urls)

    for url in pbar:
        pbar.set_description(f"{dataset_name}: Processing")

        # Parse the URL to recreate folder hierarchy
        path_parts = url.split('/')
        folder_path = os.path.join(*path_parts[3:-1])  # skip domain

        # Make sure folder exists
        full_folder_path = os.path.join(root_download, folder_path)
        os.makedirs(full_folder_path, exist_ok=True)

        filename = path_parts[-1]
        file_path = os.path.join(full_folder_path, filename)

        if os.path.exists(file_path):
            pbar.set_description(f"{dataset_name}: Already exists")
            continue

        # Download file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"{dataset_name}: Failed to download: {url}")

if __name__ == "__main__":
    datasets = {
        "VISION": "metadataVISION/VISION_base_files.txt",
        "FloreView": "metadataFloreView/FloreView_Dataset.txt"
    }

    for dataset_name, filelist_path in datasets.items():
        with open(filelist_path, 'r') as file:
            url_list = [line.strip() for line in file if line.strip()]
        download_and_organize(url_list, dataset_name)
