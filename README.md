# Attention-Based Source Device Identification Using Audio Content from Videos and Grad-CAM Explanations

This repository implements source device identification using log-mel spectrograms of audios extracted from videos. It leverages [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf) architecture and supports training on the [VISION](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-017-0067-2) dataset, with robust transfer learning and evaluation on the [FloreView](https://ieeexplore.ieee.org/document/10271281) and [POLIPHONE](https://doi.org/10.48550/arXiv.2410.06221) datasets. Additionally, it includes [Grad-CAM](https://ieeexplore.ieee.org/document/8237336) visualizations to interpret model decisions.

---

## Project Structure

| Folder                         | Description                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| `ablation_study/`              | Ablation Study on VISION dataset (channel, spatial, & both modules)        |
| `audio_extraction/`            | Extract audio from videos using FFmpeg                                     |
| `dataset/`                     | VISION download                                                            |
| `dataset_preperation/`         | Create disjoint Train, Validation, and Test Splits                         |
| `spectrogram_generation/`      | Convert audio to log-Mel spectrograms                                      |
| `spectrogram_segmentation/`    | Create log-Mel Spectrogram Segments                                        |
| `training_vision/`             | Train CBAM-ResNet models on VISION dataset (standard & merged)             |
| `transfer_learning/`           | Transfer Learning of VISION models on FloreView and POLIPHONE datasets     |

---

## Requirements

Create a `requirements.txt` with:

```txt
tensorflow
librosa
scikit-learn
matplotlib
seaborn
numpy
opencv-python
```

Also ensure `ffmpeg` is installed on your system for audio extraction.

```bash
python -m venv cbam-resnet && sourcecbam-resnet/bin/activate && pip install -r requirements.txt
```

---

## Pipeline Instructions

### 1. VISION and FloreView Download - (NOTE: Download POLIPHONE manually)

You can download the [VISION](https://lesc.dinfo.unifi.it/VISION/) and the [FloreView](https://lesc.dinfo.unifi.it/FloreView/) dataset using the script

```bash
python dataset/download_datasets.py
```

### 2. Extract Audio from Videos

```bash
python audio_extraction/video2audio.py
```

### 3. Generate log-Mel Spectrograms

```bash
# For VISION dataset
python spectrogram_generation/VISION_mel.py

# For FloreView dataset
python spectrogram_generation/Floreview_Flat_mel.py

# For POLIPHONE dataset
python spectrogram_generation/POLIPHONE_mel.py
```

### 4. Create log-Mel Spectrograms Segments

```bash
# For VISION dataset
python spectrogram_segmentation/create_spectrogram_segments_vision.py

# For FloreView dataset
python spectrogram_segmentation/create_spectrogram_segments_floreview.py

# For POLIPHONE dataset
python spectrogram_segmentation/create_spectrogram_segments_poliphone.py
```


### 5. Create DISJOINT Train, Validation, and Test Splits (.npy files)

```bash
# VISION (Standard)
python dataset_preperation/create_spectrogram_dataset.py

# VISION (Merged)
python dataset_preperation/create_spectrogram_dataset_merged.py

# FloreView (Standard)
python dataset_preperation/create_floreview_spectrogram_dataset.py

# POLIPHONE (Standard)
python dataset_preperation/create_poliphone_spectrogram_dataset.py

```

### 6. Train and Test the CBAM-ResNet Model on VISION

```bash
# Standard CBAM-ResNet
python training_vision/train_test_model_cbam.py

# Merged CBAM-ResNet
python training_vision/train_test_model_merged_cbam.py
```

### 7. Ablation Study

```bash
# Ablation Study for VISION on Validation Set
python ablation_study/abl_cbam_resnet_patch.py
```


### 8. Transfer Learning on FloreView and POLIPHONE

```bash
# Transfer learning on FloreView
python transfer_learning/transfer_cbam_floreview.py

# Transfer learning on POLIPHONE
python transfer_learning/transfer_cbam_poliphone.py
```

---

## Outputs

Each training script saves:

- Best model weights (`best_model.h5`)
- Confusion matrices (as PNG and CSV)
- Accuracy trend plots
- Class-wise performance evaluation (AUC, Classification metrics)
- Grad-CAM Heatmaps for randomly selected log-Mel spectrogram patches for every device and scenario

---


---

## Author

**Christos Korgialas (**[**ckorgial@csd.auth.gr**](mailto\:ckorgial@csd.auth.gr)**)**

---

## License

MIT License

