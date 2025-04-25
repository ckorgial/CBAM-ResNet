# Attention-Based Source Device Identification Using Audio Content from Videos and Grad-CAM Explanations

This repository implements source device identification using log-mel spectrograms of audios extracted from videos. It leverages [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf) architecture and supports training on the [VISION](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-017-0067-2) dataset, with robust transfer learning and evaluation on the [FloreView](https://ieeexplore.ieee.org/document/10271281) dataset. Additionally, it includes [Grad-CAM](https://ieeexplore.ieee.org/document/8237336) visualizations to interpret model decisions.

---

## Project Structure

| Folder                         | Description                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| `ablation_study/`              | Ablation Study on VISION dataset (channel, spatial, & both modules)        |
| `audio_extraction/`            | Extract audio from videos using FFmpeg                                     |
| `dataset/`                     | VISION and FloreView download                                              |
| `dataset_preparation/`         | Segment spectrograms and create training datasets                          |
| `spectrogram_generation/`      | Convert audio to log-Mel spectrograms                                      |
| `statistical_testing/`         | McNemar's statistical testing                                              |
| `training_vision/`             | Train CBAM-ResNet models on VISION dataset (standard & merged)             |
| `transfer_learning_floreview/` | Fine-tune VISION models on FLOREVIEW dataset (device & brand views)        |

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

---

## Pipeline Instructions

### 1. VISION and FloreView Download

You can download the [VISION](https://lesc.dinfo.unifi.it/VISION/) and the [FloreView](https://lesc.dinfo.unifi.it/FloreView/) dataset using the script

```bash
python dataset/download_datasets.py
```

### 2. Extract Audio from Videos

```bash
python audio_extraction/video2audio.py
```

### 3. Generate Mel Spectrograms

```bash
# For VISION dataset
python spectrogram_generation/VISION_mel.py

# For FLOREVIEW dataset
python spectrogram_generation/Floreview_Flat_mel.py
```

### 4. Create Spectrogram Dataset (.npy files)

```bash
# VISION (Standard)
python dataset_preparation/create_spectrogram_dataset.py

# VISION (Merged)
python dataset_preparation/create_spectrogram_dataset_merged.py

# FLOREVIEW - Device-wise
python dataset_preparation/create_floreview_spectrogram_dataset.py

# FLOREVIEW - Brand-wise
python dataset_preparation/create_floreview_spectrogram_dataset_brand.py
```

### 5. Train and Test the CBAM-ResNet-based Model on VISION

```bash
# Standard CBAM-ResNet
python training_vision/train_test_model_cbam.py

# Filtered CBAM-ResNet Variant
python training_vision/train_test_model_merged_cbam.py
```

### 6. Ablation Study

```bash
# Ablation Study for VISION on Validation Set
python abblation_study/val_model_channel_spatial_both_abl.py
```

### 7. Statistical Testing

```bash
# McNemar's statistical testing
python statistical_testing/mcnemars.py
```

### 8. Transfer Learning on FLOREVIEW

```bash
# Device-wise fine-tuning
python transfer_learning_floreview/transfer_cbam.py

# Brand-wise fine-tuning
python transfer_learning_floreview/transfer_cbam_brand.py
```

---

## Outputs

Each training script saves:

- Best model weights (`best_model.h5`)
- Confusion matrices (as PNG and CSV)
- Accuracy trend plots
- Class-wise performance evaluation (AUC, test metrics)

---

## Grad-CAM Visualizations

This repository supports Grad-CAM visualizations for model interpretability.

- A dedicated script `gradcam_vision_cbam.py` allows applying Grad-CAM using the **best training checkpoint from the CBAM-ResNet-based model**.
- The model visualizes heatmaps over mel spectrograms, indicating regions contributing most to the classification decision.

**Grad-CAM pipeline steps:**

1. Load and preprocess a log-Mel spectrogram
2. Load the best trained CBAM-ResNet-based model from this repo (e.g., `Results_test_cbam/best_model.h5`)
3. Extract features from the last convolutional layer
5. Compute gradients and create a class-discriminative heatmap
6. Overlay the heatmap on the original spectrogram for visualization

```bash
# Example (adapt the model path and image path):
python gradcam_vision_cbam.py
```

---

## Author

**Christos Korgialas (**[**ckorgial@csd.auth.gr**](mailto\:ckorgial@csd.auth.gr)**)**

---

## License

MIT License

