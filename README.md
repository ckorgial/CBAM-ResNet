# Attention-Based Source Device Identification Using Audio Content from Videos and Grad-CAM Explanations

This repository provides a full pipeline for source device identification from audio extracted from videos. It leverages CBAM-ResNet architectures and supports training on the VISION dataset and adaptation to the FLOREVIEW dataset.

---

## Project Structure

| Folder                         | Description                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| `audio_extraction/`            | Extract audio from videos using FFmpeg                                     |
| `spectrogram_generation/`      | Convert audio to mel spectrograms                                          |
| `dataset_preparation/`         | Segment spectrograms and create training datasets                          |
| `training_vision/`             | Train CBAM-ResNet models on VISION dataset (standard & filtered variant)   |
| `abblation_study/`             | Abblation Study on VISION dataset (channel, spatial, & both modules)       |
| `transfer_learning_floreview/` | Fine-tune VISION models on FLOREVIEW dataset (device & brand views)        |
| `utils/`                       | Placeholder for shared functions or helpers                                |

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

### 1. Extract Audio from Videos

```bash
python audio_extraction/video2audio.py
```

### 2. Generate Mel Spectrograms

```bash
# For VISION dataset
python spectrogram_generation/VISION_mel.py

# For FLOREVIEW dataset
python spectrogram_generation/Floreview_Flat_mel.py
```

### 3. Create Spectrogram Dataset (.npy files)

```bash
# VISION (Standard)
python dataset_preparation/create_spectrogram_dataset.py

# VISION (Filtered Variant)
python dataset_preparation/create_spectrogram_dataset_filtered_variant.py

# FLOREVIEW - Device-wise
python dataset_preparation/create_train_test_data.py

# FLOREVIEW - Brand-wise
python dataset_preparation/create_train_test_data_brand.py
```

### 4. Train CBAM Models on VISION

```bash
# Standard CBAM-ResNet
python training_vision/train_test_model_cbam.py

# Filtered CBAM-ResNet Variant
python training_vision/train_test_model_filtered_variant_cbam.py
```

### 5. Abblation Study

```bash
# Bblation Study for VISION on Validation Set
python abblation_study/val_model_channel_spatial_both_abbl.py
```

### 6. Transfer Learning on FLOREVIEW

```bash
# Device-wise fine-tuning
python transfer_learning_floreview/transfer_cbam.py

# Brand-wise fine-tuning
python transfer_learning_floreview/transfer_cbam_brand.py

# Merged-device test (unsupervised simulation)
python transfer_learning_floreview/transfer_cbam_merged.py
```

---

## Outputs

Each training script saves:

- Best model weights (`best_model.h5`)
- Confusion matrices (as PNG and CSV)
- Accuracy trend plots
- Class-wise performance evaluation (AUC, test metrics)

---

## Notes

- All spectrograms are resized to 128x128
- Spectrograms are single-channel (grayscale)
- FLOREVIEW uses 46 classes (device-wise) or 8 brands (brand-wise)
- Filtered variant uses device merging strategy for robust generalization

---

## Grad-CAM Visualizations

This repository supports Grad-CAM visualizations for model interpretability.

- A dedicated script `gradcam_vision_cbam.py` allows applying Grad-CAM using the **best training checkpoint from your CBAM model**.
- The model visualizes heatmaps over mel spectrograms indicating regions contributing most to the classification decision.

**Grad-CAM pipeline steps:**

1. Load and preprocess a mel spectrogram image
2. Load the best trained CBAM-ResNet model from this repo (e.g., `Results_test_cbam/best_model.h5`)
3. Extract features from the last convolutional layer
4. Apply CBAM modules post hoc (channel + spatial attention)
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

