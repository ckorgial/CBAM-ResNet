# Attention-Based Source Device Identification Using Audio Content from Videos and Grad-CAM Explanations

This repository provides a full pipeline for device identification from audio extracted from videos. It leverages CBAM-ResNet architectures and supports training on the VISION dataset and adaptation to the FLOREVIEW dataset.

---

## 📆 Project Structure

| Folder                         | Description                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| `audio_extraction/`            | Extract audio from videos using FFmpeg                                     |
| `spectrogram_generation/`      | Convert audio to mel spectrograms                                          |
| `dataset_preparation/`         | Segment spectrograms and create training datasets                          |
| `training_vision/`             | Train CBAM-ResNet models on VISION dataset (standard & TUBARO-filtered)    |
| `transfer_learning_floreview/` | Fine-tune VISION models on FLOREVIEW dataset (device, brand, merged views) |
| `utils/`                       | Placeholder for shared functions or helpers                                |

---

## 🔧 Requirements

Create a `requirements.txt` with:

```txt
tensorflow
librosa
scikit-learn
matplotlib
seaborn
numpy
```

Also ensure `ffmpeg` is installed on your system for audio extraction.

---

## 🚀 Pipeline Instructions

### 1. Extract Audio from Videos

```bash
python audio_extraction/video2audio.py
```

### 2. Generate Mel Spectrograms

```bash
# For VISION dataset\python spectrogram_generation/VISION_mel.py

# For FLOREVIEW dataset
python spectrogram_generation/Floreview_Flat_mel.py
```

### 3. Create Spectrogram Dataset (.npy files)

```bash
# VISION (Standard)
python dataset_preparation/create_spectrogram_dataset.py

# VISION (TUBARO-filtered)
python dataset_preparation/create_spectrogram_dataset_tubaro.py

# FLOREVIEW - Device-wise
python dataset_preparation/create_train_test_data.py

# FLOREVIEW - Brand-wise
python dataset_preparation/create_train_test_data_brand.py
```

### 4. Train CBAM Models on VISION

```bash
# Standard CBAM-ResNet
python training_vision/train_test_model_cbam.py

# TUBARO-enhanced CBAM-ResNet
python training_vision/train_test_model_tubaro_cbam.py
```

### 5. Transfer Learning on FLOREVIEW

```bash
# Device-wise fine-tuning
python transfer_learning_floreview/transfer_cbam.py

# Brand-wise fine-tuning
python transfer_learning_floreview/transfer_cbam_brand.py

# Merged-device test (unsupervised simulation)
python transfer_learning_floreview/transfer_cbam_merged.py
```

---

## 📊 Outputs

Each training script saves:

- Best model weights (`best_model.h5`)
- Confusion matrices (as PNG and CSV)
- Accuracy trend plots
- Class-wise performance evaluation (AUC, test metrics)

---

## 🚨 Notes

- All spectrograms are resized to 128x128
- Spectrograms are single-channel (grayscale)
- FLOREVIEW uses 46 classes (device-wise) or 8 brands (brand-wise)
- TUBARO version uses device merging strategy for robust generalization

---

## 🤖 Future Work

- Support for open-set and few-shot settings
- Integration with multimodal networks (video & audio)
- Expand with other datasets (e.g., DeepFake detection)

---

## 👤 Author

**Christos Korgialas**

---

## 📄 License

MIT License

