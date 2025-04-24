import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os

# --- CONFIGURATION ---
MODEL_PATH = "training_vision/best_model.h5"  # Update path as needed
IMG_PATH = "sample_input.png"  # Path to mel spectrogram image ! 
TARGET_SIZE = (128, 128)  # Size used during model training

# --- PREPROCESS IMAGE ---
image = img_to_array(load_img(IMG_PATH, target_size=TARGET_SIZE))
original_image = image.copy()
image = np.expand_dims(image, axis=0) / 255.0  # Normalize to [0, 1]

# --- LOAD TRAINED CBAM MODEL ---
model = load_model(MODEL_PATH)
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# Get last conv layer model
last_conv_layer = model.get_layer(last_conv_layer_name)
conv_output_model = tf.keras.Model(model.inputs, last_conv_layer.output)

# Classifier path from conv output to final prediction
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x_clf = classifier_input
for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
    x_clf = layer(x_clf)
classifier_model = tf.keras.Model(classifier_input, x_clf)

# --- COMPUTE GRAD-CAM ---
with tf.GradientTape() as tape:
    tape.watch(image)
    conv_outputs = conv_output_model(image)
    tape.watch(conv_outputs)
    preds = classifier_model(conv_outputs)
    top_class = tf.argmax(preds[0])
    top_output = preds[:, top_class]

grads = tape.gradient(top_output, conv_outputs)[0]
pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
feature_maps = conv_outputs[0].numpy()
for i in range(pooled_grads.shape[-1]):
    feature_maps[:, :, i] *= pooled_grads[i]

heatmap = np.mean(feature_maps, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) + 1e-10
heatmap = cv2.resize(heatmap, TARGET_SIZE)

# --- VISUALIZE GRAD-CAM ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image.astype("uint8"))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(original_image.astype("uint8"))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()