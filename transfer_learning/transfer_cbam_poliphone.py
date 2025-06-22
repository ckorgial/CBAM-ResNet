
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
DATA_DIR = "./Splits"
SAVE_DIR = "./Results"
VISION_WEIGHTS = "./CBAM-ResNet/Results/best_model.h5"
NUM_CLASSES = 20
PATCH_WIDTH = 128
LABELS = [f"D{i+1:02}" for i in range(NUM_CLASSES)]
os.makedirs(SAVE_DIR, exist_ok=True)

# --- PATCHING ---
def extract_patches(spectrogram, patch_width=128, stride=128):
    if spectrogram.ndim == 4:
        spectrogram = spectrogram.squeeze()
    if spectrogram.ndim == 3 and spectrogram.shape[-1] == 1:
        spectrogram = spectrogram.squeeze(-1)
    if spectrogram.ndim == 3 and spectrogram.shape[0] == 1:
        spectrogram = spectrogram.squeeze(0)
    patches = []
    for start in range(0, spectrogram.shape[1] - patch_width + 1, stride):
        patch = spectrogram[:, start:start + patch_width]
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-6)
        if patch.ndim == 2:
            patch = patch[..., None]
        patches.append(patch)
    return patches

# --- DATA LOADING ---
def load_data(split):
    X_raw = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, f"y_{split}.npy"))
    X_patches, y_patches = [], []
    for spec, label in zip(X_raw, y_raw):
        for patch in extract_patches(spec):
            X_patches.append(patch)
            y_patches.append(label)
    return np.stack(X_patches), np.array(y_patches)

X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

# --- CBAM-RESNET BACKBONE ---
def cbam_block(x, filters, ratio=8, block_id=None):
    suffix = f"_{block_id}" if block_id is not None else ""
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    dense_1 = layers.Dense(filters // ratio, activation='relu')
    dense_2 = layers.Dense(filters)
    avg_out = dense_2(dense_1(avg_pool))
    max_out = dense_2(dense_1(max_pool))
    channel = layers.Add(name=f"channel_attention{suffix}")([avg_out, max_out])
    channel = layers.Activation('sigmoid')(channel)
    channel = layers.Reshape((1, 1, filters))(channel)
    x = layers.Multiply()([x, channel])
    avg_pool2 = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool2 = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool2, max_pool2])
    spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=f"spatial_attention{suffix}")(concat)
    x = layers.Multiply(name=f"cbam_multiply{suffix}")([x, spatial])
    return x

def residual_block(x, filters, stride=1, block_id=None):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = cbam_block(x, filters, block_id=block_id)
    return x

def build_cbam_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = residual_block(x, 32, block_id=1)
    x = residual_block(x, 64, block_id=2)
    x = residual_block(x, 128, block_id=3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', name="feature_head")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)
    return models.Model(inputs, outputs)

# --- TRANSFER LEARNING SETUP ---
vision_model = build_cbam_resnet((128, 128, 1), 35)
vision_model.load_weights(VISION_WEIGHTS)

x = vision_model.get_layer("feature_head").output
output = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)
model = tf.keras.Model(vision_model.input, output)

for layer in vision_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# --- TRAIN ---
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.h5"), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
)

model.save_weights(os.path.join(SAVE_DIR, "cbam_transfer_poliphone_weights.h5"))

y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")
with open(os.path.join(SAVE_DIR, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")

val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")
with open(os.path.join(SAVE_DIR, "val_accuracy.txt"), "w") as f:
    f.write(f"Validation Accuracy: {val_acc * 100:.2f}%\n")

y_val_pred = np.argmax(model.predict(X_val), axis=1)
cm_val = confusion_matrix(y_val, y_val_pred, labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(14, 12))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix - POLIPHONE (Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_val_poliphone.png"))
plt.close()

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
with open(os.path.join(SAVE_DIR, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")

y_test_pred = np.argmax(model.predict(X_test), axis=1)
cm_test = confusion_matrix(y_test, y_test_pred, labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(14, 12))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix - Test Clean")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_poliphone.png"))
plt.close()



