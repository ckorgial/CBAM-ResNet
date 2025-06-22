import os
import numpy as np
import tensorflow as tf
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = "./Splits"
SAVE_DIR = "./Results"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_CLASSES = 35
LABELS = [f'D{i+1:02}' for i in range(NUM_CLASSES)]
SR = 44100
HOP_LENGTH = 512
PATCH_WIDTH = 128
SCENES = ['flat', 'indoor', 'outdoor']
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
GRAD_DIR = os.path.join(SAVE_DIR, "gradcam")
os.makedirs(GRAD_DIR, exist_ok=True)
CBAM_DIR = os.path.join(SAVE_DIR, "cbam")
os.makedirs(CBAM_DIR, exist_ok=True)



# --- PATCH EXTRACTION ---
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
def load_data(name):
    X_raw = np.load(os.path.join(DATA_DIR, f"X_{name}.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    if y_raw.max() == 35:
        y_raw -= 1
    X_patches, y_patches = [], []
    for spectro, label in zip(X_raw, y_raw):
        for patch in extract_patches(spectro, PATCH_WIDTH, PATCH_WIDTH):  # non-overlapping
            X_patches.append(patch)
            y_patches.append(label)
    return np.stack(X_patches), np.array(y_patches)

X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

# --- MODEL (CBAM-ResNet) ---
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
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = build_cbam_resnet((128, 128, 1), NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(8e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# --- TRAINING ---
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.h5"), monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
]
model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=80, batch_size=32, class_weight=class_weights, callbacks=callbacks
)
model.save_weights(os.path.join(SAVE_DIR, "cbam_resnet_trained_weights.h5"))

# --- Save predictions for McNemar (Overall) ---
print("Saving CBAM predictions and ground truth for McNemar's test...")
y_pred_test = np.argmax(model.predict(X_test), axis=1)
np.save(os.path.join(SAVE_DIR, "cbam_preds.npy"), y_pred_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

# --- Save predictions for McNemar (Overall) ---
print("Saving CBAM predictions and ground truth for McNemar's test...")
y_pred_test = np.argmax(model.predict(X_test), axis=1)
np.save(os.path.join(SAVE_DIR, "cbam_preds.npy"), y_pred_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

# --- Save predictions per scenario (Flat, Indoor, Outdoor) ---
scene_data = {}
for scene in SCENES:
    X_path = os.path.join(DATA_DIR, f"X_{scene}_test.npy")
    y_path = os.path.join(DATA_DIR, f"y_{scene}_test.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        Xs = np.load(X_path)
        ys = np.load(y_path)
        if ys.max() == 35:
            ys -= 1
        X_patches, y_patches = [], []
        for spec, label in zip(Xs, ys):
            for patch in extract_patches(spec, PATCH_WIDTH, PATCH_WIDTH):
                X_patches.append(patch)
                y_patches.append(label)
        X_scene = np.stack(X_patches)
        y_scene = np.array(y_patches)
        scene_data[f"X_{scene}_test"] = X_scene
        scene_data[f"y_{scene}_test"] = y_scene
        y_pred_scene = np.argmax(model.predict(X_scene), axis=1)
        np.save(os.path.join(SAVE_DIR, f"cbam_preds_{scene}.npy"), y_pred_scene)
        np.save(os.path.join(SAVE_DIR, f"y_{scene}.npy"), y_scene)
        print(f"✅ Saved predictions for scene: {scene}")
    else:
        print(f"⚠️ Scene data not found for: {scene}")

# --- EVALUATION ---
def report_metrics(X, y, label):
    preds = model.predict(X)
    acc = np.mean(np.argmax(preds, axis=1) == y)
    print(f"✅ {label} Accuracy: {acc:.2%}")
    return acc

def create_confusion(X, y, label):
    y_pred = np.argmax(model.predict(X), axis=1)
    cm = confusion_matrix(y, y_pred, labels=np.arange(NUM_CLASSES))
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{label.lower().replace(' ', '_')}.png"))
    plt.close()

def plot_roc_curve_ovo(model, X, y, prefix, save_dir=SAVE_DIR, max_splits=5):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # Binarize the labels for multiclass OvO ROC
    y_bin = label_binarize(y, classes=np.arange(NUM_CLASSES))

    # Determine number of CV splits
    unique, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    n_splits = min(max_splits, min_samples_per_class)
    if n_splits < 2:
        print(f"⚠️ Too few samples for ROC AUC calculation in split '{prefix}'. Skipping.")
        return

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, test_idx in cv.split(X, y):
        y_score = model.predict(X[test_idx])
        fpr, tpr, _ = roc_curve(y_bin[test_idx].ravel(), y_score.ravel())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)

    # Find the optimal point: maximum TPR - FPR
    optimal_idx = np.argmax(mean_tpr - mean_fpr)
    optimal_fpr = mean_fpr[optimal_idx]
    optimal_tpr = mean_tpr[optimal_idx]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Avg ROC (AUC = {mean_auc:.3f})', lw=2, alpha=0.9)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.scatter(optimal_fpr, optimal_tpr, c='red', s=70,
                label=f'Optimal (FPR={optimal_fpr:.2f}, TPR={optimal_tpr:.2f})')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OvO ROC Curve with Optimal Point")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f"roc_{prefix.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved ROC-AUC plot for '{prefix}' to: {save_path}")

    return mean_auc, optimal_fpr, optimal_tpr


# --- SCENE SPLITS (ALL PATCHES) ---
scene_data = {}
for split in ["train", "val", "test"]:
    for scene in SCENES:
        X_path = os.path.join(DATA_DIR, f"X_{scene}_{split}.npy")
        y_path = os.path.join(DATA_DIR, f"y_{scene}_{split}.npy")
        if os.path.exists(X_path) and os.path.exists(y_path):
            Xs = np.load(X_path)
            ys = np.load(y_path)
            if ys.max() == 35:
                ys -= 1
            X_patches, y_patches = [], []
            for spec, label in zip(Xs, ys):
                for patch in extract_patches(spec, PATCH_WIDTH, PATCH_WIDTH):
                    X_patches.append(patch)
                    y_patches.append(label)
            scene_data[f"X_{scene}_{split}"] = np.stack(X_patches)
            scene_data[f"y_{scene}_{split}"] = np.array(y_patches)

splits = [("Train Overall", X_train, y_train), ("Val Overall", X_val, y_val), ("Test Overall", X_test, y_test)]
for split in ["train", "val", "test"]:
    for scene in SCENES:
        kx = f"X_{scene}_{split}"
        ky = f"y_{scene}_{split}"
        if kx in scene_data and ky in scene_data:
            splits.append((f"{scene.capitalize()} {split.capitalize()}", scene_data[kx], scene_data[ky]))

metrics_log = []
for name, X_split, y_split in splits:
    acc = report_metrics(X_split, y_split, name)
    create_confusion(X_split, y_split, name)
    plot_roc_curve_ovo(model, X_split, y_split, name)
    try:
        y_bin = label_binarize(y_split, classes=np.arange(NUM_CLASSES))
        y_score = model.predict(X_split)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        auc_val = auc(fpr, tpr)
    except Exception:
        auc_val = np.nan
    metrics_log.append({"Split": name, "Accuracy": acc, "AUC": auc_val})

df_metrics = pd.DataFrame(metrics_log)
metrics_path = os.path.join(SAVE_DIR, "all_split_metrics.csv")
df_metrics.to_csv(metrics_path, index=False)
print(f"✅ Saved metrics to {metrics_path}")

# --- EXPLAINABILITY (GRADCAM, CBAM ATTENTION) ---
def get_last_conv_layer(model):
    mult_layers = [l.name for l in model.layers if "multiply" in l.name]
    if mult_layers:
        return mult_layers[-1]
    conv_layers = [l.name for l in model.layers if "conv2d" in l.name]
    return conv_layers[-1] if conv_layers else None

def compute_gradcam(model, sample, gradcam_layer):
    x = sample[None, ...]
    grad_model = tf.keras.Model([model.input], [model.get_layer(gradcam_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x)
        class_idx = np.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_output[0]
    gradcam = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    gradcam = np.maximum(gradcam, 0)
    gradcam = gradcam / (gradcam.max() + 1e-8)
    gradcam_resized = tf.image.resize(gradcam[..., None], sample.shape[:2]).numpy().squeeze()
    return gradcam_resized


import librosa
import numpy as np


def get_spec_axes(spec, sr=44100, hop_length=512, fmin=0, fmax=None):
    """
    Returns frequency and time axis arrays for a mel spectrogram.

    Parameters:
    - spec: 2D numpy array of shape (n_mels, time_bins)
    - sr: Sampling rate (default: 44100)
    - hop_length: Hop length used in STFT (default: 512)
    - fmin: Minimum frequency for mel scale (default: 0 Hz)
    - fmax: Maximum frequency for mel scale (default: sr/2)

    Returns:
    - freqs: Array of mel band center frequencies (y-axis)
    - times: Array of times in seconds (x-axis)
    """
    n_mels, time_bins = spec.shape[:2]
    if fmax is None:
        fmax = sr // 2

    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    times = np.arange(time_bins) * hop_length / sr
    return freqs, times

def plot_spectrogram_and_gradcam(sample, gradcam_map, device_label, scenario, sr=SR, hop_length=HOP_LENGTH, save_dir=GRAD_DIR, dpi=150):
    if sample.ndim == 3:
        sample = sample.squeeze()
    n_fft_bins, n_frames = sample.shape
    freqs, times = get_spec_axes(sample, sr=sr, hop_length=hop_length)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmin, vmax = sample.min(), sample.max()
    # 1. Original spectrogram
    im = axes[0].imshow(sample, aspect='auto', origin='lower', vmin=vmin, vmax=vmax,
                        extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='magma')
    axes[0].set_title('Original Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    fig.colorbar(im, ax=axes[0])
    # 2. Grad-CAM overlay
    axes[1].imshow(sample, aspect='auto', origin='lower', vmin=vmin, vmax=vmax,
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='magma')
    axes[1].imshow(gradcam_map, alpha=0.5, cmap='jet', aspect='auto', origin='lower',
                   extent=[times[0], times[-1], freqs[0], freqs[-1]])
    axes[1].set_title('Grad-CAM Overlay')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(axes[1].images[1], ax=axes[1], label='Grad-CAM intensity')
    plt.suptitle(f'Device {device_label} | Scenario: {scenario}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"gradcam_{device_label}_{scenario}.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def get_cbam_layer_names_and_conv(model, block_idx, expected_channels):
    channel_layer = f"channel_attention_{block_idx}"
    spatial_layer = f"spatial_attention_{block_idx}"
    conv_layer_name = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Conv2D) and l.output_shape[-1] == expected_channels:
            conv_layer_name = l.name
    if conv_layer_name is None:
        raise ValueError(f"No Conv2D with {expected_channels} channels found for block {block_idx}")
    return channel_layer, spatial_layer, conv_layer_name

def plot_channel_attention_heatmap(channel_vec, save_path):
    plt.figure(figsize=(8, 2))
    plt.bar(np.arange(len(channel_vec)), channel_vec)
    plt.xlabel("Channel Index")
    plt.ylabel("Attention Weight")
    plt.title("CBAM Channel Attention")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spatial_attention_map(spatial_map, spec, save_path, sr=SR, hop_length=HOP_LENGTH):
    freq_bins, time_bins = spec.shape[:2]
    if isinstance(spatial_map, tf.Tensor):
        spatial_map = spatial_map.numpy()
    spatial_map = np.squeeze(spatial_map)

    print(f"[INFO] Spatial attention shape before resize: {spatial_map.shape}")
    # Spatial attention maps are not necessarily square due to downsampling in the network.
    # We resize them to (freq_bins, time_bins) to match the original spectrogram dimensions for visualization only.
    spatial_map_up = tf.image.resize(spatial_map[..., np.newaxis], (freq_bins, time_bins)).numpy().squeeze()
    print(f"[INFO] CBAM spatial attention shape AFTER resize: {spatial_map_up.shape}")
    freqs, times = get_spec_axes(spec, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(7, 7))  # Ensures square canvas
    plt.imshow(spatial_map_up, cmap='jet', aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("CBAM Spatial Attention")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_combined_attention(feature_map, channel_attention, spatial_attention, spec, save_path, sr=SR, hop_length=HOP_LENGTH):
    freq_bins, time_bins = spec.shape[:2]
    if isinstance(channel_attention, tf.Tensor):
        channel_attention = channel_attention.numpy()
    if isinstance(spatial_attention, tf.Tensor):
        spatial_attention = spatial_attention.numpy()
    if isinstance(feature_map, tf.Tensor):
        feature_map = feature_map.numpy()
    ca = channel_attention.reshape(1, 1, -1)
    sa = np.squeeze(spatial_attention)
    att_map = feature_map * ca
    att_map = att_map * sa[..., np.newaxis]
    att_map_2d = np.mean(att_map, axis=-1)
    att_map_2d_up = tf.image.resize(att_map_2d[..., np.newaxis], (freq_bins, time_bins)).numpy().squeeze()
    freqs, times = get_spec_axes(spec, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(7, 7))
    plt.imshow(att_map_2d_up, cmap="jet", aspect="auto", origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("CBAM Combined Channel × Spatial Attention")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_and_plot_cbam(model, X, y):
    block_channels = {1: 32, 2: 64, 3: 128}
    for block_idx in [1, 2, 3]:
        n_channels = block_channels[block_idx]
        channel_name, spatial_name, conv_name = get_cbam_layer_names_and_conv(model, block_idx, n_channels)
        print(f"[CBAM Block {block_idx}] Layers: {channel_name}, {spatial_name}, {conv_name}")
        channel_model = tf.keras.Model(model.input, model.get_layer(channel_name).output)
        spatial_model = tf.keras.Model(model.input, model.get_layer(spatial_name).output)
        conv_model = tf.keras.Model(model.input, model.get_layer(conv_name).output)
        for cls in range(NUM_CLASSES):
            indices = np.where(y == cls)[0]
            if len(indices) == 0:
                continue
            sidx = indices[0]
            patch = X[sidx]
            label = LABELS[cls]
            ch_path = os.path.join(CBAM_DIR, f"chatt_block{block_idx}_{label}_sample{sidx}.png")
            sp_path = os.path.join(CBAM_DIR, f"spatt_block{block_idx}_{label}_sample{sidx}.png")
            both_path = os.path.join(CBAM_DIR, f"bothatt_block{block_idx}_{label}_sample{sidx}.png")
            patch_batched = patch[None, ...]
            channel_attention = channel_model(patch_batched)[0]
            plot_channel_attention_heatmap(channel_attention, save_path=ch_path)
            spatial_attention = spatial_model(patch_batched)[0]
            plot_spatial_attention_map(spatial_attention, patch.squeeze(), sp_path)
            feature_map = conv_model(patch_batched)[0]
            plot_combined_attention(feature_map, channel_attention, spatial_attention, patch.squeeze(), both_path)
            print(f"Saved: {ch_path}, {sp_path}, {both_path}")

# --- USAGE: GENERATE ATTENTION MAPS/GRADCAM FOR ALL SCENES/CLASSES ---
for scene in SCENES:
    X_key = f"X_{scene}_test"
    y_key = f"y_{scene}_test"
    if X_key not in scene_data or y_key not in scene_data:
        print(f"No data for {scene}")
        continue
    X_split = scene_data[X_key]
    y_split = scene_data[y_key]
    for dev_idx in range(NUM_CLASSES):
        indices = np.where(y_split == dev_idx)[0]
        if len(indices) == 0:
            continue
        idx = indices[0]
        sample = X_split[idx]
        device_label = f"D{dev_idx+1:02}"
        gradcam_layer = get_last_conv_layer(model)
        gradcam_map = compute_gradcam(model, sample, gradcam_layer)
        print(f"Saving Grad-CAM for {device_label} in {scene} to {GRAD_DIR}")
        plot_spectrogram_and_gradcam(
            sample, gradcam_map,
            device_label=device_label,
            scenario=scene,
            sr=SR,
            hop_length=HOP_LENGTH,
            save_dir=GRAD_DIR
        )
print("✅ Grad-CAM images saved.")

extract_and_plot_cbam(model, X_test, y_test)
