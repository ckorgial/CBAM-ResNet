import os
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# --- CONFIGURATION ---
DATA_DIR = "./Splits"
SAVE_DIR = "./Results_Ablation"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_CLASSES = 35
SEED = 42
SCENES = ['flat', 'indoor', 'outdoor']
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

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
        for patch in extract_patches(spectro):
            X_patches.append(patch)
            y_patches.append(label)
    return np.stack(X_patches), np.array(y_patches)

def load_scene_data(scene):
    X_scene_raw = np.load(os.path.join(DATA_DIR, f"X_{scene}_val.npy"))
    y_scene_raw = np.load(os.path.join(DATA_DIR, f"y_{scene}_val.npy"))
    if y_scene_raw.max() == 35:
        y_scene_raw -= 1
    X_patches, y_patches = [], []
    for spec, label in zip(X_scene_raw, y_scene_raw):
        for patch in extract_patches(spec):
            X_patches.append(patch)
            y_patches.append(label)
    return np.stack(X_patches), np.array(y_patches)

# --- MODEL VARIANTS ---
def build_cbam_variant(input_shape, num_classes, use_channel=True, use_spatial=True):
    def cbam_block(x, filters, ratio=8, block_id=None):
        suffix = f"_{block_id}" if block_id is not None else ""
        if use_channel:
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
        if use_spatial:
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

# --- ROC PLOT OVO ---
def plot_roc_curve_ovo(model, X, y, prefix, save_dir=SAVE_DIR, max_splits=5):
    y_bin = label_binarize(y, classes=np.arange(NUM_CLASSES))
    unique, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    n_splits = min(max_splits, min_samples_per_class)
    if n_splits < 2:
        print(f"⚠️ Too few samples for ROC AUC calculation in split '{prefix}'. Skipping.")
        return float('nan'), 0, 0

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
    optimal_idx = np.argmax(mean_tpr - mean_fpr)
    optimal_fpr = mean_fpr[optimal_idx]
    optimal_tpr = mean_tpr[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Avg ROC (AUC = {mean_auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.scatter(optimal_fpr, optimal_tpr, c='red', s=70,
                label=f'Optimal (FPR={optimal_fpr:.2f}, TPR={optimal_tpr:.2f})')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"OvO ROC Curve - {prefix}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"roc_{prefix.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved ROC-AUC plot for '{prefix}' to: {save_path}")
    return mean_auc, optimal_fpr, optimal_tpr

# --- TRAINING AND EVALUATION ---
def train_and_evaluate(model_name, model_builder, X_train, y_train, X_val, y_val):
    print(f"\n🔧 Training {model_name}...")
    model = model_builder((128, 128, 1), NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, f"{model_name}.h5"), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=80, batch_size=32, class_weight=class_weights,
        callbacks=callbacks, verbose=2
    )
    return model

def evaluate_model(model, model_name, X_val, y_val):
    print(f"\n📊 Evaluating {model_name} on Validation...")
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    auc_val, _, _ = plot_roc_curve_ovo(model, X_val, y_val, f"{model_name}_Overall")
    results = [("Overall", val_acc, auc_val)]

    for scene in SCENES:
        X_scene, y_scene = load_scene_data(scene)
        y_scene_cat = tf.keras.utils.to_categorical(y_scene, NUM_CLASSES)
        _, acc = model.evaluate(X_scene, y_scene_cat, verbose=0)
        auc_score, _, _ = plot_roc_curve_ovo(model, X_scene, y_scene, f"{model_name}_{scene.capitalize()}")
        results.append((scene.capitalize(), acc, auc_score))
        print(f"🔹 {model_name} - {scene.capitalize()}: Accuracy = {acc:.2%}, AUC = {auc_score:.3f}")
    return results

# --- MAIN EXECUTION ---
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")

model_builders = {
    "cbam_channel_only": lambda s, n: build_cbam_variant(s, n, use_channel=True, use_spatial=False),
    "cbam_spatial_only": lambda s, n: build_cbam_variant(s, n, use_channel=False, use_spatial=True)
}

summary = []
for model_name, builder in model_builders.items():
    model = train_and_evaluate(model_name, builder, X_train, y_train, X_val, y_val)
    results = evaluate_model(model, model_name, X_val, y_val)
    for scene, acc, auc_val in results:
        summary.append({
            "Model": model_name,
            "Scene": scene,
            "Accuracy": round(acc * 100, 2),
            "AUC_OVO": round(auc_val, 4)
        })

# --- SAVE RESULTS ---
df = pd.DataFrame(summary)
df.to_csv(os.path.join(SAVE_DIR, "ablation_scene_accuracies_with_auc.csv"), index=False)
print("\n✅ Results saved to ablation_scene_accuracies_with_auc.csv")
