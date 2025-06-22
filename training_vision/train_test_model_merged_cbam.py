import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_DIR = "./Splits_Merged"
SAVE_DIR = "/Results_Merged"
os.makedirs(SAVE_DIR, exist_ok=True)

LABELS = [
    'D01-D26', 'D02-D10', 'D03', 'D05-D14-D18', 'D06-D15',
    'D07', 'D08', 'D09', 'D11', 'D13', 'D16', 'D19', 'D20',
    'D21', 'D23', 'D24', 'D25', 'D27', 'D28', 'D29-D34',
    'D30', 'D31', 'D32', 'D33', 'D35'
]

# Constants
PATCH_WIDTH = 128
PATCH_HEIGHT = 128
SEED = 42
NUM_CLASSES = len(LABELS)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


# --- CBAM ATTENTION MODULE ---
def cbam_block(x, filters, ratio=8, block_id=None):
    """Convolutional Block Attention Module (CBAM)"""
    suffix = f"_{block_id}" if block_id is not None else ""

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D(name=f"channel_avg_pool{suffix}")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"channel_max_pool{suffix}")(x)

    dense_1 = layers.Dense(filters // ratio, activation='relu', name=f"channel_dense_1{suffix}")
    dense_2 = layers.Dense(filters, name=f"channel_dense_2{suffix}")

    avg_out = dense_2(dense_1(avg_pool))
    max_out = dense_2(dense_1(max_pool))

    channel = layers.Add(name=f"channel_add{suffix}")([avg_out, max_out])
    channel = layers.Activation('sigmoid', name=f"channel_sigmoid{suffix}")(channel)
    channel = layers.Reshape((1, 1, filters), name=f"channel_reshape{suffix}")(channel)
    x = layers.Multiply(name=f"channel_multiply{suffix}")([x, channel])

    # Spatial attention
    avg_pool2 = tf.reduce_mean(x, axis=-1, keepdims=True, name=f"spatial_avg_pool{suffix}")
    max_pool2 = tf.reduce_max(x, axis=-1, keepdims=True, name=f"spatial_max_pool{suffix}")
    concat = layers.Concatenate(axis=-1, name=f"spatial_concat{suffix}")([avg_pool2, max_pool2])

    spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid',
                            name=f"spatial_conv{suffix}")(concat)
    x = layers.Multiply(name=f"spatial_multiply{suffix}")([x, spatial])

    return x


# --- RESIDUAL BLOCK WITH CBAM ---
def residual_block(x, filters, stride=1, block_id=None):
    """Residual block with CBAM attention"""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"res_{block_id}_conv1")(x)
    x = layers.BatchNormalization(name=f"res_{block_id}_bn1")(x)
    x = layers.ReLU(name=f"res_{block_id}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"res_{block_id}_conv2")(x)
    x = layers.BatchNormalization(name=f"res_{block_id}_bn2")(x)

    if shortcut.shape[-1] != filters or stride > 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4),
                                 name=f"res_{block_id}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"res_{block_id}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"res_{block_id}_add")([x, shortcut])
    x = layers.ReLU(name=f"res_{block_id}_relu2")(x)

    # Add CBAM attention
    x = cbam_block(x, filters, block_id=block_id)

    return x


# --- CBAM-RESNET ARCHITECTURE ---
def build_cbam_resnet(input_shape, num_classes):
    """Build ResNet with CBAM attention modules"""
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Initial stem
    x = layers.Conv2D(32, 7, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="stem_pool")(x)

    # Residual blocks with CBAM
    x = residual_block(x, 32, block_id=1)
    x = residual_block(x, 32, block_id=2)

    x = residual_block(x, 64, stride=2, block_id=3)
    x = residual_block(x, 64, block_id=4)

    x = residual_block(x, 128, stride=2, block_id=5)
    x = residual_block(x, 128, block_id=6)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="final_pool")(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3),
                     name="final_dense1")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Dropout(0.8, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    return models.Model(inputs, outputs, name="CBAM_ResNet")


# --- DATA LOADING ---
def extract_patches(spectrogram, patch_height=128, patch_width=128):
    if spectrogram.ndim == 3:
        spectrogram = spectrogram.squeeze(0)
    patches = []
    for start in range(0, spectrogram.shape[1] - patch_width + 1, patch_width):
        patch = spectrogram[:patch_height, start:start + patch_width]
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-6)
        patch = patch[..., np.newaxis]
        patches.append(patch)
    return patches


def load_data(name):
    X_raw = np.load(os.path.join(DATA_DIR, f"X_{name}.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    X_patches, y_patches = [], []
    for spectro, label in zip(X_raw, y_raw):
        patches = extract_patches(spectro)
        X_patches.extend(patches)
        y_patches.extend([label] * len(patches))
    return np.stack(X_patches), np.array(y_patches)


# Load data
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

# --- LABEL PROCESSING ---
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# --- MODEL COMPILATION ---
model = build_cbam_resnet((128, 128, 1), NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

# --- TRAINING ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# --- BLUE-STYLE CONFUSION MATRIX ---
def create_confusion_matrix(X, y, label):
    """Create and save blue-style confusion matrix"""
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    cm = confusion_matrix(y, y_pred, labels=np.arange(NUM_CLASSES))

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot_kws={'size': 8}
    )

    accuracy = np.trace(cm) / np.sum(cm)
    plt.title(f'Confusion Matrix - Test Overall', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, f'confusion_matrix_{label.lower().replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ Saved confusion matrix for {label} | Accuracy: {accuracy:.2%}')


# --- EVALUATION ---
def evaluate_model(X, y, label):
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = np.mean(y_pred_classes == y)
    report = classification_report(y, y_pred_classes, target_names=LABELS, output_dict=True)

    pd.DataFrame(report).transpose().to_csv(os.path.join(SAVE_DIR, f'{label}_report.csv'))
    print(f'\n{label} Classification Report:')
    print(classification_report(y, y_pred_classes, target_names=LABELS))

    create_confusion_matrix(X, y, label)
    return accuracy


# Run evaluation
train_acc = evaluate_model(X_train, y_train, "Train")
val_acc = evaluate_model(X_val, y_val, "Validation")
test_acc = evaluate_model(X_test, y_test, "Test")

# Save training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
plt.close()

print('\n✅ Training complete! Results saved to:', SAVE_DIR)
print(f'Final Accuracies: Train={train_acc:.2%}, Val={val_acc:.2%}, Test={test_acc:.2%}')
