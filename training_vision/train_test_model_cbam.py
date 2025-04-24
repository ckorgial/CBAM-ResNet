import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Set GPU memory fraction
gpu_fraction = 0.5
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Directory to save the models and reports
save_dir = '/media/blue/ckorgial/XAI_VISION/Results_test_cbam_2/'

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def split_equal_train_test(X_train, y_train, num_samples_per_class=10):
    X_test = np.array([X_train[0]])
    y_test = np.array([])

    # Find indices for each class in the training set
    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]

        # Randomly select specified number of samples for each class and move them to the testing set
        selected_indices = np.random.choice(class_indices, size=num_samples_per_class, replace=False)
        X_test = np.append(X_test, X_train[selected_indices], axis=0)
        y_test = np.append(y_test, y_train[selected_indices], axis=0)
        X_train = np.delete(X_train, selected_indices, axis=0)
        y_train = np.delete(y_train, selected_indices, axis=0)

    X_test = np.delete(X_test, [0], axis=0)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test

# Load your data (replace 'your_X_file.npy' and 'your_y_file.npy' with your actual file paths)
X_flat = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/X_flat.npy')
y_flat = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/y_flat.npy')

y_flat = y_flat - 1

X_indoor = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/X_indoor.npy')
y_indoor = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/y_indoor.npy')

y_indoor = y_indoor - 1

X_outdoor = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/X_outdoor.npy')
y_outdoor = np.load('/media/blue/ckorgial/XAI_VISION/VISION_train/y_outdoor.npy')

y_outdoor = y_outdoor - 1

# Expand the dimensions to include a channel (assuming your original spectrograms have shape (128, 300))
X_flat_expanded = np.expand_dims(X_flat, axis=-1)
X_indoor_expanded = np.expand_dims(X_indoor, axis=-1)
X_outdoor_expanded = np.expand_dims(X_outdoor, axis=-1)

# Resize spectrograms to meet ResNet input size (128x128)
target_size = (128, 128)
X_flat_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_flat_expanded])
X_indoor_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_indoor_expanded])
X_outdoor_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_outdoor_expanded])

# Convert spectrograms to single-channel (grayscale)
X_flat_grayscale = np.mean(X_flat_resized, axis=-1, keepdims=True)
X_indoor_grayscale = np.mean(X_indoor_resized, axis=-1, keepdims=True)
X_outdoor_grayscale = np.mean(X_outdoor_resized, axis=-1, keepdims=True)

# Split the data into training, validation, and test sets
X_flat_temp, X_flat_test, y_flat_temp, y_flat_test = split_equal_train_test(X_flat_grayscale, y_flat, num_samples_per_class=10)
X_indoor_temp, X_indoor_test, y_indoor_temp, y_indoor_test = split_equal_train_test(X_indoor_grayscale, y_indoor, num_samples_per_class=10)
X_outdoor_temp, X_outdoor_test, y_outdoor_temp, y_outdoor_test = split_equal_train_test(X_outdoor_grayscale, y_outdoor, num_samples_per_class=10)
X_test = np.concatenate((X_flat_test, X_indoor_test, X_outdoor_test))
y_test = np.concatenate((y_flat_test, y_indoor_test, y_outdoor_test))

X_flat_train, X_flat_val, y_flat_train, y_flat_val = split_equal_train_test(X_flat_temp, y_flat_temp, num_samples_per_class=10)
X_indoor_train, X_indoor_val, y_indoor_train, y_indoor_val = split_equal_train_test(X_indoor_temp, y_indoor_temp, num_samples_per_class=10)
X_outdoor_train, X_outdoor_val, y_outdoor_train, y_outdoor_val = split_equal_train_test(X_outdoor_temp, y_outdoor_temp, num_samples_per_class=10)
X_train = np.concatenate((X_flat_train, X_indoor_train, X_outdoor_train))
y_train = np.concatenate((y_flat_train, y_indoor_train, y_outdoor_train))

X_val = np.concatenate((X_flat_val, X_indoor_val, X_outdoor_val))
y_val = np.concatenate((y_flat_val, y_indoor_val, y_outdoor_val))

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1))).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape((X_val.shape[0], -1))).reshape(X_val.shape)
X_val_flat_scaled = scaler.transform(X_flat_val.reshape((X_flat_val.shape[0], -1))).reshape(X_flat_val.shape)
X_val_indoor_scaled = scaler.transform(X_indoor_val.reshape((X_indoor_val.shape[0], -1))).reshape(X_indoor_val.shape)
X_val_outdoor_scaled = scaler.transform(X_outdoor_val.reshape((X_outdoor_val.shape[0], -1))).reshape(X_outdoor_val.shape)

X_test_scaled = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)
X_test_flat_scaled = scaler.transform(X_flat_test.reshape((X_flat_test.shape[0], -1))).reshape(X_flat_test.shape)
X_test_indoor_scaled = scaler.transform(X_indoor_test.reshape((X_indoor_test.shape[0], -1))).reshape(X_indoor_test.shape)
X_test_outdoor_scaled = scaler.transform(X_outdoor_test.reshape((X_outdoor_test.shape[0], -1))).reshape(X_outdoor_test.shape)

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    scale = layers.Add()([avg_pool, max_pool])
    scale = layers.Activation('sigmoid')(scale)

    return layers.Multiply()([input_feature, scale])


def spatial_attention(input_feature):
    kernel_size = 7
    if input_feature.shape[-2] < kernel_size or input_feature.shape[-3] < kernel_size:
        kernel_size = 3  # Adjust the kernel size to the shape of input feature map

    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                                 kernel_initializer='he_normal', use_bias=False)(concat)

    return layers.Multiply()([input_feature, cbam_feature])


def build_resnet_cbam(input_shape, nof_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    for _ in range(3):
        identity = x
        x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (1, 1), activation=None, padding='same')(x)  # Identity mapping
        identity = layers.Conv2D(256,(1, 1), activation=None, padding='same')(identity)
        x = layers.Add()([x, identity])
        x = layers.Activation('relu')(x)
        x = channel_attention(x)
        x = spatial_attention(x)

    # Global average pooling and fully connected layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(nof_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    # model.summary()
    return model

# Build the ResNet model
input_shape = (target_size[0], target_size[1], 1)  # Single-channel input
resnet_model = build_resnet_cbam(input_shape, 35)

# Plot and save the model architecture
plot_model(resnet_model, to_file=os.path.join(save_dir, 'resnet_model.png'), show_shapes=True, show_layer_names=True)

# Compile the model
learning_rate = 1e-3
beta_1 = 0.89
beta_2 = 0.98

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

resnet_model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Checkpoint to save the best model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model with ModelCheckpoint
history = resnet_model.fit(X_train_scaled, y_train, epochs=80
                           , validation_data=(X_val_scaled, y_val), callbacks=[model_checkpoint])

# Load the best model
best_model = tf.keras.models.load_model(os.path.join(save_dir, 'best_model.h5'))

# Evaluate the model on the test set
val_loss, val_acc = best_model.evaluate(X_val_scaled, y_val)
print(f'Overall Validation Accuracy: {val_acc * 100:.2f}% on {X_val_scaled.shape[0]} samples')
val_loss, val_acc = best_model.evaluate(X_val_flat_scaled, y_flat_val)
print(f'Flat Validation Accuracy: {val_acc * 100:.2f}% on {X_val_flat_scaled.shape[0]} samples')
val_loss, val_acc = best_model.evaluate(X_val_indoor_scaled, y_indoor_val)
print(f'Indoor Validation Accuracy: {val_acc * 100:.2f}% on {X_val_indoor_scaled.shape[0]} samples')
val_loss, val_acc = best_model.evaluate(X_val_outdoor_scaled, y_outdoor_val)
print(f'Outdoor Validation Accuracy: {val_acc * 100:.2f}% on {X_val_outdoor_scaled.shape[0]} samples')

# Function to create confusion matrix and save the plot
def create_confusion_matrix(X, y, title, save_path, labels_prefix):
    y_pred = np.argmax(best_model.predict(X), axis=1)
    conf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'{labels_prefix}{i:02}' for i in range(1, 36)],
                yticklabels=[f'{labels_prefix}{i:02}' for i in range(1, 36)])

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

# Create confusion matrices for different scenarios
create_confusion_matrix(X_val_scaled, y_val, 'Confusion Matrix for Overall Validation Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_val_overall.png'), 'D')
create_confusion_matrix(X_val_flat_scaled, y_flat_val, 'Confusion Matrix for Flat Validation Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_val_flat.png'), 'D')
create_confusion_matrix(X_val_indoor_scaled, y_indoor_val, 'Confusion Matrix for Indoor Validation Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_val_indoor.png'), 'D')
create_confusion_matrix(X_val_outdoor_scaled, y_outdoor_val, 'Confusion Matrix for Outdoor Validation Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_val_outdoor.png'), 'D')

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test)
print(f'Overall Test Accuracy: {test_acc * 100:.2f}% on {X_test_scaled.shape[0]} samples')
test_loss, test_acc = best_model.evaluate(X_test_flat_scaled, y_flat_test)
print(f'Flat Test Accuracy: {test_acc * 100:.2f}% on {X_test_flat_scaled.shape[0]} samples')
test_loss, test_acc = best_model.evaluate(X_test_indoor_scaled, y_indoor_test)
print(f'Indoor Test Accuracy: {test_acc * 100:.2f}% on {X_test_indoor_scaled.shape[0]} samples')
test_loss, test_acc = best_model.evaluate(X_test_outdoor_scaled, y_outdoor_test)
print(f'Outdoor Test Accuracy: {test_acc * 100:.2f}% on {X_test_outdoor_scaled.shape[0]} samples')

# Function to create confusion matrix and save the plot
def create_confusion_matrix(X, y, title, save_path, labels_prefix):
    y_pred = np.argmax(best_model.predict(X), axis=1)
    conf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'{labels_prefix}{i:02}' for i in range(1, 36)],
                yticklabels=[f'{labels_prefix}{i:02}' for i in range(1, 36)])

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

    # Save confusion matrix as CSV
    df_conf_matrix = pd.DataFrame(conf_matrix, index=[f'{labels_prefix}{i:02}' for i in range(1, 36)],
                                  columns=[f'{labels_prefix}{i:02}' for i in range(1, 36)])
    df_conf_matrix.to_csv(os.path.splitext(save_path)[0] + '.csv')

# Create confusion matrices for different scenarios
create_confusion_matrix(X_test_scaled, y_test, 'Confusion Matrix for Overall Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_test_overall.png'), 'D')
create_confusion_matrix(X_test_flat_scaled, y_flat_test, 'Confusion Matrix for Flat Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_test_flat.png'), 'D')
create_confusion_matrix(X_test_indoor_scaled, y_indoor_test, 'Confusion Matrix for Indoor Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_test_indoor.png'), 'D')
create_confusion_matrix(X_test_outdoor_scaled, y_outdoor_test, 'Confusion Matrix for Overall Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_test_outdoor.png'), 'D')

# Save the accuracy plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
plt.close()

# Compute AUC for the test set
def compute_multiclass_auc(model, X, y, n_classes, labels_prefix, save_path):
    # One-hot encode the true labels
    y_one_hot = label_binarize(y, classes=range(n_classes))

    # Get the predicted probabilities
    y_pred_prob = model.predict(X)

    # Compute AUC for each class
    aucs = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_one_hot[:, i], y_pred_prob[:, i])
            aucs.append(auc)
        except ValueError:
            # If AUC computation fails (e.g., if a class is not present in the data)
            aucs.append(float('nan'))

    # Average AUC across all classes
    avg_auc = np.nanmean(aucs)

    print(f"Average AUC for {labels_prefix} data: {avg_auc:.4f}")

    # Save AUC scores for each class
    auc_df = pd.DataFrame({"Class": [f"{labels_prefix}{i:02}" for i in range(n_classes)], "AUC": aucs})
    auc_df.to_csv(save_path, index=False)
    print(f"AUC results saved to: {save_path}")


# Define the number of classes
num_classes = 35

# Compute and save AUC for test data
compute_multiclass_auc(best_model, X_val_scaled, y_val, num_classes, 'Overall',
                       os.path.join(save_dir, 'auc_overall_val_results.csv'))
compute_multiclass_auc(best_model, X_val_flat_scaled, y_flat_val, num_classes, 'Flat',
                       os.path.join(save_dir, 'auc_flat_val_results.csv'))
compute_multiclass_auc(best_model, X_val_indoor_scaled, y_indoor_val, num_classes, 'Indoor',
                       os.path.join(save_dir, 'auc_indoor_val_results.csv'))
compute_multiclass_auc(best_model, X_val_outdoor_scaled, y_outdoor_val, num_classes, 'Outdoor',
                       os.path.join(save_dir, 'auc_outdoor_val_results.csv'))


# Define the number of classes and class names
num_classes = 35
class_names = [f'D{i+1}' for i in range(num_classes)]  # Generates names from D1 to D35

# One-hot encode the validation labels
y_val_one_hot = label_binarize(y_val, classes=range(num_classes))

# Predict probabilities for the validation data
y_pred_prob_val = best_model.predict(X_val_scaled)

# Calculate AUC for each class
auc_values = []
plt.figure(figsize=(16, 12))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_val_one_hot[:, i], y_pred_prob_val[:, i])
    roc_auc = auc(fpr, tpr)
    auc_values.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
# Calculate the overall multiclass AUC as the mean of the individual AUCs
overall_multiclass_auc = np.mean(auc_values)
# Plot the random guess line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
# Add titles and labels
plt.title(f'ROC Curves for the Overall Validation Data (One-vs-All)\n Multiclass AUC = {overall_multiclass_auc:.3f}', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right', fontsize=12, ncol=2)  # Adjust ncol for compact legend
plt.grid(alpha=0.3)
plt.tight_layout()
# Show the plot
plt.show()

# Print the exact number of samples in each set
print(f"Number of samples in the Training Set: {X_train_scaled.shape[0]} (Classes: {np.unique(y_train, return_counts=True)})")
print(f"Number of samples in the Validation Set: {X_val_scaled.shape[0]} (Classes: {np.unique(y_val, return_counts=True)})")
print(f"Number of samples in the Test Set: {X_test_scaled.shape[0]} (Classes: {np.unique(y_test, return_counts=True)})")

# Print class distribution for specific splits
print(f"Class distribution in Training Set: {np.unique(y_train, return_counts=True)}")
print(f"Class distribution in Validation Set: {np.unique(y_val, return_counts=True)}")
print(f"Class distribution in Test Set: {np.unique(y_test, return_counts=True)}")


# Print a message indicating the save path
print(f'Models and reports saved in: {save_dir}')
