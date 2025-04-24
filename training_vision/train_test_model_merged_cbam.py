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

# Set GPU memory fraction
gpu_fraction = 0.5
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Directory to save the models and reports
save_dir = ''

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def split_equal_train_test(X_train, y_train, num_samples_per_class=30):
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
X = np.load('./X.npy')
y = np.load('./y.npy')

# y = y - 1

# Expand the dimensions to include a channel (assuming your original spectrograms have shape (128, 300))
X_expanded = np.expand_dims(X, axis=-1)

# Resize spectrograms to meet ResNet50 input size (128x128)
target_size = (128, 128)
X_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_expanded])

# Convert spectrograms to single-channel (grayscale)
X_grayscale = np.mean(X_resized, axis=-1, keepdims=True)

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = split_equal_train_test(X_grayscale, y, num_samples_per_class=30)
X_train, X_val, y_train, y_val = split_equal_train_test(X_temp, y_temp, num_samples_per_class=30)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1))).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape((X_val.shape[0], -1))).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)

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
resnet_model = build_resnet_cbam(input_shape, 25)

# Plot and save the model architecture
plot_model(resnet_model, to_file=os.path.join(save_dir, 'resnet50_model.png'), show_shapes=True, show_layer_names=True)

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
history = resnet_model.fit(X_train_scaled, y_train, epochs=80, validation_data=(X_val_scaled, y_val), callbacks=[model_checkpoint])

# Load the best model
best_model = tf.keras.models.load_model(os.path.join(save_dir, 'best_model.h5'))

# Evaluate the model on the test set
val_loss, val_acc = best_model.evaluate(X_val_scaled, y_val)
print(f'Overall Validation Accuracy: {val_acc * 100:.2f}% on {X_val_scaled.shape[0]} samples')

# Function to create confusion matrix and save the plot
def create_confusion_matrix(X, y, title, save_path, labels_prefix):
    y_pred = np.argmax(best_model.predict(X), axis=1)
    conf_matrix = confusion_matrix(y, y_pred)

    # Define custom labels based on the provided template
    custom_labels = [
        'D01-D26','D02-D10','D03','D05-D14-D18','D06-D15','D07','D08','D09','D11','D13','D16','D19','D20','D21','D23',
        'D24','D25','D27','D28','D29-D34','D30','D31','D32','D33','D35']

    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=custom_labels,
                yticklabels=custom_labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

# Create confusion matrices for different scenarios
create_confusion_matrix(X_val_scaled, y_val, 'Confusion Matrix for Overall Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_val_overall.png'), 'D')

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test)
print(f'Overall Test Accuracy: {test_acc * 100:.2f}% on {X_test_scaled.shape[0]} samples')

# Function to create confusion matrix and save the plot
def create_confusion_matrix(X, y, title, save_path, labels_prefix):
    y_pred = np.argmax(best_model.predict(X), axis=1)
    conf_matrix = confusion_matrix(y, y_pred)

    # Define custom labels based on the provided template
    custom_labels = [
        'D01-D26','D02-D10','D03','D05-D14-D18','D06-D15','D07','D08','D09','D11','D13','D16','D19','D20','D21','D23',
        'D24','D25','D27','D28','D29-D34','D30','D31','D32','D33','D35']

    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=custom_labels,
                yticklabels=custom_labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

    # Save confusion matrix as CSV
    df_conf_matrix = pd.DataFrame(conf_matrix, index=custom_labels, columns=custom_labels)
    df_conf_matrix.to_csv(os.path.splitext(save_path)[0] + '.csv')

# Create confusion matrices for different scenarios
create_confusion_matrix(X_test_scaled, y_test, 'Confusion Matrix for Overall Test Data',
                        os.path.join(save_dir, 'confusion_matrix_plot_test_overall.png'), 'D')


# Save the accuracy plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
plt.close()

# Print a message indicating the save path
print(f'Models and reports saved in: {save_dir}')
