import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function for equal class representation and ensure Xiaomi is included
def split_equal_train_test(X_train, y_train, num_samples_per_class=40):
    """
    Ensure an equal representation of all classes in the test set with a fixed number of samples per class.
    """
    X_test = []
    y_test = []

    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]

        # If not enough samples, use all available samples
        num_samples = min(len(class_indices), num_samples_per_class)
        selected_indices = np.random.choice(class_indices, size=num_samples, replace=False)

        X_test.append(X_train[selected_indices])
        y_test.append(y_train[selected_indices])

        X_train = np.delete(X_train, selected_indices, axis=0)
        y_train = np.delete(y_train, selected_indices, axis=0)

    # Concatenate all test samples
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Ensure all classes, including Xiaomi, are represented
    for missing_class in range(len(brand_names)):
        if missing_class not in y_test:
            print(f"Class {missing_class} missing in test set. Forcing inclusion.")
            missing_class_indices = np.where(y_train == missing_class)[0]
            if len(missing_class_indices) > 0:
                forced_indices = np.random.choice(missing_class_indices, size=1, replace=False)
                X_test = np.append(X_test, X_train[forced_indices], axis=0)
                y_test = np.append(y_test, y_train[forced_indices], axis=0)
                X_train = np.delete(X_train, forced_indices, axis=0)
                y_train = np.delete(y_train, forced_indices, axis=0)

    return X_train, X_test, y_train, y_test


def create_confusion_matrix(X, y, save_path, brand_names):
    """
    Generate and save the confusion matrix with brand names.
    """
    # Get predictions
    y_pred = np.argmax(best_model.predict(X), axis=1)

    # Ensure Xiaomi (and all classes) are included as labels
    conf_matrix = confusion_matrix(y, y_pred, labels=np.arange(len(brand_names)))

    # Debugging: Check predictions for Xiaomi
    xiaomi_indices = np.where(y == brand_names.index("Xiaomi"))[0]
    if len(xiaomi_indices) > 0:
        xiaomi_predictions = y_pred[xiaomi_indices]
        print(f"Predicted labels for Xiaomi samples: {xiaomi_predictions}")
    else:
        print("No Xiaomi samples in the test set.")

    # Plot the heatmap with all labels
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=brand_names,
        yticklabels=brand_names
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")


def create_confusion_matrix_with_scores(X, y, save_path, brand_names):
    """
    Generate and save the confusion matrix with scores (percentages) instead of counts.
    """
    # Get predictions
    y_pred = np.argmax(best_model.predict(X), axis=1)

    # Ensure all classes are included in the matrix
    conf_matrix = confusion_matrix(y, y_pred, labels=np.arange(len(brand_names)))

    # Convert to scores (percentages)
    conf_matrix_scores = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Replace NaN values with 0 (caused by division by zero for missing classes)
    conf_matrix_scores = np.nan_to_num(conf_matrix_scores)

    # Plot the heatmap with scores
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        conf_matrix_scores,
        annot=True,
        fmt='.2f',  # Format as decimal with 2 precision
        cmap="Blues",
        xticklabels=brand_names,
        yticklabels=brand_names
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Scores)', fontsize=14)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix with scores saved at: {save_path}")


# Set directories
save_dir = ''
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load Floreview data
X_floreview = np.load('./X_FloreView_brand.npy')
y_floreview = np.load('./y_FloreView_brand.npy')

# Define brand names
brand_names = ["Apple", "Google", "Huawei", "LG", "Motorola", "OnePlus", "Samsung", "Xiaomi"]

# Preprocess the labels (ensure they start at 0)
y_floreview = y_floreview - y_floreview.min()

# Expand dimensions for single-channel data
X_floreview_expanded = np.expand_dims(X_floreview, axis=-1)

# Resize to match the input shape required by ResNet
target_size = (128, 128)
X_floreview_resized = np.array([
    tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.array_to_img(x).resize(target_size)
    ) for x in X_floreview_expanded
])
X_floreview_grayscale = np.mean(X_floreview_resized, axis=-1, keepdims=True)

# Use the custom function to create an equal train-test split
X_train_temp, X_test, y_train_temp, y_test = split_equal_train_test(
    X_floreview_grayscale, y_floreview, num_samples_per_class=40
)

# Split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.2, stratify=y_train_temp
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1))).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape((X_val.shape[0], -1))).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)

# Load pre-trained VISION model
pretrained_model = tf.keras.models.load_model('/media/blue/ckorgial/XAI_VISION/Results_test_cbam/best_model.h5')

# Modify the output layer for 8 classes
for layer in pretrained_model.layers[:-1]:
    layer.trainable = False  # Freeze all layers except the last layer

x = pretrained_model.layers[-2].output  # Get the second-to-last layer output
outputs = tf.keras.layers.Dense(8, activation='softmax', name="dense_floreview_output")(x)
floreview_model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=outputs)

# Compile the modified model
floreview_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# Model checkpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, 'best_floreview_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
history = floreview_model.fit(
    X_train_scaled, y_train,
    epochs=30,
    validation_data=(X_val_scaled, y_val),
    callbacks=[model_checkpoint]
)

# Load the best model
best_model = tf.keras.models.load_model(os.path.join(save_dir, 'best_floreview_model.h5'))

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test)
print(f'Floreview Test Accuracy: {test_acc * 100:.2f}%')

# Generate confusion matrix
create_confusion_matrix(
    X=X_test_scaled,
    y=y_test,
    save_path=os.path.join(save_dir, 'confusion_matrix_floreview_test.png'),
    brand_names=brand_names
)

create_confusion_matrix_with_scores(
    X=X_test_scaled,
    y=y_test,
    save_path=os.path.join(save_dir, 'confusion_matrix_scores_floreview_test.png'),
    brand_names=brand_names
)

# Plot accuracy trends
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Trends')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
plt.close()

print(f"Models and reports saved in: {save_dir}")

# Verify test set distribution
unique_classes, class_counts = np.unique(y_test, return_counts=True)
print(f"Classes in test set: {unique_classes}")
print(f"Counts per class: {dict(zip(unique_classes, class_counts))}")
print(f"Total samples: {X_floreview.shape[0]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
