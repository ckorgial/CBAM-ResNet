import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function for equal class representation
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

# Set directories
save_dir = ''  # Update this path as needed
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load Floreview data
X_floreview = np.load('./X_FloreView.npy')
y_floreview = np.load('./y_FloreView.npy')

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
X_train_temp, X_test, y_train_temp, y_test = split_equal_train_test(X_floreview_grayscale, y_floreview, num_samples_per_class=10)

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

# Modify the output layer for 46 classes
for layer in pretrained_model.layers[:-1]:
    layer.trainable = False  # Freeze all layers except the last layer

x = pretrained_model.layers[-2].output  # Get the second-to-last layer output
outputs = tf.keras.layers.Dense(46, activation='softmax', name="dense_floreview_output")(x)
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


def create_confusion_matrix(X, y, save_path):
    # Get predictions
    y_pred = np.argmax(best_model.predict(X), axis=1)
    conf_matrix = confusion_matrix(y, y_pred)

    # Create labels from D01 to D46
    labels = [f'D{i:02}' for i in range(1, 47)]

    # Adjust the confusion matrix size to include all 46 labels
    full_conf_matrix = np.zeros((46, 46), dtype=int)
    for i, true_class in enumerate(np.unique(y)):
        for j, pred_class in enumerate(np.unique(y_pred)):
            full_conf_matrix[true_class, pred_class] = conf_matrix[i, j]

    # Plot the heatmap with all labels
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        full_conf_matrix,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,  # Include color bar
        annot_kws={"size": 12}  # Increase annotation size
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    # Save the figure with adjusted layout
    plt.savefig(save_path, bbox_inches='tight')  # Ensure labels are not clipped
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")


# Generate confusion matrix
create_confusion_matrix(
    X=X_test_scaled,
    y=y_test,
    save_path=os.path.join(save_dir, 'confusion_matrix_floreview_test.png')
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

print(f"Total samples: {X_floreview.shape[0]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

unique_classes, class_counts = np.unique(y_test, return_counts=True)
print(f"Classes in test set: {unique_classes}")
print(f"Counts per class: {dict(zip(unique_classes, class_counts))}")

