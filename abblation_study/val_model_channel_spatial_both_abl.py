import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.preprocessing import StandardScaler

# Define path to save results
save_dir = ''
results_file = os.path.join(save_dir, "validation_accuracy_results.txt")

# Ensure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to split train/test equally per class
def split_equal_train_test(X_train, y_train, num_samples_per_class=10):
    X_test = np.array([X_train[0]])
    y_test = np.array([])

    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]
        selected_indices = np.random.choice(class_indices, size=num_samples_per_class, replace=False)
        X_test = np.append(X_test, X_train[selected_indices], axis=0)
        y_test = np.append(y_test, y_train[selected_indices], axis=0)
        X_train = np.delete(X_train, selected_indices, axis=0)
        y_train = np.delete(y_train, selected_indices, axis=0)

    X_test = np.delete(X_test, [0], axis=0)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test

# Load datasets
try:
    X_flat = np.load('./X_flat.npy')
    y_flat = np.load('./y_flat.npy') - 1

    X_indoor = np.load('./X_indoor.npy')
    y_indoor = np.load('./y_indoor.npy') - 1

    X_outdoor = np.load('./X_outdoor.npy')
    y_outdoor = np.load('./y_outdoor.npy') - 1

except FileNotFoundError as e:
    print(f"Error: {e}. Check file paths.")
    exit(1)

# Expand dimensions (for grayscale channel)
X_flat_expanded = np.expand_dims(X_flat, axis=-1)
X_indoor_expanded = np.expand_dims(X_indoor, axis=-1)
X_outdoor_expanded = np.expand_dims(X_outdoor, axis=-1)

# Resize spectrograms to 128x128
target_size = (128, 128)
X_flat_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_flat_expanded])
X_indoor_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_indoor_expanded])
X_outdoor_resized = np.array([img_to_array(array_to_img(x).resize(target_size)) for x in X_outdoor_expanded])

# Convert to grayscale (single channel)
X_flat_grayscale = np.mean(X_flat_resized, axis=-1, keepdims=True)
X_indoor_grayscale = np.mean(X_indoor_resized, axis=-1, keepdims=True)
X_outdoor_grayscale = np.mean(X_outdoor_resized, axis=-1, keepdims=True)

# Split into train, validation, test sets
X_flat_temp, X_flat_test, y_flat_temp, y_flat_test = split_equal_train_test(X_flat_grayscale, y_flat)
X_indoor_temp, X_indoor_test, y_indoor_temp, y_indoor_test = split_equal_train_test(X_indoor_grayscale, y_indoor)
X_outdoor_temp, X_outdoor_test, y_outdoor_temp, y_outdoor_test = split_equal_train_test(X_outdoor_grayscale, y_outdoor)

X_test = np.concatenate((X_flat_test, X_indoor_test, X_outdoor_test))
y_test = np.concatenate((y_flat_test, y_indoor_test, y_outdoor_test))

X_flat_train, X_flat_val, y_flat_train, y_flat_val = split_equal_train_test(X_flat_temp, y_flat_temp)
X_indoor_train, X_indoor_val, y_indoor_train, y_indoor_val = split_equal_train_test(X_indoor_temp, y_indoor_temp)
X_outdoor_train, X_outdoor_val, y_outdoor_train, y_outdoor_val = split_equal_train_test(X_outdoor_temp, y_outdoor_temp)

X_train = np.concatenate((X_flat_train, X_indoor_train, X_outdoor_train))
y_train = np.concatenate((y_flat_train, y_indoor_train, y_outdoor_train))

X_val = np.concatenate((X_flat_val, X_indoor_val, X_outdoor_val))
y_val = np.concatenate((y_flat_val, y_indoor_val, y_outdoor_val))

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1))).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape((X_val.shape[0], -1))).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)

# Standardize individual validation sets
X_val_flat_scaled = scaler.transform(X_flat_val.reshape((X_flat_val.shape[0], -1))).reshape(X_flat_val.shape)
X_val_indoor_scaled = scaler.transform(X_indoor_val.reshape((X_indoor_val.shape[0], -1))).reshape(X_indoor_val.shape)
X_val_outdoor_scaled = scaler.transform(X_outdoor_val.reshape((X_outdoor_val.shape[0], -1))).reshape(X_outdoor_val.shape)

# Load trained models
channel_model = load_model(os.path.join(save_dir, "best_model_channel_attention.h5"))
spatial_model = load_model(os.path.join(save_dir, "best_model_spatial_attention.h5"))
both_model = load_model(os.path.join(save_dir, "best_model_cbam.h5"))

# Function to evaluate model
def evaluate_model(model, X_val, y_val, scenario_name):
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"{scenario_name} Validation Accuracy: {accuracy:.4f}")
    return accuracy

# Evaluate models on overall validation set
channel_acc = evaluate_model(channel_model, X_val_scaled, y_val, "Channel Attention Overall")
spatial_acc = evaluate_model(spatial_model, X_val_scaled, y_val, "Spatial Attention Overall")
both_acc = evaluate_model(both_model, X_val_scaled, y_val, "CBAM Attention Overall")

# Evaluate models on individual scenarios (Flat, Indoor, Outdoor)
channel_acc_flat = evaluate_model(channel_model, X_val_flat_scaled, y_flat_val, "Channel Attention Flat")
spatial_acc_flat = evaluate_model(spatial_model, X_val_flat_scaled, y_flat_val, "Spatial Attention Flat")
both_acc_flat = evaluate_model(both_model, X_val_flat_scaled, y_flat_val, "CBAM Attention Flat")

channel_acc_indoor = evaluate_model(channel_model, X_val_indoor_scaled, y_indoor_val, "Channel Attention Indoor")
spatial_acc_indoor = evaluate_model(spatial_model, X_val_indoor_scaled, y_indoor_val, "Spatial Attention Indoor")
both_acc_indoor = evaluate_model(both_model, X_val_indoor_scaled, y_indoor_val, "CBAM Attention Indoor")

channel_acc_outdoor = evaluate_model(channel_model, X_val_outdoor_scaled, y_outdoor_val, "Channel Attention Outdoor")
spatial_acc_outdoor = evaluate_model(spatial_model, X_val_outdoor_scaled, y_outdoor_val, "Spatial Attention Outdoor")
both_acc_outdoor = evaluate_model(both_model, X_val_outdoor_scaled, y_outdoor_val, "CBAM Attention Outdoor")

# Save results to a text file
with open(results_file, "w") as f:
    f.write("Validation Accuracy Results:\n")
    f.write("------------------------------------\n")
    f.write(f"Channel Attention Overall: {channel_acc:.4f}\n")
    f.write(f"Spatial Attention Overall: {spatial_acc:.4f}\n")
    f.write(f"CBAM Attention Overall: {both_acc:.4f}\n\n")
    f.write(f"Channel Attention Flat: {channel_acc_flat:.4f}\n")
    f.write(f"Spatial Attention Flat: {spatial_acc_flat:.4f}\n")
    f.write(f"CBAM Attention Flat: {both_acc_flat:.4f}\n\n")

print(f"\nValidation accuracy results saved to {results_file}")
