import numpy as np
from tensorflow.keras.models import load_model
from statsmodels.stats.contingency_tables import mcnemar
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.preprocessing import StandardScaler

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
X_flat = np.load('./X_flat.npy')
y_flat = np.load('./y_flat.npy')

y_flat = y_flat - 1

X_indoor = np.load('./X_indoor.npy')
y_indoor = np.load('./y_indoor.npy')

y_indoor = y_indoor - 1

X_outdoor = np.load('./X_outdoor.npy')
y_outdoor = np.load('./y_outdoor.npy')

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
X_test_scaled = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)

# Load your models
best_model = load_model('./best_model.h5')
model1 = load_model('./best_model.h5')
model2 = load_model('./best_model.h5')

# Generate predictions for the validation set
y_pred_proposed = np.argmax(best_model.predict(X_val_scaled), axis=1)
y_pred_method1 = np.argmax(model1.predict(X_val_scaled), axis=1)
y_pred_method2 = np.argmax(model2.predict(X_val_scaled), axis=1)

# Generate predictions for each scenario
y_pred_flat_proposed = np.argmax(best_model.predict(X_flat_val), axis=1)
y_pred_flat_method1 = np.argmax(model1.predict(X_flat_val), axis=1)
y_pred_flat_method2 = np.argmax(model2.predict(X_flat_val), axis=1)

y_pred_indoor_proposed = np.argmax(best_model.predict(X_indoor_val), axis=1)
y_pred_indoor_method1 = np.argmax(model1.predict(X_indoor_val), axis=1)
y_pred_indoor_method2 = np.argmax(model2.predict(X_indoor_val), axis=1)

y_pred_outdoor_proposed = np.argmax(best_model.predict(X_outdoor_val), axis=1)
y_pred_outdoor_method1 = np.argmax(model1.predict(X_outdoor_val), axis=1)
y_pred_outdoor_method2 = np.argmax(model2.predict(X_outdoor_val), axis=1)

# Create the contingency tables for McNemar's test
def get_contingency_table(y_true, y_pred_1, y_pred_2):
    both_correct = np.sum((y_pred_1 == y_true) & (y_pred_2 == y_true))
    both_incorrect = np.sum((y_pred_1 != y_true) & (y_pred_2 != y_true))
    method1_correct_method2_incorrect = np.sum((y_pred_1 == y_true) & (y_pred_2 != y_true))
    method2_correct_method1_incorrect = np.sum((y_pred_2 == y_true) & (y_pred_1 != y_true))

    return np.array([[both_correct, method1_correct_method2_incorrect],
                     [method2_correct_method1_incorrect, both_incorrect]])

# Define the significance level
significance_level = 0.005

# Function to perform McNemar's test and return the p-value
def perform_mcnemar_test(y_true, y_pred_1, y_pred_2):
    contingency_table = get_contingency_table(y_true, y_pred_1, y_pred_2)
    result = mcnemar(contingency_table, exact=True)
    return result.pvalue

# Calculate p-values for each scenario
p_values = {
    "Overall": {
        "Proposed vs Method 1": perform_mcnemar_test(y_val, y_pred_proposed, y_pred_method1),
        "Proposed vs Method 2": perform_mcnemar_test(y_val, y_pred_proposed, y_pred_method2)
    },
    "Flat": {
        "Proposed vs Method 1": perform_mcnemar_test(y_flat_val, y_pred_flat_proposed, y_pred_flat_method1),
        "Proposed vs Method 2": perform_mcnemar_test(y_flat_val, y_pred_flat_proposed, y_pred_flat_method2)
    },
    "Indoor": {
        "Proposed vs Method 1": perform_mcnemar_test(y_indoor_val, y_pred_indoor_proposed, y_pred_indoor_method1),
        "Proposed vs Method 2": perform_mcnemar_test(y_indoor_val, y_pred_indoor_proposed, y_pred_indoor_method2)
    },
    "Outdoor": {
        "Proposed vs Method 1": perform_mcnemar_test(y_outdoor_val, y_pred_outdoor_proposed, y_pred_outdoor_method1),
        "Proposed vs Method 2": perform_mcnemar_test(y_outdoor_val, y_pred_outdoor_proposed, y_pred_outdoor_method2)
    }
}

# Print p-values for each scenario
print("P-values for each scenario:")

# Overall
print("Overall - Proposed vs Method 1: {:.5f}".format(p_values["Overall"]["Proposed vs Method 1"]))
print("Overall - Proposed vs Method 2: {:.5f}".format(p_values["Overall"]["Proposed vs Method 2"]))

# Flat
print("Flat - Proposed vs Method 1: {:.5f}".format(p_values["Flat"]["Proposed vs Method 1"]))
print("Flat - Proposed vs Method 2: {:.5f}".format(p_values["Flat"]["Proposed vs Method 2"]))

# Indoor
print("Indoor - Proposed vs Method 1: {:.5f}".format(p_values["Indoor"]["Proposed vs Method 1"]))
print("Indoor - Proposed vs Method 2: {:.5f}".format(p_values["Indoor"]["Proposed vs Method 2"]))

# Outdoor
print("Outdoor - Proposed vs Method 1: {:.5f}".format(p_values["Outdoor"]["Proposed vs Method 1"]))
print("Outdoor - Proposed vs Method 2: {:.5f}".format(p_values["Outdoor"]["Proposed vs Method 2"]))
