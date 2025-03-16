import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from wfdb import get_record_list

# Enable mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# 1. Data Preprocessing
def load_mitbih_data():
    """Load and preprocess MIT-BIH dataset automatically from PhysioNet."""
    # Get list of MIT-BIH Arrhythmia Database record names
    record_names = get_record_list('mitdb')
    signals, labels = [], []
    for record_name in record_names:
        record = wfdb.rdrecord(f'mitdb/{record_name}')
        annotation = wfdb.rdann(f'mitdb/{record_name}', 'atr')
        signal = record.p_signal
        signal = bandpass_filter(signal)
        segmented_signals, segmented_labels = segment_signal(signal, annotation.sample)
        signals.extend(segmented_signals)
        labels.extend(segmented_labels)
    return np.array(signals), np.array(labels)


def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    """Apply a bandpass filter to remove noise."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def segment_signal(signal, annotations, window_size=3600, stride=1800):
    """Segment the signal into fixed-size windows."""
    segments, labels = [], []
    for start in range(0, len(signal) - window_size, stride):
        end = start + window_size
        segments.append(signal[start:end])
        label = get_majority_label(annotations, start, end)
        labels.append(label)
    return segments, labels


def get_majority_label(annotations, start, end):
    """Get the most frequent label in the given window."""
    return np.bincount(annotations[(annotations >= start) & (annotations < end)]).argmax()


# Normalize signals
def normalize_signals(signals):
    scaler = StandardScaler()
    signals = scaler.fit_transform(signals.reshape(-1, signals.shape[-1])).reshape(signals.shape)
    return signals


# 2. Handling Imbalanced Data with SMOTE
def apply_smote(X, y):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)  # Ensure k_neighbors <= n_samples
    # Check if there are enough samples before applying SMOTE
    class_counts = np.bincount(y.argmax(axis=1))  # Counts of samples per class
    if any(class_counts <= 1):  # Check if any class has <= 1 sample
        print("Warning: Some classes have insufficient samples, skipping SMOTE.")
        return X, y  # Return original data without applying SMOTE
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    return X_resampled.reshape(X_resampled.shape[0], X.shape[1], X.shape[2]), y_resampled


# 3. CNN + LSTM Model
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 4. Training the Model
# Load data from MIT-BIH Arrhythmia Database
X, y = load_mitbih_data()
X = normalize_signals(X)
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data to handle class imbalance
X_train, y_train = apply_smote(X_train, y_train)

# Create the CNN + LSTM model
model = create_cnn_lstm_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

# Define early stopping and model checkpoint callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model with a smaller batch size
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,  # Reduced batch size to fit in 4GB VRAM
    callbacks=callbacks
)

# 5. Evaluate the Model
eval_results = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {eval_results[1] * 100:.2f}%")