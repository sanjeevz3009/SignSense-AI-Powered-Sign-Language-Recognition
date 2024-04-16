# For operating system related operations
import os
# For numerical computations
import numpy as np
# Importing necessary libraries for data processing, model building,
# and evaluation
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Custom modules are imported which contain predefined gestures to detect and data location
from gestures_to_detect import gestures, no_sequences, sequence_length
from utils import data_location

def load_data():
    """
    Loads gesture sequences and labels from the data directory and pre-processes them.
    """
    label_map = {label: num for num, label in enumerate(gestures)}
    sequences, labels = [], []
    
    for gesture in gestures:
        for sequence in range(no_sequences):
            window = [
                np.load(
                    os.path.join(data_location, gesture, str(sequence), f"{frame_count}.npy")
                )
                for frame_count in range(sequence_length)
            ]
            sequences.append(window)
            labels.append(label_map[gesture])
    
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return x, y

def build_model(input_shape, num_classes):
    """
     Defines and compiles the LSTM model architecture.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=input_shape),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, x_train, y_train, log_dir):
    """
    Train the model and log the training progress using TensorBoard.
    """
    tensor_board_callback = TensorBoard(log_dir=log_dir)
    model.fit(x_train, y_train, epochs=200, callbacks=[tensor_board_callback])
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model's performance on the test dataset and
    computes accuracy and confusion matrix.
    """
    yhat = model.predict(x_test)
    ytrue = np.argmax(y_test, axis=1)
    yhat = np.argmax(yhat, axis=1)
    confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
    accuracy = accuracy_score(ytrue, yhat)
    return confusion_matrix, accuracy

# The main block of code loads the data, splits it into training and
# testing sets, builds the model, trains it, saves the trained model,
# evaluates its performance, and prints the confusion matrix and accuracy
if __name__ == "__main__":
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    print("Training data shape:", x_train.shape)
    
    log_location = os.path.join("Logs")
    model = build_model(input_shape=(sequence_length, 126), num_classes=len(gestures))
    trained_model = train_model(model, x_train, y_train, log_location)
    trained_model.save("gestures_4.h5")
    
    confusion_matrix, accuracy = evaluate_model(trained_model, x_test, y_test)
    print("Confusion Matrix:\n", confusion_matrix)
    print("Accuracy:", accuracy)

# This script facilitates the training of an LSTM neural network model
# for hand gesture recognition using data collected from the MediaPipe
# library, and it provides functionalities for model evaluation and performance analysis.