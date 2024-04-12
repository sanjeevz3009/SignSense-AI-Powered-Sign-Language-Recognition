import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
from utils import data_location
from gestures_to_detect import gestures, no_sequences, sequence_length

# Holistic model
mediapipe_holistic = mp.solutions.holistic

label_map = {label:num for num, label in enumerate(gestures)}

sequences, labels = [], []
for gesture in gestures:
    for sequence in range(no_sequences):
        window = []
        for frame_count in range(sequence_length):
            result = np.load(os.path.join(data_location, gesture, str(sequence), f"{frame_count}.npy"))
            window.append(result)
        sequences.append(window)
        labels.append(label_map[gesture])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_location = os.path.join("Logs")
tensor_board_callback = TensorBoard(log_dir=log_location)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))

model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(gestures.shape[0], activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.fit(x_train, y_train, epochs=1000, callbacks=[tensor_board_callback])

model.save("gestures_2.h5")

model.load_weights("gestures.h5")

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))
