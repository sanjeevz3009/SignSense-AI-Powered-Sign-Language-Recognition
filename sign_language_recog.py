import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
from utils import mediapipe_detection, draw_landmarks_custom, extract_landmarks, prob_visualation, colours, data_location
from gestures_to_detect import gestures

# Holistic model
mediapipe_holistic = mp.solutions.holistic

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

model.load_weights("gestures.h5")

sequence = []
sentence = []
predictions = []
threshold = 0.95

capture = cv2.VideoCapture(0)
# Access/ set media pipe mode
with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        image = draw_landmarks_custom(image, results)

        get_key_points = extract_landmarks(results)
        sequence.append(get_key_points)
        sequence = sequence[-30:]

        # Prediction logic
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(gestures[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        # Visualation logic
            if np.unique(predictions[-10:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if gestures[np.argmax(res)] != sentence[-1]:
                            sentence.append(gestures[np.argmax(res)])
                    else:
                        sentence.append(gestures[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Probabilities
            image = prob_visualation(res, gestures, image, colours)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
