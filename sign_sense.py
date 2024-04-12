import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Holistic model
mediapipe_holistic = mp.solutions.holistic
# Drawing utilities
mediapipe_draw = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # Colour conversion BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Image will not be writeable
    image.flags.writeable = False
    # Detecting image using media pipe/ make prediction
    results = model.process(image)
    # Image will be writeable now
    image.flags.writeable = True
    # Colour conversion BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks_custom(image, results):
    # Draw face landmarks
    # Can use FACEMESH_CONTOURS or FACEMESH_TESSELATION
    # Make the colour customisations as variables so it can be adjusted
    mediapipe_draw.draw_landmarks(image, results.face_landmarks, mediapipe_holistic.FACEMESH_CONTOURS,
                                  mediapipe_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mediapipe_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
    # Draw left hand landmarks
    mediapipe_draw.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                                  mediapipe_draw.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
                                  )
    # Draw right hand landmarks
    mediapipe_draw.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                  mediapipe_draw.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
                                  )
    # Draw pose landmarks
    mediapipe_draw.draw_landmarks(image, results.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                                  mediapipe_draw.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
                                  )
    
    return image

# capture = cv2.VideoCapture(0)
# # Access/ set media pipe mode
# with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while capture.isOpened():
#         ret, frame = capture.read()

#         # Make detections
#         image, results = mediapipe_detection(frame, holistic)
#         print(results)

#         # Draw landmarks
#         image = draw_landmarks_custom(image, results)

#         cv2.imshow("Feed", image)
#         if cv2.waitKey(1) == ord("q"):
#             break

#     capture.release()
#     cv2.destroyAllWindows()

# Extracting key points
def extract_landmarks(results):
    face = np.array([[result.x, result.y, result.z] for result in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    left_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    
    return np.concatenate([face, left_hand_landmark, right_hand_landmark, pose])

# Path for exported data, numpy arrays
data_location = os.path.join("mediapipe_data")

# Actions/ sign language gestures to detect
# gestures = np.array(["Hello", "Thanks", "Iloveyou", "Good"])
gestures = np.array(["Hello", "Iloveyou", "Good"])
no_sequences = 30
sequence_length = 30

# for gesture in gestures:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(data_location, gesture, str(sequence)))
#         except:
#             pass

# capture = cv2.VideoCapture(0)
# # Access/ set media pipe mode
# with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # Iterate through all the gestures we need to train
#     for gesture in gestures:
#         # Next mediapipe_holistic through the sequences/ videos
#         for sequence in range(no_sequences):
#             #mediapipe_holistic through the video length
#             for frame_count in range(sequence_length):
#                 ret, frame = capture.read()

#                 # Make detections
#                 image, results = mediapipe_detection(frame, holistic)

#                 # Draw landmarks
#                 draw_landmarks_custom(image, results)

#                 # Delay for the next frame
#                 if frame_count == 0:
#                     cv2.putText(image, "The program will now start to collect training data", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, f"Frames being collected for {gesture} video number {sequence}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.imshow("Feed", image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, f"Frames being collected for {gesture} video number {sequence}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.imshow("Feed", image)

#                 # Retrieve and export gesture key points
#                 get_key_points = extract_landmarks(results)
#                 numpy_path = os.path.join(data_location, gesture, str(sequence), str(frame_count))
#                 np.save(numpy_path, get_key_points)

#                 if cv2.waitKey(1) == ord("q"):
#                     break

#     capture.release()
#     cv2.destroyAllWindows()

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
# model.fit(x_train, y_train, epochs=1000, callbacks=[tensor_board_callback])

# model.save("gestures.h5")

model.load_weights("gestures.h5")

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

sequence = []
sentence = []
threshold = 0.4

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
        sequence.insert(0, get_key_points)
        sequence = sequence[:30]

        if len(sequence) == 30:
            result = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(result)
            print(gestures[np.argmax(result)])

        if result[np.argmax(result)] > threshold:
            if len(sentence) > 0:
                if gestures[np.argmax(result)] != sentence[-1]:
                    sentence.append(gestures[np.argmax(result)])
            else:
                sentence.append(gestures[np.argmax(result)])
            
        if len(sentence) > 5:
            sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, " ".join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_AA) 

        cv2.imshow("Feed", image)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
