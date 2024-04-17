# OpenCV library for computer vision tasks
import cv2

# MediaPipe library for various media processing tasks, including pose estimation,
# hand tracking etc
import mediapipe as mp

# For numerical computations
import numpy as np

# For building interactive web applications
import streamlit as st

# For defining and training neural network models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Custom modules are imported which contain predefined gestures to detect
# and utility functions for data handling and visualisation
from gestures_to_detect import gestures
from utils import draw_landmarks_custom, extract_landmarks, mediapipe_detection

from twilio_turn_server import get_ice_servers

# Constants
# These constants define minimum confidence thresholds for detection and tracking
# model input shape, and log directory
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_INPUT_SHAPE = (30, 126)
LOG_DIR = "Logs"

# Holistic model
mediapipe_holistic = mp.solutions.holistic

# Load pre-trained model
# A pre-trained LSTM neural network model is defined and loaded
model = Sequential(
    [
        LSTM(
            64, return_sequences=True, activation="relu", input_shape=MODEL_INPUT_SHAPE
        ),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(gestures.shape[0], activation="softmax"),
    ]
)
model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
model.load_weights("gestures_3.h5")


# def main():
#     """
#     Main function to run the sign language recognition system
#     The main() function is the entry point of the application.
#     It sets up the Streamlit interface, captures video feed, detects hand
#     gestures, updates predictions, updates recognized sentences, and
#     displays results
#     """
#     # Streamlit setup
#     st.title("Sign Language Recognition")
#     frame_placeholder = st.empty()
#     gesture_text = st.empty()
#     stop_button_pressed = st.button("Stop")

#     # Variables
#     sequence = []
#     sentence = []
#     predictions = []
#     threshold = 0.95

#     # Video capture setup
#     capture = cv2.VideoCapture(0)
#     if not capture.isOpened():
#         st.write("Error: Unable to open video capture device.")
#         return

#     # Hand gesture detection setup
#     with mediapipe_holistic.Holistic(
#         min_detection_confidence=MIN_DETECTION_CONFIDENCE,
#         min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
#     ) as holistic:
#         while capture.isOpened() and not stop_button_pressed:
#             ret, frame = capture.read()
#             if not ret:
#                 st.write("The video feed has ended.")
#                 break

#             # Hand gesture detection and processing
#             image, results = process_frame(frame, holistic)
#             sequence = update_sequence(sequence, results)
#             predictions = update_predictions(sequence, predictions)
#             sentence = update_sentence(
#                 predictions, sequence, sentence, gestures, threshold
#             )

#             # Display results
#             display_results(image, sentence, frame_placeholder, gesture_text)

#             if cv2.waitKey(1) == ord("q") or stop_button_pressed:
#                 break

#     # Release resources
#     capture.release()
#     cv2.destroyAllWindows()


# def process_frame(frame, holistic):
#     """
#     Process each frame for hand gesture detection.
#     This function processes each frame for hand gesture detection using
#     the MediaPipe library.
#     """
#     image, results = mediapipe_detection(frame, holistic)
#     image = draw_landmarks_custom(image, results)
#     return image, results


# def update_sequence(sequence, results):
#     """
#     Update the sequence of hand gesture keypoints.
#     These functions update the sequence of hand gesture keypoints
#     and predictions based on the sequence.
#     """
#     get_key_points = extract_landmarks(results)
#     sequence.append(get_key_points)
#     return sequence[-30:]


# def update_predictions(sequence, predictions):
#     """Update the predictions based on the sequence."""
#     if len(sequence) == 30:
#         res = model.predict(np.expand_dims(sequence, axis=0))[0]
#         predictions.append(np.argmax(res))
#     return predictions


# def update_sentence(predictions, sequence, sentence, gestures, threshold):
#     """
#     Updating the recognised sentence based on gesture predictions.
#     This function updates the recognised sentence based on
#     gesture predictions.

#     Args:
#         predictions (List[int]): List of gesture predictions.
#         sequence (List): Sequence of hand gestures.
#         sentence (List[str] or None): Recognised sentence or None.
#         gestures (List[str]): List of gesture labels.
#         threshold (float): Confidence threshold for predictions.
#     """
#     # Check if sentence is None, initialise as empty list
#     if sentence is None:
#         sentence = []

#     # Check if enough frames are available for prediction
#     if len(sequence) != 30:
#         return sentence  # Return sentence unchanged if sequence length is not 30

#     res = model.predict(np.expand_dims(sequence, axis=0))[0]
#     predictions.append(np.argmax(res))

#     # Check if the most predicted gesture is consistent over the last 10 frames
#     if np.unique(predictions[-10:])[0] != np.argmax(res):
#         return sentence  # Return sentence unchanged if prediction is inconsistent

#     confidence = res[np.argmax(res)]
#     # Check if confidence is above threshold
#     if confidence <= threshold:
#         return sentence  # Return sentence unchanged if confidence is below threshold

#     current_gesture = gestures[np.argmax(res)]
#     # Update sentence if it's the first gesture or different from the last one
#     if len(sentence) == 0 or current_gesture != sentence[-1]:
#         sentence.append(current_gesture)

#     # Limit sentence length to 5
#     if len(sentence) > 5:
#         sentence = sentence[-5:]

#     return sentence  # Return updated sentence


# def display_results(image, sentence, frame_placeholder, gesture_text):
#     """
#     Display results in the Streamlit interface.
#     This function displays results in the Streamlit interface.
#     """
#     frame_conversion = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     frame_placeholder.image(frame_conversion, channels="RGB")

#     if sentence:
#         gesture_text.markdown(f"# {sentence[-1]}")
#     else:
#         gesture_text.empty()


# # This script sets up a real-time sign language recognition system
# # that captures video feed, processes frames for hand gesture detection,
# # predicts gestures using a pre-trained LSTM model,
# # and displays results interactively using Streamlit.
# if __name__ == "__main__":
#     main()

import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
from utils import draw_landmarks_custom, mediapipe_detection
import av

# Holistic model
mediapipe_holistic = mp.solutions.holistic
st.title("Sign Language Recognition")
gesture_text = st.empty()
stop_button_pressed = st.button("Stop")

class HolisticTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mediapipe_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Make detections
        image, results = mediapipe_detection(image, self.holistic)
        print(results)

        # Draw landmarks
        image = draw_landmarks_custom(image, results)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="example", 
    rtc_configuration={"iceServers": get_ice_servers()},
    video_transformer_factory=HolisticTransformer,
    async_transform=True,
)

if webrtc_ctx.video_transformer:
    if stop_button_pressed:
        webrtc_ctx.video_transformer.holistic.close()