# For operating system related operations
import os

# OpenCV library for computer vision tasks
import cv2

# MediaPipe library for various media processing tasks, including pose estimation
import mediapipe as mp

# For numerical computations
import numpy as np

# Custom modules are imported which contain predefined gestures to detect,
# sequence information, and utility functions for data handling and visualisation
from gestures_to_detect import gestures, no_sequences, sequence_length
from utils import (
    data_location,
    draw_landmarks_custom,
    extract_landmarks,
    mediapipe_detection,
)

# Constants and configurations
# These constants define minimum confidence thresholds for detection and tracking,
# and waiting times for different stages of the program
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WAIT_TIME_INITIAL = 3000
WAIT_TIME_NORMAL = 1


def create_directories():
    """
    Create directories for storing training data.
    """
    for gesture in gestures:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(data_location, gesture, str(sequence)))
            except FileExistsError:
                pass


def collect_training_data():
    """
    Collect training data for gesture recognition.
    """
    capture = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        for gesture in gestures:
            for sequence in range(no_sequences):
                for frame_count in range(sequence_length):
                    ret, frame = capture.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results, gesture, sequence, frame_count)
                    save_key_points(results, gesture, sequence, frame_count)
                    if cv2.waitKey(WAIT_TIME_NORMAL) == ord("q"):
                        break
    capture.release()
    cv2.destroyAllWindows()


def draw_landmarks(image, results, gesture, sequence, frame_count):
    """
    Draw landmarks on the image.
    Draws landmarks on the image, overlays text for feedback,
    and displays the feed.
    """
    draw_landmarks_custom(image, results)
    cv2.putText(
        image,
        f"Frames being collected for {gesture} video number {sequence}",
        (15, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("Feed", image)
    if frame_count == 0:
        cv2.putText(
            image,
            "The program will now start to collect training data",
            (120, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.waitKey(WAIT_TIME_INITIAL)
    else:
        cv2.waitKey(WAIT_TIME_NORMAL)


def save_key_points(results, gesture, sequence, frame_count):
    """
    Save key points to numpy file.
    Extracts and saves key points (landmarks) to numpy files.
    """
    key_points = extract_landmarks(results)
    numpy_path = os.path.join(data_location, gesture, str(sequence), str(frame_count))
    np.save(numpy_path, key_points)


# The main block of code executes the create_directories() function
# to set up the directory structure for storing training data and
# then calls the collect_training_data() function to start collecting data from the webcam feed.
# This Python code is for collecting training data for gesture recognition using the MediaPipe library.
# This script is designed to facilitate the collection of gesture data for subsequent training of a gesture
# recognition model using the MediaPipe library
if __name__ == "__main__":
    create_directories()
    collect_training_data()
