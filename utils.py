import os

import cv2
import mediapipe as mp
import numpy as np

# Global constants
# Define parameters for drawing landmarks
HAND_LANDMARK_THICKNESS = 2
HAND_CIRCLE_RADIUS_NUM = 2
LANDMARK_LINE_COLOUR = (0, 255, 8)
LANDMARK_POINT_COLOUR = (255, 0, 200)

# Holistic model
MEDIAPIPE_HOLISTIC = mp.solutions.holistic
# Drawing utilities
MEDIAPIPE_DRAW = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """
    Perform detection on the given image using the specified model.
    This function performs hand landmark detection on the given image using
    the specified MediaPipe model.
    It converts the image to RGB format, processes the detection,
    and then converts the image back to BGR format before returning it.

    Args:
        image: Input image.
        model: Mediapipe model for detection.

    Returns:
        image: Image with detections.
        results: Detection results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks_custom(image, results):
    """
    Draw custom landmarks on the image.
    This function draws custom landmarks (points and connections) on the image based on
    the detection results.
    It uses MediaPipe's drawing utilities to draw landmarks for both left and right hands.

    Args:
        image: Input image.
        results: Detection results.

    Returns:
        image: Image with drawn landmarks.
    """
    MEDIAPIPE_DRAW.draw_landmarks(
        image,
        results.left_hand_landmarks,
        MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
        MEDIAPIPE_DRAW.DrawingSpec(
            color=LANDMARK_LINE_COLOUR,
            thickness=HAND_LANDMARK_THICKNESS,
            circle_radius=HAND_CIRCLE_RADIUS_NUM,
        ),
        MEDIAPIPE_DRAW.DrawingSpec(
            color=LANDMARK_POINT_COLOUR,
            thickness=HAND_LANDMARK_THICKNESS,
            circle_radius=HAND_CIRCLE_RADIUS_NUM,
        ),
    )
    MEDIAPIPE_DRAW.draw_landmarks(
        image,
        results.right_hand_landmarks,
        MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
        MEDIAPIPE_DRAW.DrawingSpec(
            color=LANDMARK_LINE_COLOUR,
            thickness=HAND_LANDMARK_THICKNESS,
            circle_radius=HAND_CIRCLE_RADIUS_NUM,
        ),
        MEDIAPIPE_DRAW.DrawingSpec(
            color=LANDMARK_POINT_COLOUR,
            thickness=HAND_LANDMARK_THICKNESS,
            circle_radius=HAND_CIRCLE_RADIUS_NUM,
        ),
    )

    return image


def prob_visualisation(res, actions, input_frame, colors):
    """
    Visualise probabilities of actions.
    This function visualises probabilities of actions on the input frame.
    It draws rectangles proportional to the probabilities and overlays action names on them.

    Args:
        res: List of probabilities.
        actions: List of action names.
        input_frame: Input frame to draw on.
        colors: List of colors.

    Returns:
        output_frame: Frame with visualized probabilities.
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


def extract_landmarks(results):
    """
    Extract landmark points from the detection results.
    This function extracts landmark points from the detection results.
    It retrieves the coordinates of landmark points for both left and
    right hands and flattens them into a 1D array.

    Args:
        results: Detection results.

    Returns:
        landmarks: Extracted landmark points.
    """
    left_hand_landmark = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        left_hand_landmark = np.array(
            [
                [result.x, result.y, result.z]
                for result in results.left_hand_landmarks.landmark
            ]
        ).flatten()

    right_hand_landmark = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        right_hand_landmark = np.array(
            [
                [result.x, result.y, result.z]
                for result in results.right_hand_landmarks.landmark
            ]
        ).flatten()

    return np.concatenate([left_hand_landmark, right_hand_landmark])


# Path for exported data, numpy arrays
# Specifies the path for exported data, which includes numpy arrays storing hand landmark information
data_location = os.path.join("mediapipe_data")

# These utility functions are likely used in conjunction with a MediaPipe-based hand landmark
# detection system to process and visualise hand landmarks in real-time video stream
