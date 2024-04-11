import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np
import os
import time

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

capture = cv2.VideoCapture(0)
# Access/ set media pipe mode
with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
