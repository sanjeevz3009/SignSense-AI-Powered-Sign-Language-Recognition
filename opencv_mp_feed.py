import cv2
import mediapipe as mp
from utils import draw_landmarks_custom, mediapipe_detection

# Holistic model
mediapipe_holistic = mp.solutions.holistic

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

        cv2.imshow("Feed", image)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
