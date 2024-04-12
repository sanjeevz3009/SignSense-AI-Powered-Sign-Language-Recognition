import cv2
import mediapipe as mp
import numpy as np
import os
from utils import draw_landmarks_custom, extract_landmarks, mediapipe_detection

# Holistic model
mediapipe_holistic = mp.solutions.holistic

# Path for exported data, numpy arrays
data_location = os.path.join("mediapipe_data")

# Actions/ sign language gestures to detect
# gestures = np.array(["Hello", "Thanks", "Iloveyou", "Good"])
gestures = np.array(["Hello", "Iloveyou", "Good"])
no_sequences = 30
sequence_length = 30

for gesture in gestures:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(data_location, gesture, str(sequence)))
        except:
            pass

capture = cv2.VideoCapture(0)
# Access/ set media pipe mode
with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Iterate through all the gestures we need to train
    for gesture in gestures:
        # Next mediapipe_holistic through the sequences/ videos
        for sequence in range(no_sequences):
            #mediapipe_holistic through the video length
            for frame_count in range(sequence_length):
                ret, frame = capture.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_landmarks_custom(image, results)

                # Delay for the next frame
                if frame_count == 0:
                    cv2.putText(image, "The program will now start to collect training data", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f"Frames being collected for {gesture} video number {sequence}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Feed", image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f"Frames being collected for {gesture} video number {sequence}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Feed", image)

                # Retrieve and export gesture key points
                get_key_points = extract_landmarks(results)
                numpy_path = os.path.join(data_location, gesture, str(sequence), str(frame_count))
                np.save(numpy_path, get_key_points)

                if cv2.waitKey(1) == ord("q"):
                    break

    capture.release()
    cv2.destroyAllWindows()
