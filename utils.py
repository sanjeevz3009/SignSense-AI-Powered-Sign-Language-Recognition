import cv2
import mediapipe as mp
import numpy as np
import os

hand_landmark_thickness = 2
hand_circle_radius_num = 2
face_landmark_thickness = 1
face_circle_radius_num = 1
pose_landmark_thickness = 1
pose_landmark_radius_num = 1
landmark_line_colour = (0, 255, 8)
landmark_point_colour = (255, 0, 200)

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
                                  mediapipe_draw.DrawingSpec(color=landmark_line_colour, thickness=face_landmark_thickness,
                                                             circle_radius=face_circle_radius_num),
                                  mediapipe_draw.DrawingSpec(color=landmark_point_colour, thickness=face_landmark_thickness,
                                                             circle_radius=face_circle_radius_num)
                                  )
    # Draw left hand landmarks
    mediapipe_draw.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=landmark_line_colour, thickness=hand_landmark_thickness,
                                                             circle_radius=hand_circle_radius_num),
                                  mediapipe_draw.DrawingSpec(color=landmark_point_colour, thickness=hand_landmark_thickness,
                                                             circle_radius=hand_circle_radius_num)
                                  )
    # Draw right hand landmarks
    mediapipe_draw.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=landmark_line_colour, thickness=hand_landmark_thickness,
                                                             circle_radius=hand_circle_radius_num),
                                  mediapipe_draw.DrawingSpec(color=landmark_point_colour, thickness=hand_landmark_thickness,
                                                             circle_radius=hand_circle_radius_num)
                                  )
    # Draw pose landmarks
    mediapipe_draw.draw_landmarks(image, results.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS,
                                  mediapipe_draw.DrawingSpec(color=landmark_line_colour, thickness=pose_landmark_thickness,
                                                             circle_radius=pose_landmark_radius_num),
                                  mediapipe_draw.DrawingSpec(color=landmark_point_colour, thickness=pose_landmark_thickness,
                                                             circle_radius=pose_landmark_radius_num)
                                  )
    
    return image


colours = [(245,117,16), (117,245,16), (16,117,245)]
def prob_visualation(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# # Extracting key points
# def extract_landmarks(results):
#     face = np.array([[result.x, result.y, result.z] for result in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
#     left_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     right_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    
#     return np.concatenate([face, left_hand_landmark, right_hand_landmark, pose])

# Extracting key points
def extract_landmarks(results):
    face = np.zeros(1404)
    if results.face_landmarks:
        face = np.array([[result.x, result.y, result.z] for result in results.face_landmarks.landmark]).flatten()
    
    left_hand_landmark = np.zeros(21*3)
    if results.left_hand_landmarks:
        left_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.left_hand_landmarks.landmark]).flatten()
    
    right_hand_landmark = np.zeros(21*3)
    if results.right_hand_landmarks:
        right_hand_landmark = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten()
    
    pose = np.zeros(132)
    if results.pose_landmarks:
        pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten()
    
    return np.concatenate([face, left_hand_landmark, right_hand_landmark, pose])

# Path for exported data, numpy arrays
data_location = os.path.join("mediapipe_data")
