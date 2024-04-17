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
    video_transformer_factory=HolisticTransformer,
    async_transform=True,
)

if webrtc_ctx.video_transformer:
    if stop_button_pressed:
        webrtc_ctx.video_transformer.holistic.close()