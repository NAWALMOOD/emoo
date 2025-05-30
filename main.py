
import copy
import csv
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import streamlit as st
from huggingface_hub import InferenceClient
from model import KeyPointClassifier
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ----------------------------
# Hugging Face Chatbot Setup
# ----------------------------
llm_client = InferenceClient(model="microsoft/Phi-3-mini-4k-instruct", timeout=120)
def chat_with_bot(user_input):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant who gives thoughtful and emotionally intelligent activity suggestions."}
        ]

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get model response
    response = llm_client.chat_completion(
        messages=st.session_state.messages,
        temperature=0.7,
        max_tokens=900
    )

    bot_message = response['choices'][0]['message']['content']
    # Add bot message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_message})

    return bot_message


# ----------------------------
# Utility Functions
# ----------------------------
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        max_value = 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([
        [min(int(lm.x * image_width), image_width - 1), min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

# ----------------------------
# Emotion Recognition Class
# ----------------------------
class EmotionRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                brect = calc_bounding_rect(debug_image, face_landmarks)
                landmark_list = calc_landmark_list(debug_image, face_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                facial_emotion_id = self.keypoint_classifier(pre_processed_landmark_list)
                facial_text = self.keypoint_classifier_labels[facial_emotion_id]
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, facial_text)

        return debug_image

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Real-time Emotion + Chatbot", layout="wide")
st.title("😃 Real-time Emotion Recognition + 🤖 Chatbot Assistant")
st.markdown("Using Mediapipe, MLP classifier + Streamlit WebRTC, and HuggingFace")

col1, col2 = st.columns(2)

with col1:
    st.header("🎥 Emotion Detection")
    webrtc_streamer(
        key="emotion_detection",
        video_transformer_factory=EmotionRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


with col2:
    st.header("💬 Chatbot Assistant")
    st.markdown("Ask for mood-improving activity suggestions.")

    # Use chat_input to get the new message
    user_input = st.chat_input("You:")

    if user_input:
        # Clear previous chat history
        st.session_state.chat_log = []

        # Append the current user message and bot response
        st.session_state.chat_log.append(("You", user_input))

        with st.spinner("Bot is thinking..."):
            response = chat_with_bot(user_input)

        st.session_state.chat_log.append(("Bot", response))

    # Display only the latest chat log (which has max two messages)
    if "chat_log" in st.session_state:
        for speaker, message in st.session_state.chat_log:
            with st.chat_message("user" if speaker == "You" else "assistant"):
                st.markdown(message)