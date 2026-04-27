import os
import time
import zipfile
import tempfile
import threading
from collections import deque

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from utils import (
    extract_extended_single_hand_features,
    extract_dual_hand_features,
)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SignBridge Web", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    bundle = joblib.load("signbridge_model.joblib")
    return bundle.get("pipeline"), bundle.get("label_encoder")


pipeline, label_encoder = load_model()

# =========================
# FONT LOADING FOR CYRILLIC
# =========================
@st.cache_resource
def load_cyrillic_font(size=34):
    # Put this font file in the same folder as web_app.py in your repo:
    # Roboto-VariableFont_wdth,wght.ttf
    font_paths = [
        "Roboto-VariableFont_wdth,wght.ttf",
        os.path.join(os.path.dirname(__file__), "Roboto-VariableFont_wdth,wght.ttf"),
        "/mount/src/signbridge-web/Roboto-VariableFont_wdth,wght.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    return ImageFont.load_default()


CYRILLIC_FONT = load_cyrillic_font(34)

# =========================
# ZIP DOWNLOAD
# =========================
def create_zip():
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "signbridge_desktop.zip")

    files = ["app.py", "utils.py", "signbridge_model.joblib", "Roboto-VariableFont_wdth,wght.ttf"]

    with zipfile.ZipFile(zip_path, "w") as z:
        for f in files:
            if os.path.exists(f):
                z.write(f)

    return zip_path


# =========================
# VIDEO PROCESSOR
# =========================
class SignBridgeProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.threshold = 0.7
        self.mode = "single"
        self.label = "..."
        self.confidence = 0.0
        self.status = "Покажи жест пред камерата"

        self.prediction_buffer = deque(maxlen=20)
        self.prob_buffer = deque(maxlen=5)
        self.last_pred_time = 0.0
        self.pred_interval = 0.12

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        all_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(np.array(coords, dtype=np.float32))

        with self.lock:
            threshold = self.threshold
            mode = self.mode

        if all_landmarks:
            if len(all_landmarks) >= 2 and mode == "dual":
                features = extract_dual_hand_features(
                    all_landmarks[0],
                    all_landmarks[1],
                )[:257]
            else:
                features = extract_extended_single_hand_features(all_landmarks[0])

            self.prediction_buffer.append(features)

        now = time.time()
        if now - self.last_pred_time >= self.pred_interval:
            self.last_pred_time = now

            if len(self.prediction_buffer) >= 8:
                try:
                    features_array = np.vstack(self.prediction_buffer)

                    mean_features = np.mean(features_array, axis=0)
                    std_features = np.std(features_array, axis=0)

                    combined = np.concatenate([mean_features, std_features])
                    X = combined.reshape(1, -1)

                    proba = pipeline.predict_proba(X)[0]
                    self.prob_buffer.append(proba)

                    avg_probs = np.mean(self.prob_buffer, axis=0)
                    idx = int(np.argmax(avg_probs))

                    conf = float(avg_probs[idx])
                    label = label_encoder.inverse_transform([idx])[0]

                    with self.lock:
                        self.confidence = conf

                        if conf >= threshold:
                            self.label = str(label)
                            self.status = f"{label}"
                        else:
                            self.label = "..."
                            self.status = "..."

                except Exception:
                    with self.lock:
                        self.status = "Грешка при предсказване"
                        self.label = "Грешка"
                        self.confidence = 0.0

        with self.lock:
            overlay_text = f"{self.status} | {self.confidence:.2f}"

        # =========================
        # CYRILLIC OVERLAY WITH PIL
        # OpenCV cv2.putText does NOT support Cyrillic.
        # Convert BGR -> RGB before PIL, then RGB -> BGR after drawing.
        # =========================
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil, "RGBA")

        try:
            bbox = draw.textbbox((0, 0), overlay_text, font=CYRILLIC_FONT)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = 620, 40

        x, y = 16, 16
        padding_x, padding_y = 14, 10
        box = (
            x,
            y,
            x + text_w + padding_x * 2,
            y + text_h + padding_y * 2,
        )

        draw.rounded_rectangle(box, radius=12, fill=(0, 0, 0, 180))
        draw.text(
            (x + padding_x, y + padding_y),
            overlay_text,
            font=CYRILLIC_FONT,
            fill=(255, 255, 255, 255),
        )

        image_rgb = np.array(img_pil)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def __del__(self):
        try:
            self.hands.close()
        except Exception:
            pass


# =========================
# UI
# =========================
st.title("🖐️ SignBridge Web")
st.caption("Глас за хората без глас.")

col1, col2 = st.columns([2, 1])

with col2:
    threshold = st.slider("Праг", 0.3, 0.95, 0.7)
    mode = st.selectbox("Режим", ["single", "dual"])

    st.markdown("### Резултат")
    result_box = st.empty()
    conf_box = st.empty()

    st.markdown("---")
    st.markdown("### ⬇️ Desktop версия")

    if st.button("📦 Подготви ZIP"):
        zip_path = create_zip()
        with open(zip_path, "rb") as f:
            st.download_button(
                "💾 Свали",
                f,
                file_name="signbridge_desktop.zip",
                mime="application/zip",
            )

with col1:
    ctx = webrtc_streamer(
        key="signbridge-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignBridgeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        async_processing=True,
    )

if ctx.video_processor:
    with ctx.video_processor.lock:
        ctx.video_processor.threshold = threshold
        ctx.video_processor.mode = mode
        label = ctx.video_processor.label
        confidence = ctx.video_processor.confidence
        status = ctx.video_processor.status

    if label not in ("...", "Грешка") and confidence >= threshold:
        result_box.success(status)
    elif label == "Грешка":
        result_box.error(status)
    else:
        result_box.warning(status)

    conf_box.metric("Сигурност", f"{confidence:.2f}")
else:
    result_box.info("Натисни START и позволи достъп до камерата.")
    conf_box.metric("Сигурност", "0.00")
