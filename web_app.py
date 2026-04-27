import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import time
import zipfile
import tempfile
import os

from utils import (
    extract_extended_single_hand_features,
    extract_dual_hand_features
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
    return (
        bundle.get("pipeline"),
        bundle.get("label_encoder")
    )

pipeline, label_encoder = load_model()

# =========================
# ZIP DOWNLOAD
# =========================
def create_zip():
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "signbridge_desktop.zip")

    files = ["app.py", "utils.py", "signbridge_model.joblib"]

    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in files:
            if os.path.exists(f):
                z.write(f)

    return zip_path

# =========================
# STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

# =========================
# UI
# =========================
st.title("🖐️ SignBridge Web")

col1, col2 = st.columns([2, 1])

with col2:
    if st.button("▶️ Стартирай"):
        st.session_state.running = True

    if st.button("⏹️ Спри"):
        st.session_state.running = False

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
                mime="application/zip"
            )

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# =========================
# VIDEO PLACEHOLDER
# =========================
frame_window = col1.image([])

# =========================
# BUFFERS
# =========================
prediction_buffer = deque(maxlen=20)
prob_buffer = deque(maxlen=5)

# =========================
# CAMERA LOOP (SAFE)
# =========================
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    last_pred_time = 0
    pred_interval = 0.12

    while st.session_state.running:
        ret, frame = cap.read()

        if not ret:
            st.error("Няма камера")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        all_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(np.array(coords, dtype=np.float32))

        # FEATURES
        if all_landmarks:
            if len(all_landmarks) == 2 and mode == "dual":
                features = extract_dual_hand_features(
                    all_landmarks[0],
                    all_landmarks[1]
                )[:257]
            else:
                features = extract_extended_single_hand_features(
                    all_landmarks[0]
                )

            prediction_buffer.append(features)

        # PREDICTION
        now = time.time()

        if now - last_pred_time >= pred_interval:
            last_pred_time = now

            if len(prediction_buffer) >= 8:
                features_array = np.vstack(prediction_buffer)

                mean_features = np.mean(features_array, axis=0)
                std_features = np.std(features_array, axis=0)

                combined = np.concatenate([mean_features, std_features])
                X = combined.reshape(1, -1)

                try:
                    proba = pipeline.predict_proba(X)[0]

                    prob_buffer.append(proba)
                    avg_probs = np.mean(prob_buffer, axis=0)

                    idx = np.argmax(avg_probs)
                    conf = float(avg_probs[idx])
                    label = label_encoder.inverse_transform([idx])[0]

                    if conf >= threshold:
                        result_box.success(f"👉 {label}")
                    else:
                        result_box.warning("...")

                    conf_box.metric("Сигурност", f"{conf:.2f}")

                except:
                    result_box.error("Грешка")

        frame_window.image(rgb)

    cap.release()
else:
    st.info("Натисни 'Стартирай' за да включиш камерата")