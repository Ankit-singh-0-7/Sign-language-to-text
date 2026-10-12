import json
import os
import sys
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
import mediapipe as mp

import preprocessing as pp

# -----------------------------------------------------------------------------
# Page Configuration & Metadata
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ISL Sentence Recognizer | AI Deep Learning",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern Glassmorphic Dark Aesthetics
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #0a0e1a;
    --card-bg: rgba(17, 24, 39, 0.75);
    --card-border: rgba(99, 102, 241, 0.2);
    --primary: #6366f1;
    --primary-glow: rgba(99, 102, 241, 0.4);
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(15, 23, 42, 0.6) 0%, transparent 40%),
                #070a13;
    background-attachment: fixed;
}

/* Glassmorphism Cards */
.glass-panel {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.3s ease;
}

.glass-panel:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 35px -8px var(--primary-glow);
}

/* Header Banner */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* Prediction Result Card */
.result-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
}

.result-label {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 12px 0 6px 0;
}

.result-conf-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.conf-high {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
}

.conf-med {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fbbf24;
}

.conf-low {
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid #f43f5e;
    color: #fb7185;
}

/* Metric Pill */
.metric-pill {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-pill .title {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.metric-pill .val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Progress bar labels */
.candidate-row {
    margin-bottom: 14px;
}
.candidate-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 6px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0b1120 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Loading & Initialization (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading deep learning model & classes...")
def load_model_and_classes():
    """Load pre-trained ISL GRU neural network and 101 target sentences."""
    model_path = "isl_gru_v2.keras"
    classes_path = "classes.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found in current directory.")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file '{classes_path}' not found in current directory.")

    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Warm up inference engine
    dummy_input = np.zeros((1, 32, 368), dtype=np.float32)
    model(dummy_input, training=False)

    return model, classes


def make_landmarkers():
    """Create fresh MediaPipe Pose and Hand Landmarker instances with reset timestamps."""
    pose_path = "models/pose.task"
    hand_path = "models/hand.task"

    if not os.path.exists(pose_path) or not os.path.exists(hand_path):
        os.makedirs("models", exist_ok=True)
        import urllib.request
        if not os.path.exists(pose_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                pose_path
            )
        if not os.path.exists(hand_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                hand_path
            )

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=pose_path),
            running_mode=vision.RunningMode.VIDEO
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=hand_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
    )
    return pose, hands


# -----------------------------------------------------------------------------
# Video Feature Extraction Pipeline
# -----------------------------------------------------------------------------
def frame_features(pr, hr):
    """
    Extract 258 floating-point features per frame:
    - Pose: 33 landmarks x (x, y, z, visibility) = 132 features
    - Left Hand: 21 landmarks x (x, y, z) = 63 features
    - Right Hand: 21 landmarks x (x, y, z) = 63 features
    Matches the ISL-CSLRT trained data representation.
    """
    pose_feat = (
        np.array([[q.x, q.y, q.z, q.visibility] for q in pr.pose_landmarks[0]]).ravel()
        if pr.pose_landmarks else np.zeros(132, dtype=np.float32)
    )
    lh, rh = np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = np.array([[q.x, q.y, q.z] for q in lms]).ravel()
        if hd[0].category_name == "Left":
            lh = arr
        else:
            rh = arr
    return np.concatenate([pose_feat, lh, rh]).astype(np.float32)


def draw_skeleton(frame, pr, hr):
    """Draw MediaPipe landmarks and connections directly onto a preview frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw Pose Keypoints (upper body joints)
    if pr.pose_landmarks:
        for idx in [0, 11, 12, 13, 14, 15, 16]:  # Nose, shoulders, elbows, wrists
            lm = pr.pose_landmarks[0][idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 6, (46, 204, 113), -1)  # Emerald green

        # Shoulders & Arms connectors
        connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for s, e in connections:
            p1 = pr.pose_landmarks[0][s]
            p2 = pr.pose_landmarks[0][e]
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(annotated, pt1, pt2, (46, 204, 113), 2)

    # Draw Hand Keypoints & Connectors
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm base
    ]

    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        is_left = (hd[0].category_name == "Left")
        color = (255, 229, 0) if is_left else (255, 60, 255)  # Cyan vs Magenta

        for s, e in hand_connections:
            p1, p2 = lms[s], lms[e]
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(annotated, pt1, pt2, color, 2)

        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, color, -1)

    return annotated


def extract_video_features(video_path, stride=2, max_seconds=12, progress_bar=None, status_text=None):
    """
    Extract pose & hand landmarks over all frames in the video.
    Returns:
        raw_keypoints: ndarray of shape (frames, 258)
        sample_frame: annotated image representing the peak signing gesture
        stats: dictionary with video and detection metadata
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps > 120 or fps < 5:
        fps = 25.0
    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * fps)

    pose, hands = make_landmarkers()

    seq = []
    frame_idx = 0
    annotated_keyframe = None
    max_hand_dist = 0.0
    last_timestamp_ms = -1

    t_start = time.time()

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                target_w = 640
                frame_resized = cv2.resize(frame, (target_w, int(h * target_w / w)))

                img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                )
                computed_ts = int(frame_idx * 1000 / fps)
                timestamp_ms = max(computed_ts, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                pr = pose.detect_for_video(img, timestamp_ms)
                hr = hands.detect_for_video(img, timestamp_ms)

                feats = frame_features(pr, hr)
                seq.append(feats)

                # Detect keyframe with highest hand presence
                hand_mag = np.abs(feats[132:]).sum()
                if hand_mag > max_hand_dist:
                    max_hand_dist = hand_mag
                    annotated_keyframe = draw_skeleton(frame_resized, pr, hr)

            frame_idx += 1
            if progress_bar and total_file_frames > 0:
                pct = min(frame_idx / min(total_file_frames, max_frames), 1.0)
                progress_bar.progress(pct)
                if status_text and frame_idx % 10 == 0:
                    status_text.text(f"Processed frame {frame_idx} of {min(total_file_frames, max_frames)}...")
    finally:
        cap.release()
        try:
            pose.close()
            hands.close()
        except Exception:
            pass

    total_time = time.time() - t_start

    raw = np.array(seq, dtype=np.float32) if len(seq) > 0 else np.zeros((0, 258), dtype=np.float32)

    stats = {
        "fps": fps,
        "frames_extracted": len(raw),
        "total_source_frames": frame_idx,
        "duration_sec": frame_idx / fps,
        "processing_time": total_time,
    }

    return raw, annotated_keyframe, stats


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # Load core AI model
    try:
        model, classes = load_model_and_classes()
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        st.stop()

    # Header / Hero Section
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <div class="hero-title">🤟 ISL Sentence Recognizer</div>
            <div class="hero-subtitle">
                Real-Time Deep Learning System for <b>Indian Sign Language (ISL)</b> Continuous Sentence Recognition.
            </div>
        </div>
        <div>
            <span style="background:rgba(99,102,241,0.2); border:1px solid #6366f1; color:#a5b4fc;
                         padding:6px 14px; border-radius:999px; font-size:0.85rem; font-weight:600;">
                🧠 Bi-GRU + MediaPipe 3D Landmark Pipeline
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.markdown("### ⚙️ Model & Inference Controls")

        conf_min = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.80,
            value=0.30,
            step=0.05,
            help="Minimum certainty before the model suggests re-signing."
        )

        stride_choice = st.selectbox(
            "Sampling Stride",
            options=[1, 2, 3],
            index=1,
            help="Stride 2 matches the training configuration (~12.5–15 FPS)."
        )

        max_clip_seconds = st.slider(
            "Max Duration (seconds)",
            min_value=4,
            max_value=20,
            value=12,
            step=1,
            help="Maximum length of video to process."
        )

        st.markdown("---")
        st.markdown("### 🎯 Sentence Filter Preset")
        st.caption("Restrict candidate search space to enhance accuracy on known phrases.")

        filter_preset = st.radio(
            "Quick Filter",
            options=["All 101 Sentences", "Common Greetings & Daily", "Help & Emergency", "Custom Selection"],
            index=0
        )

        active_classes = classes.copy()
        if filter_preset == "Common Greetings & Daily":
            active_classes = [
                c for c in classes if any(k in c.lower() for k in [
                    "hi", "hello", "how are you", "good morning", "thank", "welcome",
                    "nice", "food", "sleep", "care", "beautiful", "happy"
                ])
            ]
        elif filter_preset == "Help & Emergency":
            active_classes = [
                c for c in classes if any(k in c.lower() for k in [
                    "help", "water", "worry", "doctor", "crying", "cold", "hurt", "angry", "bad"
                ])
            ]
        elif filter_preset == "Custom Selection":
            selected_custom = st.multiselect(
                "Select Target Sentences",
                options=sorted(classes, key=str.lower),
                default=["help me", "hi how are you", "congratulations", "you can do it"]
            )
            if selected_custom:
                active_classes = selected_custom

        st.info(f"Target Vocabulary: **{len(active_classes)} / {len(classes)}** sentences active.")

        st.markdown("---")
        st.markdown("### ℹ️ Architecture Specs")
        st.markdown("""
        - **Dataset:** ISL-CSLRT (687 videos, 101 sentence classes)
        - **Keypoints:** 33 Pose + 42 Hand (258 coordinates/frame)
        - **Temporal Normalizer:** Mid-shoulder centered, 32-frame resampled
        - **Classifier:** Spatial Dropout + Bidirectional GRU (64 units) + Global Avg/Max Pooling
        """)

    # Main Area: Video Input Tabs
    tab_upload, tab_samples, tab_camera = st.tabs([
        "📁 Upload Sign Video",
        "🎬 Try Sample Clips",
        "📹 Webcam Instructions"
    ])

    video_source_path = None
    sample_name = None

    with tab_upload:
        st.markdown("Upload a video clip of an Indian Sign Language sentence (3 to 6 seconds recommended).")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "webm"],
            help="Upper body and both hands should be clearly visible in well-lit conditions."
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_source_path = tfile.name

    with tab_samples:
        st.markdown("Instantly test the model with curated sample sign recordings:")
        sample_dir = "sample_videos"
        sample_files = []
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".mp4", ".mov", ".avi"))]

        if sample_files:
            cols = st.columns(len(sample_files))
            for i, f in enumerate(sample_files):
                with cols[i]:
                    st.markdown(f"**{f}**")
                    full_p = os.path.join(sample_dir, f)
                    if st.button(f"Load Clip {i+1}", key=f"btn_sample_{i}"):
                        video_source_path = full_p
                        sample_name = f
        else:
            st.info("No sample videos found in sample_videos directory.")

    with tab_camera:
        st.markdown("""
        #### 📹 Real-Time / Webcam Guidelines:
        1. **Framing:** Keep your head, upper torso, and both hands in view (landscape mode works best).
        2. **Idle Boundaries:** Keep hands down at your sides for 1 second before and after signing.
        3. **Lighting:** Ensure good contrast so MediaPipe can detect fingers clearly without motion blur.
        4. Record a 3–5 second video using your camera app and upload it via the **Upload Sign Video** tab!
        """)

    # Video Player Preview & Action Button
    if video_source_path:
        st.markdown("---")
        col_preview, col_action = st.columns([1.2, 1])

        with col_preview:
            st.markdown(f"#### 📺 Video Preview {'(' + sample_name + ')' if sample_name else ''}")
            st.video(video_source_path)

        with col_action:
            st.markdown("#### ⚡ Recognition Trigger")
            st.write("Click below to process landmarks, extract temporal features, and run neural inference:")
            run_btn = st.button("🚀 Analyze & Recognize Sign", use_container_width=True)

            if run_btn:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                status_text.text("Tracking pose and hand 3D landmarks...")
                raw, keyframe, stats = extract_video_features(
                    video_source_path,
                    stride=stride_choice,
                    max_seconds=max_clip_seconds,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

                if len(raw) < 10:
                    st.error("Error: Could not extract enough video frames. Please check the video file.")
                    st.stop()

                status_text.text("Segmenting active gesture frames...")
                hand_present = np.abs(raw[:, 132:]).sum(axis=1) > 0
                hand_rate = float(hand_present.mean())

                def find_segments(hand, merge_gap=8, min_len=15):
                    runs, start = [], None
                    for i, h in enumerate(hand):
                        if h and start is None: start = i
                        if not h and start is not None: runs.append([start, i - 1]); start = None
                    if start is not None: runs.append([start, len(hand) - 1])
                    merged = []
                    for r in runs:
                        if merged and r[0] - merged[-1][1] <= merge_gap: merged[-1][1] = r[1]
                        else: merged.append(r)
                    return [(a, b) for a, b in merged if b - a + 1 >= min_len]

                detected_segs = find_segments(hand_present)
                if detected_segs:
                    a, b = max(detected_segs, key=lambda s: s[1] - s[0])
                    clip = raw[max(a - 3, 0): b + 4]
                else:
                    clip = pp.sign_segment(raw)

                if clip is None or len(clip) < 5:
                    st.warning(
                        f"Hands were barely detected (only {hand_rate:.0%} of frames). "
                        "Please ensure both hands are clearly visible in the frame with good lighting."
                    )
                    st.stop()

                status_text.text("Running Bidirectional GRU model...")
                features = pp.build(pp.resample(pp.normalize(clip), 32))
                raw_probs = model(features[None], training=False).numpy()[0]

                # Apply active vocabulary mask
                mask = np.array([1.0 if c in active_classes else 0.0 for c in classes], dtype=np.float32)
                masked_probs = raw_probs * mask

                if masked_probs.sum() == 0:
                    st.error("No active sentences matched the model vocabulary.")
                    st.stop()

                norm_probs = masked_probs / masked_probs.sum()
                best_idx = int(norm_probs.argmax())
                best_label = classes[best_idx]
                best_conf = float(norm_probs[best_idx])
                raw_best_conf = float(raw_probs[best_idx])

                progress_bar.progress(1.0)
                status_text.text("Recognition completed!")

                # -------------------------------------------------------------
                # Display Results Dashboard
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("## 📊 Recognition Analysis & Results")

                res_col, metric_col = st.columns([1.3, 1])

                with res_col:
                    conf_class = "conf-high" if best_conf >= 0.65 else ("conf-med" if best_conf >= conf_min else "conf-low")
                    conf_status = "High Confidence" if best_conf >= 0.65 else ("Moderate Confidence" if best_conf >= conf_min else "Low Confidence")

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:0.9rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">
                            Recognized Sign Sentence
                        </div>
                        <div class="result-label">"{best_label}"</div>
                        <div class="result-conf-badge {conf_class}">
                            {best_conf*100:.1f}% Match ({conf_status})
                        </div>
                        <div style="margin-top:12px; font-size:0.85rem; color:#64748b;">
                            Raw Model Confidence: {raw_best_conf*100:.1f}% across all 101 corpus classes
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_col:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-pill">
                            <div class="title">Hand Detection</div>
                            <div class="val">{hand_rate*100:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-pill">
                            <div class="title">Active Frames</div>
                            <div class="val">{len(segs)}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    m3, m4 = st.columns(2)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-pill">
                            <div class="title">Total Frames</div>
                            <div class="val">{stats['frames_extracted']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-pill">
                            <div class="title">Inference Time</div>
                            <div class="val">{stats['processing_time']:.2f}s</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Top 5 Predictions Breakdown
                st.markdown("### 🏆 Top Candidate Sentences")
                top_indices = norm_probs.argsort()[::-1][:5]

                for rank, idx in enumerate(top_indices, start=1):
                    cand_label = classes[idx]
                    cand_prob = float(norm_probs[idx])
                    color_bar = "#10b981" if rank == 1 else "#6366f1"

                    st.markdown(f"""
                    <div class="candidate-row">
                        <div class="candidate-label">
                            <span><b>#{rank}</b> {cand_label}</span>
                            <span style="font-family:'JetBrains Mono'; font-weight:700; color:{color_bar};">{cand_prob*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(cand_prob)

                # Keyframe Landmark Visualization
                if keyframe is not None:
                    st.markdown("---")
                    st.markdown("### 🦾 Landmark Tracking Visualization")
                    st.caption("MediaPipe real-time 3D joint extraction (Green: Upper Body Pose, Cyan: Left Hand, Magenta: Right Hand):")
                    st.image(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Dictionary / Vocabulary Browser
    st.markdown("---")
    with st.expander("📚 Explore Complete Vocabulary of 101 ISL Sentences"):
        search_query = st.text_input("🔍 Search sentence vocabulary...", placeholder="e.g. help, train, happy, sleep")
        filtered_voc = [c for c in classes if search_query.lower() in c.lower()] if search_query else classes

        voc_cols = st.columns(3)
        for i, sentence in enumerate(sorted(filtered_voc, key=str.lower)):
            with voc_cols[i % 3]:
                st.markdown(f"• `{sentence}`")

    # Footer
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px 0; color:#64748b; font-size:0.85rem;">
        Indian Sign Language Sentence Recognition | Built with Streamlit, TensorFlow & MediaPipe Tasks
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
