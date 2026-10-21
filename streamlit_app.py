import collections
import json
import os
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

import preprocessing as pp

st.set_page_config(
    page_title="ISL Sentence Recognizer | Live & Video",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "isl_gru_v2.keras"
CLASSES_PATH = BASE_DIR / "classes.json"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_videos"

# Exact MediaPipe model URLs used in `sign Model.ipynb`
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)


def ensure_mediapipe_assets():
    """Download MediaPipe Pose & Hand task files if not already present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pose_path = MODELS_DIR / "pose.task"
    hand_path = MODELS_DIR / "hand.task"

    if not pose_path.exists():
        urllib.request.urlretrieve(POSE_URL, str(pose_path))
    if not hand_path.exists():
        urllib.request.urlretrieve(HAND_URL, str(hand_path))

    return str(pose_path), str(hand_path)


def create_landmarkers(
    running_mode=vision.RunningMode.VIDEO,
    hand_conf=0.5,
):
    """
    Create fresh MediaPipe PoseLandmarker and HandLandmarker instances.
    Note: In RunningMode.VIDEO, MediaPipe requires strictly monotonically increasing
    timestamps per instance, so we create a fresh pair per video file or per live stream.
    """
    pose_path, hand_path = ensure_mediapipe_assets()

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=pose_path),
        running_mode=running_mode,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=hand_path),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=hand_conf,
        min_hand_presence_confidence=hand_conf,
        min_tracking_confidence=hand_conf,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_opts),
        vision.HandLandmarker.create_from_options(hand_opts),
    )


@st.cache_resource
def load_model_and_classes():
    """Load the trained Keras Bidirectional GRU model and class list once."""
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw_classes = json.load(f)

    # Support both JSON list (["class0", "class1", ...]) and dict formats
    if isinstance(raw_classes, list):
        classes = raw_classes
    elif isinstance(raw_classes, dict):
        # If {"label": idx} or {"0": "label"}
        first_val = next(iter(raw_classes.values()))
        if isinstance(first_val, int):
            classes = [k for k, _ in sorted(raw_classes.items(), key=lambda item: item[1])]
        else:
            classes = [raw_classes[str(i)] for i in range(len(raw_classes))]
    else:
        classes = list(raw_classes)

    return model, classes


def draw_landmarks_on_frame(frame_bgr, pose_res, hand_res):
    """Draw lightweight pose and hand skeleton overlays on a BGR frame."""
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # Draw upper-body pose joints [0, 11, 12, 13, 14, 15, 16]
    if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
        lms = pose_res.pose_landmarks[0]
        pose_edges = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for i, j in pose_edges:
            p1 = (int(lms[i].x * w), int(lms[i].y * h))
            p2 = (int(lms[j].x * w), int(lms[j].y * h))
            cv2.line(out, p1, p2, (255, 180, 0), 2)
        for idx in [0, 11, 12, 13, 14, 15, 16]:
            pt = (int(lms[idx].x * w), int(lms[idx].y * h))
            cv2.circle(out, pt, 4, (0, 215, 255), -1)

    # Draw hand landmarks
    if hand_res.hand_landmarks:
        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        for lms in hand_res.hand_landmarks:
            for i, j in hand_edges:
                p1 = (int(lms[i].x * w), int(lms[i].y * h))
                p2 = (int(lms[j].x * w), int(lms[j].y * h))
                cv2.line(out, p1, p2, (0, 255, 128), 2)
            for lm in lms:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(out, pt, 3, (255, 255, 255), -1)

    return out


def extract_video_keypoints(
    video_path,
    width=640,
    stride=2,
    hand_conf=0.5,
    mirror=False,
    flip_handedness=False,
    progress_bar=None,
):
    """
    Extract (N, 258) raw keypoints from a video file using the exact Colab pipeline:
      - Resize frame to width=640 preserving aspect ratio
      - Sample every `stride=2` frame
      - Use strictly increasing millisecond timestamps in RunningMode.VIDEO
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, 258), dtype=np.float32), {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pose_lm, hand_lm = create_landmarkers(
        running_mode=vision.RunningMode.VIDEO,
        hand_conf=hand_conf,
    )

    out = []
    frame_idx = 0
    last_ts = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                if mirror:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                if w > width:
                    frame = cv2.resize(frame, (width, int(h * width / w)))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts = int(frame_idx * 1000 / fps)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                pr = pose_lm.detect_for_video(mp_img, ts)
                hr = hand_lm.detect_for_video(mp_img, ts)
                out.append(pp.extract_frame_keypoints(pr, hr, flip_handedness=flip_handedness))

            frame_idx += 1
            if progress_bar is not None and total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        pose_lm.close()
        hand_lm.close()

    raw = np.array(out, dtype=np.float32) if out else np.zeros((0, 258), dtype=np.float32)
    hand_detected = int((np.abs(raw[:, 132:]).sum(1) > 0).sum()) if len(raw) else 0
    stats = {
        "total_video_frames": frame_idx,
        "processed_frames": len(raw),
        "hand_detected_frames": hand_detected,
        "fps": round(float(fps), 1),
    }
    return raw, stats


# ---------------------------------------------------------------------------
# Real-Time WebRTC Live Sign Translator Processor
# ---------------------------------------------------------------------------
class LiveSignProcessor(VideoProcessorBase):
    """
    Processes browser webcam frames in real time via WebRTC:
      1. Extracts MediaPipe Pose + Hand keypoints on every `stride` frame.
      2. Supports two live translation modes:
         - Auto-Sign Boundary Mode: Buffers frames while hands are raised/signing,
           and automatically translates the complete sign once hands drop (or buffer hits max).
         - Continuous Sliding Window Mode: Runs inference every N frames over the latest window.
    """

    def __init__(self):
        self.model, self.classes = load_model_and_classes()
        self.pose_lm, self.hand_lm = create_landmarkers(
            running_mode=vision.RunningMode.VIDEO,
            hand_conf=0.5,
        )
        self.lock = threading.Lock()

        # Configurable parameters updated from Streamlit UI
        self.mirror_webcam = True
        self.flip_handedness = False
        self.draw_skeleton = True
        self.live_mode = "Auto-Detect Sign (Recommended)"
        self.window_frames = 45
        self.conf_threshold = 0.35
        self.stride = 2
        self.trim_idle = False
        self.fill_missing_pose = False

        # Internal state
        self.frame_count = 0
        self.last_ts = 0
        self.sliding_buffer = collections.deque(maxlen=60)
        self.active_sign_buffer = []
        self.idle_streak = 0
        self.ema_probs = None

        # Thread-safe outputs read by Streamlit UI
        self.current_prediction = "Waiting for sign..."
        self.current_confidence = 0.0
        self.top5 = []
        self.history = collections.deque(maxlen=10)
        self.status_text = "Raise hands in frame to start signing"
        self.buffered_frames = 0

    def _run_prediction(self, raw_seq):
        probs = pp.predict_probs(
            self.model,
            raw_seq,
            trim_idle=self.trim_idle,
            fill_missing_pose=self.fill_missing_pose,
            min_hand_frames=5,
        )
        if probs is None:
            return

        if self.live_mode == "Continuous Sliding Window":
            if self.ema_probs is None:
                self.ema_probs = probs
            else:
                self.ema_probs = 0.6 * probs + 0.4 * self.ema_probs
            used_probs = self.ema_probs
        else:
            used_probs = probs

        top_indices = np.argsort(used_probs)[::-1][:5]
        top_idx = int(top_indices[0])
        conf = float(used_probs[top_idx])
        label = self.classes[top_idx]

        with self.lock:
            self.top5 = [(self.classes[int(i)], float(used_probs[int(i)])) for i in top_indices]
            if conf >= self.conf_threshold:
                self.current_prediction = label
                self.current_confidence = conf
                if not self.history or self.history[0]["sentence"] != label:
                    self.history.appendleft(
                        {
                            "sentence": label,
                            "confidence": f"{conf * 100:.1f}%",
                            "time": time.strftime("%H:%M:%S"),
                        }
                    )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.mirror_webcam:
            img = cv2.flip(img, 1)

        self.frame_count += 1
        h, w = img.shape[:2]
        proc_img = cv2.resize(img, (640, int(h * 640 / w))) if w > 640 else img

        # Only run MediaPipe every `stride` frames to keep FPS smooth and match Colab stride=2
        if self.frame_count % self.stride == 0:
            rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now_ms = int(time.time() * 1000)
            if now_ms <= self.last_ts:
                now_ms = self.last_ts + 1
            self.last_ts = now_ms

            pr = self.pose_lm.detect_for_video(mp_img, now_ms)
            hr = self.hand_lm.detect_for_video(mp_img, now_ms)
            kp = pp.extract_frame_keypoints(pr, hr, flip_handedness=self.flip_handedness)
            hand_present = bool(np.abs(kp[132:]).sum() > 0)

            if self.draw_skeleton:
                img = draw_landmarks_on_frame(img, pr, hr)

            if self.live_mode == "Auto-Detect Sign (Recommended)":
                if hand_present:
                    self.active_sign_buffer.append(kp)
                    self.idle_streak = 0
                    with self.lock:
                        self.buffered_frames = len(self.active_sign_buffer)
                        self.status_text = f"Recording sign... ({len(self.active_sign_buffer)} frames)"
                    # If user signs continuously for ~3.5 seconds (55 sampled frames), trigger prediction
                    if len(self.active_sign_buffer) >= self.window_frames:
                        seq = np.array(self.active_sign_buffer, dtype=np.float32)
                        self.active_sign_buffer = []
                        self._run_prediction(seq)
                        with self.lock:
                            self.buffered_frames = 0
                            self.status_text = "Sign translated! Ready for next sign."
                else:
                    self.idle_streak += 1
                    # When hands leave frame for ~6 sampled frames after collecting >=10 sign frames, translate
                    if self.idle_streak >= 6 and len(self.active_sign_buffer) >= 10:
                        seq = np.array(self.active_sign_buffer, dtype=np.float32)
                        self.active_sign_buffer = []
                        self._run_prediction(seq)
                        with self.lock:
                            self.buffered_frames = 0
                            self.status_text = "Sign translated! Ready for next sign."
                    elif self.idle_streak >= 6:
                        self.active_sign_buffer = []
                        with self.lock:
                            self.buffered_frames = 0
                            self.status_text = "Raise hands in frame to start signing"
            else:
                # Continuous Sliding Window mode
                self.sliding_buffer.append(kp)
                with self.lock:
                    self.buffered_frames = len(self.sliding_buffer)
                    self.status_text = (
                        f"Sliding window active ({len(self.sliding_buffer)}/{self.window_frames} frames)"
                    )
                if (
                    len(self.sliding_buffer) >= min(24, self.window_frames)
                    and (self.frame_count // self.stride) % 8 == 0
                ):
                    seq = np.array(list(self.sliding_buffer)[-self.window_frames :], dtype=np.float32)
                    self._run_prediction(seq)

        # Draw live HUD banner at bottom of video frame
        with self.lock:
            pred_label = self.current_prediction
            pred_conf = self.current_confidence
            status = self.status_text

        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 72), (w, h), (15, 23, 42), -1)
        img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)

        cv2.putText(
            img,
            f"Translation: {pred_label} ({pred_conf * 100:.0f}%)" if pred_conf > 0 else pred_label,
            (16, h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (56, 189, 248),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            status,
            (16, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (203, 213, 225),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar Controls & Pipeline Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Model & Pipeline Settings")
st.sidebar.caption("Aligned with `sign Model.ipynb` training pipeline")

with st.sidebar.expander("🎯 Accuracy & Preprocessing", expanded=True):
    stride_opt = st.selectbox(
        "Frame Stride (Colab default = 2)",
        options=[1, 2, 3],
        index=1,
        help="Your Colab notebook extracted landmarks with `stride=2` (every 2nd frame).",
    )
    hand_conf_opt = st.slider(
        "MediaPipe Hand Confidence (Colab default = 0.5)",
        min_value=0.3,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Colab used default 0.5. Lower values (0.4) can introduce false-positive hand detections.",
    )
    trim_idle_opt = st.checkbox(
        "Trim Idle Start/End Frames (`sign_segment`)",
        value=False,
        help="Keep UNCHECKED to match Colab training (`dataset.npz` did not use `sign_segment`).",
    )
    fill_pose_opt = st.checkbox(
        "Forward-Fill Missing Pose (`FILL_MISSING_POSE`)",
        value=False,
        help="Keep UNCHECKED to match Colab training (`dataset.npz` normalized directly with scale fallback=1.0).",
    )
    mirror_input_opt = st.checkbox(
        "Mirror Video / Webcam Horizontally",
        value=False,
        help="Enable if your input video or camera is mirrored relative to training videos.",
    )
    flip_hand_opt = st.checkbox(
        "Swap Left/Right Handedness Labels",
        value=False,
        help="Toggle if MediaPipe Left/Right hand labels are inverted on your camera.",
    )

model, classes = load_model_and_classes()
st.sidebar.success(f"Loaded `isl_gru_v2.keras` ({len(classes)} sign classes)")


def render_prediction_results(probs, stats=None):
    """Display top predicted sentence and Top-5 probability breakdown."""
    if probs is None:
        st.warning(
            "Not enough hand landmarks were detected in the video. "
            "Ensure hands are clearly visible and well-lit."
        )
        return

    top5_idx = np.argsort(probs)[::-1][:5]
    best_idx = int(top5_idx[0])
    best_sentence = classes[best_idx]
    best_conf = float(probs[best_idx])

    col_main, col_top5 = st.columns([1.2, 1])
    with col_main:
        st.subheader("🗣️ Predicted Translation")
        st.markdown(
            f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; background: #0f172a; border-left: 6px solid #38bdf8; color: #f8fafc;">
                <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Recognized Sentence</div>
                <div style="font-size: 1.65rem; font-weight: 700; margin-top: 0.35rem; color: #38bdf8;">{best_sentence}</div>
                <div style="font-size: 1rem; margin-top: 0.5rem; color: #cbd5e1;">Confidence: <strong>{best_conf * 100:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if stats:
            st.caption(
                f"Processed **{stats['processed_frames']}** sampled frames "
                f"(from {stats['total_video_frames']} total frames @ {stats['fps']} FPS) · "
                f"Hands detected in **{stats['hand_detected_frames']}** frames."
            )

    with col_top5:
        st.subheader("📊 Top-5 Candidates")
        for rank, idx in enumerate(top5_idx, start=1):
            label = classes[int(idx)]
            score = float(probs[int(idx)])
            st.write(f"**{rank}. {label}** — `{score * 100:.1f}%`")
            st.progress(min(max(score, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Main Application UI
# ---------------------------------------------------------------------------
st.title("🤟 Indian Sign Language (ISL) to Text Translator")
st.write(
    "Translate Indian Sign Language gestures into English sentences using MediaPipe landmarks "
    "and a Bidirectional GRU neural network."
)

tab_live, tab_upload, tab_samples = st.tabs(
    ["🔴 Live Webcam Translation", "📁 Upload Sign Video", "🎬 Try Sample Clips"]
)

# --- TAB 1: LIVE WEBCAM TRANSLATION ---
with tab_live:
    st.subheader("Real-Time Live Webcam Translation")
    st.write(
        "Stream live video directly from your browser via WebRTC, or record a quick webcam clip below."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        live_mode = st.selectbox(
            "Live Translation Trigger",
            options=["Auto-Detect Sign (Recommended)", "Continuous Sliding Window"],
            help=(
                "Auto-Detect buffers frames while your hands are visible and translates the sentence "
                "as soon as you finish signing. Continuous mode runs on a rolling window."
            ),
        )
    with col_cfg2:
        window_frames = st.slider(
            "Max Sign Window (sampled frames)",
            min_value=24,
            max_value=64,
            value=45,
            step=4,
            help="Number of sampled frames (~2.5-3.5 seconds) before resampling to 32 frames.",
        )
    with col_cfg3:
        conf_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.15,
            max_value=0.85,
            value=0.35,
            step=0.05,
        )

    col_vid, col_live_out = st.columns([1.25, 1])
    with col_vid:
        webrtc_ctx = webrtc_streamer(
            key="isl-live-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=LiveSignProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.live_mode = live_mode
            webrtc_ctx.video_processor.window_frames = window_frames
            webrtc_ctx.video_processor.conf_threshold = conf_threshold
            webrtc_ctx.video_processor.stride = stride_opt
            webrtc_ctx.video_processor.trim_idle = trim_idle_opt
            webrtc_ctx.video_processor.fill_missing_pose = fill_pose_opt
            webrtc_ctx.video_processor.mirror_webcam = mirror_input_opt
            webrtc_ctx.video_processor.flip_handedness = flip_hand_opt

    with col_live_out:
        st.markdown("#### Live Translation Feed")
        live_placeholder = st.empty()
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            if st.button("🔄 Refresh Live Transcript"):
                pass
            with webrtc_ctx.video_processor.lock:
                pred = webrtc_ctx.video_processor.current_prediction
                conf = webrtc_ctx.video_processor.current_confidence
                top5_live = list(webrtc_ctx.video_processor.top5)
                hist_live = list(webrtc_ctx.video_processor.history)
                status_live = webrtc_ctx.video_processor.status_text

            st.info(f"**Status:** {status_live}")
            st.metric("Latest Translation", pred, f"{conf * 100:.1f}% confidence" if conf > 0 else None)

            if top5_live:
                st.markdown("**Current Top Candidates:**")
                for lbl, sc in top5_live[:3]:
                    st.write(f"- {lbl} (`{sc * 100:.1f}%`)")

            if hist_live:
                st.markdown("**Recent Translations:**")
                st.table(hist_live)
        else:
            st.caption("Click **START** on the video player to begin live webcam translation.")

# --- TAB 2: UPLOAD SIGN VIDEO ---
with tab_upload:
    st.subheader("Translate an Uploaded Video File")
    uploaded_file = st.file_uploader(
        "Upload a sign language video (.mp4, .mov, .avi, .webm)",
        type=["mp4", "mov", "avi", "webm"],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        col_u1, col_u2 = st.columns([1, 1.3])
        with col_u1:
            st.video(uploaded_file)

        with col_u2:
            prog = st.progress(0.0, text="Extracting MediaPipe Pose & Hand keypoints...")
            raw_kp, stats = extract_video_keypoints(
                tmp_path,
                width=640,
                stride=stride_opt,
                hand_conf=hand_conf_opt,
                mirror=mirror_input_opt,
                flip_handedness=flip_hand_opt,
                progress_bar=prog,
            )
            prog.empty()

            probs = pp.predict_probs(
                model,
                raw_kp,
                trim_idle=trim_idle_opt,
                fill_missing_pose=fill_pose_opt,
            )
            render_prediction_results(probs, stats=stats)

        try:
            os.remove(tmp_path)
        except OSError:
            pass

# --- TAB 3: SAMPLE VIDEOS ---
with tab_samples:
    st.subheader("Test with Repository Sample Videos")
    if SAMPLE_DIR.exists():
        sample_files = sorted(
            [p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}]
        )
    else:
        sample_files = []

    if not sample_files:
        st.info("No sample videos found in `sample_videos/` directory.")
    else:
        selected_sample = st.selectbox(
            "Choose a sample clip",
            options=sample_files,
            format_func=lambda p: p.name,
        )
        if selected_sample:
            col_s1, col_s2 = st.columns([1, 1.3])
            with col_s1:
                st.video(str(selected_sample))
            with col_s2:
                if st.button("▶️ Run Translation on Sample Clip", type="primary"):
                    prog = st.progress(0.0, text="Extracting MediaPipe landmarks...")
                    raw_kp, stats = extract_video_keypoints(
                        selected_sample,
                        width=640,
                        stride=stride_opt,
                        hand_conf=hand_conf_opt,
                        mirror=mirror_input_opt,
                        flip_handedness=flip_hand_opt,
                        progress_bar=prog,
                    )
                    prog.empty()
                    probs = pp.predict_probs(
                        model,
                        raw_kp,
                        trim_idle=trim_idle_opt,
                        fill_missing_pose=fill_pose_opt,
                    )
                    render_prediction_results(probs, stats=stats)
