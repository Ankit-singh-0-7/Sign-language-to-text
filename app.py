import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

st.set_page_config(page_title="Sign Language Translator", layout="wide")

st.title("Sign Language Translation")
st.write("Upload a video of a sign, and the model will predict the translation.")

# ==============================================================================
# Model Architecture (PyTorch)
# ==============================================================================
class SpatialEmbed1DCNN(nn.Module):
    def __init__(self, in_features: int = 225, embed_dim: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        padding = (kernel_size - 1) // 2
        mid_dim = embed_dim // 2

        self.conv1 = nn.Conv1d(in_features, mid_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(mid_dim, embed_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual_proj = nn.Conv1d(in_features, embed_dim, kernel_size=1, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C_in = x.shape
        x_conv = x.transpose(1, 2)
        res = self.residual_proj(x_conv)
        out = self.drop1(self.act1(self.bn1(self.conv1(x_conv))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        out = out + res
        out = out.transpose(1, 2)
        return self.layer_norm(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class ISLTransformerPipeline(nn.Module):
    def __init__(self, in_features: int = 225, embed_dim: int = 256, num_classes: int = 264):
        super().__init__()
        self.spatial_embed = SpatialEmbed1DCNN(in_features=in_features, embed_dim=embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.spatial_embed(x)
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer(tokens)
        logits = self.classifier(encoded)
        return logits

# ==============================================================================
# Helper functions for processing
# ==============================================================================
@st.cache_resource
def load_mediapipe_models():
    # Streamlit Cloud might not have GPU, CPU is fine.
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    import urllib.request
    
    # Download models if they don't exist
    os.makedirs("models", exist_ok=True)
    pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    
    if not os.path.exists("models/pose_landmarker_full.task"):
        urllib.request.urlretrieve(pose_url, "models/pose_landmarker_full.task")
    if not os.path.exists("models/hand_landmarker.task"):
        urllib.request.urlretrieve(hand_url, "models/hand_landmarker.task")
        
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
        running_mode=vision.RunningMode.VIDEO)
    hand_options = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    
    pose = vision.PoseLandmarker.create_from_options(pose_options)
    hands = vision.HandLandmarker.create_from_options(hand_options)
    return pose, hands

def extract_features(pr, hr):
    pose = (np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks[0]]).ravel()
            if pr.pose_landmarks else np.zeros(99))
    lh, rh = np.zeros(63), np.zeros(63)
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms]).ravel()
        if hd[0].category_name == "Left": lh = arr
        else: rh = arr
    return np.concatenate([pose, lh, rh])

def process_video(path, pose_model, hands_model, width=640, stride=2, max_frames=30):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    seq, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % stride == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            ts = int(i * 1000 / fps)
            try:
                pr = pose_model.detect_for_video(img, ts)
                hr = hands_model.detect_for_video(img, ts)
                seq.append(extract_features(pr, hr))
            except Exception:
                pass
        i += 1
    cap.release()
    
    if len(seq) == 0:
        return np.zeros((1, max_frames, 225), dtype=np.float32)
        
    seq = np.array(seq, dtype=np.float32)
    
    # Resample to T frames (e.g., 30)
    T = max_frames
    resampled = seq[np.linspace(0, len(seq) - 1, T).astype(int)]
    return np.expand_dims(resampled, axis=0) # Add batch dim

# ==============================================================================
# UI
# ==============================================================================
uploaded_file = st.file_uploader("Upload Sign Language Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    st.video(temp_video_path)

    if st.button("Translate"):
        with st.spinner("Extracting features using MediaPipe..."):
            pose_model, hands_model = load_mediapipe_models()
            features = process_video(temp_video_path, pose_model, hands_model, max_frames=30)
            
        with st.spinner("Running model prediction..."):
            model_path = 'isl_model.pth'
            classes_path = 'classes.json'
            
            if os.path.exists(model_path) and os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    classes = json.load(f)
                num_classes = len(classes)
                
                model = ISLTransformerPipeline(in_features=225, num_classes=num_classes)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                
                with torch.no_grad():
                    input_tensor = torch.tensor(features)
                    logits = model(input_tensor)
                    # Simplified prediction logic
                    pred = torch.argmax(logits.mean(dim=1), dim=-1).item()
                    prediction_text = classes[pred]
                    
                st.success(f"**Predicted Sign:** {prediction_text}")
            else:
                st.warning("⚠️ Model file `isl_model.pth` or `classes.json` not found in the directory.")
                st.info("Since this is a demo, please place the downloaded model and classes.json in the app folder to get actual predictions. Below is a mock output:")
                st.success(f"**Predicted Sign (Mock):** hello")
                
    # Clean up temp file
    try:
        os.remove(temp_video_path)
    except:
        pass
