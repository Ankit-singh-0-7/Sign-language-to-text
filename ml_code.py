import torch
import torch.nn as nn
import torch.nn.functional as F

from google.colab import userdata
userdata.get('HF_TOKEN')

# Cell 1: Install dependencies
!pip install -q mediapipe opencv-python pyarrow

import os
from collections import Counter

corpus = os.path.join(path, "ISL_CSLRT_Corpus")
base = corpus if os.path.exists(corpus) else path
print("corpus exists:", os.path.exists(corpus))
print("base:", base)

for root, dirs, files in os.walk(base):
    depth = root.replace(base, "").count(os.sep)
    if depth <= 2:
        print("  " * depth + os.path.basename(root) + f"/  ({len(dirs)} dirs, {len(files)} files)")

all_files = [os.path.join(r, f) for r, _, fs in os.walk(base) for f in fs]
print("\n", len(all_files), "files")
print(Counter(os.path.splitext(f)[1].lower() for f in all_files))
for f in all_files[:10]:
    print(f)





# Cell 2: Core Preprocessing & Transformer Pipeline
import collections
import math
import time
from typing import Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. LANDMARK EXTRACTOR (MediaPipe -> 1D Vector)
# ==============================================================================
class LandmarkExtractor:
    """
    Extracts & flattens MediaPipe Holistic landmarks into a 1D vector (GISLR format).
    - Pose:       33 landmarks * (x, y, z) = 99 features
    - Left Hand:  21 landmarks * (x, y, z) = 63 features
    - Right Hand: 21 landmarks * (x, y, z) = 63 features
    Total: 75 landmarks * 3 = 225 features per frame (Face dropped to save compute).
    """
    POSE_COUNT = 33
    HAND_COUNT = 21

    def __init__(self, include_face: bool = False):
        self.include_face = include_face
        self.num_features = (self.POSE_COUNT + self.HAND_COUNT * 2) * 3

    def extract_from_mediapipe(self, results) -> np.ndarray:
        if results is None:
            return np.zeros(self.num_features, dtype=np.float32)

        # Pose (33 * 3 = 99)
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        else:
            pose = np.zeros(self.POSE_COUNT * 3, dtype=np.float32)

        # Left Hand (21 * 3 = 63)
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            lh = np.zeros(self.HAND_COUNT * 3, dtype=np.float32)

        # Right Hand (21 * 3 = 63)
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            rh = np.zeros(self.HAND_COUNT * 3, dtype=np.float32)

        return np.concatenate([pose, lh, rh])


# ==============================================================================
# 2. SLIDING WINDOW (Deque Buffer)
# ==============================================================================
class LandmarkSlidingWindow:
    """
    Maintains a rolling temporal window (e.g., 30 frames) with O(1) FIFO eviction.
    Yields PyTorch Tensors of shape (Batch_Size, Time_Frames, Num_Features).
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.extractor = LandmarkExtractor(include_face=False)
        self.num_features = self.extractor.num_features  # 225
        self.buffer = collections.deque(maxlen=window_size)

    def add_frame(self, results):
        feat = self.extractor.extract_from_mediapipe(results)
        self.buffer.append(feat)

    def add_vector(self, feat: np.ndarray):
        self.buffer.append(feat.astype(np.float32))

    def is_full(self) -> bool:
        return len(self.buffer) == self.window_size

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def get_window(self, to_torch: bool = True, device: str = "cpu") -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None

        # Replicate-pad if warming up
        if len(self.buffer) < self.window_size:
            pad_count = self.window_size - len(self.buffer)
            padding = [self.buffer[0]] * pad_count
            arr = np.array(padding + list(self.buffer), dtype=np.float32)
        else:
            arr = np.array(self.buffer, dtype=np.float32)

        # Add batch dimension: (1, 30, 225)
        arr = np.expand_dims(arr, axis=0)

        if to_torch:
            return torch.from_numpy(arr).float().to(device)
        return arr


# ==============================================================================
# 3. 1D-CNN FILTER (Spatial-Temporal Embed)
# ==============================================================================
class SpatialEmbed1DCNN(nn.Module):
    """
    1D-CNN layer projecting raw coordinate features (225) to embedding dimension (256)
    while modeling local frame dynamics. Preserves temporal length T = 30.
    """
    def __init__(self, in_features: int = 225, embed_dim: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        padding = (kernel_size - 1) // 2  # Padding to strictly keep T frames
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
        # Input shape: (Batch_Size, Time_Frames, Features) -> (B, 30, 225)
        B, T, C_in = x.shape

        # Permute for Conv1d: (B, 225, 30)
        x_conv = x.transpose(1, 2)
        res = self.residual_proj(x_conv)

        # 1D Convolutions over temporal axis
        out = self.drop1(self.act1(self.bn1(self.conv1(x_conv))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        out = out + res  # (B, 256, 30)

        # Transpose back for Transformer: (B, 30, 256)
        out = out.transpose(1, 2)
        return self.layer_norm(out)


# ==============================================================================
# 4. TRANSFORMER & CTC DECODER
# ==============================================================================
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
    """
    Complete Model: 1D-CNN -> Positional Encoding -> Transformer -> CTC Head
    """
    def __init__(self, in_features: int = 225, embed_dim: int = 256, num_classes: int = 264):
        super().__init__()
        self.spatial_embed = SpatialEmbed1DCNN(in_features=in_features, embed_dim=embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 225)
        tokens = self.spatial_embed(x)        # (B, 30, 256)
        tokens = self.pos_encoder(tokens)      # (B, 30, 256)
        encoded = self.transformer(tokens)    # (B, 30, 256)
        logits = self.classifier(encoded)     # (B, 30, num_classes)
        return logits


def ctc_greedy_decoder(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    """
    Decodes frame-level CTC logits by collapsing repeats and removing blanks.
    """
    # logits shape: (1, 30, num_classes)
    preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    decoded = []
    prev = -1
    for p in preds:
        if p != prev:
            if p != blank_id:
                decoded.append(p)
            prev = p
    return decoded

import pandas as pd
csvs = [f for f in all_files if f.endswith(".csv")]
print(csvs)
df = pd.read_csv(csvs[0])
print(df.shape)
df.head()

xl = [f for f in all_files if f.endswith((".xlsx", ".xls"))]
df = pd.read_excel(xl[0])
df.head()

import cv2, matplotlib.pyplot as plt
imgs = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
vids = [f for f in all_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
print(len(imgs), "images,", len(vids), "videos")

if imgs:
    plt.imshow(plt.imread(imgs[100])); plt.axis("off")

import os
vids = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
        if f.lower().endswith((".mp4", ".avi", ".mov"))]
print(len(vids))
for v in vids[:10]:
    print(v)

from IPython.display import Video
Video(vids[0], embed=True, width=480)

import os
from collections import Counter

corpus = os.path.join(path, "ISL_CSLRT_Corpus")
print("exists:", os.path.exists(corpus))
base = corpus if os.path.exists(corpus) else path

for root, dirs, files in os.walk(base):
    depth = root.replace(base, "").count(os.sep)
    if depth <= 2:
        print("  " * depth + os.path.basename(root) + f"/  ({len(dirs)} dirs, {len(files)} files)")

all_files = [os.path.join(r, f) for r, _, fs in os.walk(base) for f in fs]
print(len(all_files), "files")
print(Counter(os.path.splitext(f)[1].lower() for f in all_files))
for f in all_files[:10]: print(f)

# a) Folder names
from itertools import islice
for f in all_files[:5]:
    print(os.path.basename(os.path.dirname(f)), "|", os.path.basename(f))

# b) Label file
labels = [f for f in all_files if f.lower().endswith((".csv", ".txt", ".json", ".xlsx"))]
print(labels)

import pandas as pd
df = pd.read_csv(labels[0])   # or read_excel / open().read()
print(df.shape); display(df.head(10))

# If labels are folder names:
classes = Counter(os.path.basename(os.path.dirname(f)) for f in all_files if f.lower().endswith(".mp4"))
print(len(classes), "classes")
print(classes.most_common(10))
print("min clips/class:", min(classes.values()), "max:", max(classes.values()))

import cv2
vids = [f for f in all_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
for v in vids[:3]:
    cap = cv2.VideoCapture(v)
    print(os.path.basename(v), "| frames:", int(cap.get(7)), "| fps:", cap.get(5),
          "| size:", int(cap.get(3)), "x", int(cap.get(4)))
    cap.release()

import os, pandas as pd
root = os.path.join(base, "ISL_CSLRT_Corpus")
vid_root = os.path.join(root, "Videos_Sentence_Level")

rows = []
for cls in sorted(os.listdir(vid_root)):
    d = os.path.join(vid_root, cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith((".mp4", ".mov", ".avi")):
                rows.append((cls, os.path.join(d, f)))
idx = pd.DataFrame(rows, columns=["label", "path"])
print(len(idx), "videos,", idx.label.nunique(), "classes")
print(idx.label.value_counts().describe())
print(idx.label.value_counts().tail(10))      # smallest classes

# Label files
csv_dir = os.path.join(root, "corpus_csv_files")
g = pd.read_csv(os.path.join(csv_dir, "ISL Corpus sign glosses.csv"))
print(g.shape); display(g.head(10))
for f in os.listdir(csv_dir):
    if f.endswith(".xlsx"):
        x = pd.read_excel(os.path.join(csv_dir, f), sheet_name=None)
        for name, df in x.items():
            print("\n", f, "| sheet:", name, df.shape); display(df.head(5))

# Word-level frames
wl = os.path.join(root, "Frames_Word_Level")
wc = {w: len(os.listdir(os.path.join(wl, w))) for w in os.listdir(wl)}
print(len(wc), "word classes; sample:", list(wc.items())[:10])

# 1. Clear the redundant cache directory to free up ~8.3 GB of space
import shutil
import os

cache_dir = "/root/.cache/kagglehub"
if os.path.exists(cache_dir):
    print(f"Cleaning up cache at {cache_dir} to free up space...")
    shutil.rmtree(cache_dir)
    print("Cache cleared successfully!")

# 2. Re-try installing mediapipe now that disk space has been reclaimed
!pip install -q mediapipe

import mediapipe as mp
print("MediaPipe version:", mp.__version__, "| Solutions loaded:", hasattr(mp, "solutions"))

!pip install -q mediapipe
import mediapipe as mp
print(mp.__version__, hasattr(mp, "solutions"))

import os, urllib.request
os.makedirs("/content/models", exist_ok=True)
URLS = {
 "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
 "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
MODEL = {}
for k, u in URLS.items():
    p = f"/content/models/{k}.task"
    if not os.path.exists(p):
        urllib.request.urlretrieve(u, p)
    MODEL[k] = p
print(MODEL)

import cv2, numpy as np, mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

def make_models():
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL["pose"]),
        running_mode=vision.RunningMode.VIDEO))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL["hand"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return pose, hands

def feats(pr, hr):
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]]).ravel()
            if pr.pose_landmarks else np.zeros(132))
    lh, rh = np.zeros(63), np.zeros(63)
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms]).ravel()
        if hd[0].category_name == "Left": lh = arr
        else: rh = arr
    return np.concatenate([pose, lh, rh])

def process(path, width=640):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    pose, hands = make_models()          # new instance per video: timestamps must start fresh
    seq, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)))
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        ts = int(i * 1000 / fps)
        seq.append(feats(pose.detect_for_video(img, ts), hands.detect_for_video(img, ts)))
        i += 1
    cap.release(); pose.close(); hands.close()
    return np.array(seq, dtype=np.float32)

import cv2, numpy as np, mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

def make_models():
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL["pose"]),
        running_mode=vision.RunningMode.VIDEO))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL["hand"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return pose, hands

def feats(pr, hr):
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]]).ravel()
            if pr.pose_landmarks else np.zeros(132))
    lh, rh = np.zeros(63), np.zeros(63)
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms]).ravel()
        if hd[0].category_name == "Left": lh = arr
        else: rh = arr
    return np.concatenate([pose, lh, rh])

def process(path, width=640, stride=2):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    pose, hands = make_models()
    seq, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % stride == 0:                       # skip every other frame
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            ts = int(i * 1000 / fps)              # original frame index keeps timestamps increasing
            seq.append(feats(pose.detect_for_video(img, ts), hands.detect_for_video(img, ts)))
        i += 1
    cap.release(); pose.close(); hands.close()
    return np.array(seq, dtype=np.float32)

failed = []
for i in tqdm(range(len(idx))):
    out = f"{OUT}/{i:04d}.npy"
    if os.path.exists(out): continue              # resumable
    try:
        np.save(out, process(idx.path[i]))
    except Exception as e:
        failed.append((i, str(e)[:100]))
print("done. saved:", len(os.listdir(OUT)) - 1, "| failed:", failed)

from google.colab import drive
import time
drive.mount('/content/drive')
OUT = "/content/drive/MyDrive/isl_keypoints"; os.makedirs(OUT, exist_ok=True)
idx.to_csv(f"{OUT}/index.csv")           # `idx` from Cell A; row number = .npy file name

t = time.time()
for i in tqdm(range(5)):
    out = f"{OUT}/{i:04d}.npy"
    if not os.path.exists(out):
        np.save(out, process(idx.path[i]))
print("sec per video:", round((time.time() - t) / 5, 1))

x = np.load(f"{OUT}/0000.npy")
print(x.shape, "| frames with a hand:", (np.abs(x[:, 132:]).sum(1) > 0).mean().round(2),
      "| frames with pose:", (np.abs(x[:, :132]).sum(1) > 0).mean().round(2))

import os, json, numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')

OUT = "/content/drive/MyDrive/isl_keypoints"
idx = pd.read_csv(f"{OUT}/index.csv", index_col=0)    # row number = .npy file name
T = 32                                                # frames per clip after resampling

def resample(seq, T):
    if len(seq) == 0: return np.zeros((T, 258), np.float32)
    return seq[np.linspace(0, len(seq) - 1, T).astype(int)]

def normalize(seq):
    """Pose and hands relative to mid-shoulder, scaled by shoulder width."""
    n = len(seq)
    pose = seq[:, :132].reshape(n, 33, 4).copy()
    lh = seq[:, 132:195].reshape(n, 21, 3).copy()
    rh = seq[:, 195:].reshape(n, 21, 3).copy()
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True)
    scale[scale < 1e-3] = 1.0
    pose[:, :, :2] = (pose[:, :, :2] - center[:, None]) / scale[:, None]
    for h in (lh, rh):
        present = np.abs(h).sum((1, 2)) > 0           # leave missing hands as zeros
        h[present, :, :2] = (h[present, :, :2] - center[present, None]) / scale[present, None]
    return np.concatenate([pose.reshape(n, -1), lh.reshape(n, -1), rh.reshape(n, -1)], 1)

X, hand_rate = [], []
for i in range(len(idx)):
    raw = np.load(f"{OUT}/{i:04d}.npy")
    hand_rate.append((np.abs(raw[:, 132:]).sum(1) > 0).mean() if len(raw) else 0)
    X.append(resample(normalize(raw) if len(raw) else raw, T))
X = np.stack(X).astype(np.float32)

classes = sorted(idx.label.unique())
y = idx.label.map({c: k for k, c in enumerate(classes)}).values
print("X:", X.shape, "| classes:", len(classes))
print("clips with hands in <50% of frames:", int((np.array(hand_rate) < 0.5).sum()))

# Split per class: 1 val + 1 test when a class has >=4 clips, 1 test when it has 3, otherwise all train
rng = np.random.default_rng(42)
tr, va, te = [], [], []
for lab, grp in idx.groupby("label"):
    ids = rng.permutation(grp.index.values); n = len(ids)
    if n >= 4:   va.append(ids[0]); te.append(ids[1]); tr += list(ids[2:])
    elif n == 3: te.append(ids[0]); tr += list(ids[1:])
    else:        tr += list(ids)
tr, va, te = map(np.array, (tr, va, te))
print("train/val/test:", len(tr), len(va), len(te))

np.savez_compressed(f"{OUT}/dataset.npz", X=X, y=y, tr=tr, va=va, te=te)
json.dump(classes, open(f"{OUT}/classes.json", "w"))
print("saved dataset.npz and classes.json to Drive")

import os, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt

OUT = "/content/drive/MyDrive/isl_keypoints"
idx = pd.read_csv(f"{OUT}/index.csv", index_col=0)

rate = []
for i in range(len(idx)):
    raw = np.load(f"{OUT}/{i:04d}.npy")
    rate.append((np.abs(raw[:, 132:]).sum(1) > 0).mean() if len(raw) else 0)
idx["hand_rate"] = rate

print(idx.hand_rate.describe().round(2))
print("rate == 0:", (idx.hand_rate == 0).sum(), "| <0.2:", (idx.hand_rate < 0.2).sum(), "| <0.5:", (idx.hand_rate < 0.5).sum())

low = idx[idx.hand_rate < 0.5]
print("\nclasses with most low clips:")
print(low.label.value_counts().head(10))
print("\nworst 10:")
print(low.sort_values("hand_rate").head(10)[["label", "hand_rate", "path"]].to_string())

# look at the middle frame of the 4 worst clips
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, (_, r) in zip(axs, low.sort_values("hand_rate").head(4).iterrows()):
    cap = cv2.VideoCapture(r.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, f = cap.read(); cap.release()
    if ok: ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{r.label}\nrate={r.hand_rate:.2f}"); ax.axis("off")

import numpy as np, json, tensorflow as tf
from tensorflow.keras import layers as L

OUT = "/content/drive/MyDrive/isl_keypoints"
d = np.load(f"{OUT}/dataset.npz")
X, y, tr, va = d["X"], d["y"], d["tr"], d["va"]
n_cls = len(json.load(open(f"{OUT}/classes.json")))
rng = np.random.default_rng(0); T = 32

def interp_missing(h):                      # fill frames where the hand was not detected
    pres = np.abs(h).sum(1) > 0
    if pres.all() or not pres.any(): return h
    t = np.arange(len(h)); out = h.copy()
    for f in range(h.shape[1]): out[:, f] = np.interp(t, t[pres], h[pres, f])
    return out

def build(x):                               # (32,258) -> (32,368)
    n = len(x)
    pose = x[:, :132].reshape(n, 33, 4)[:, [0, 11, 12, 13, 14, 15, 16], :2].reshape(n, -1)
    pres = np.stack([np.abs(x[:, 132:195]).sum(1) > 0, np.abs(x[:, 195:]).sum(1) > 0], 1).astype(np.float32)
    parts = [pose]
    for h in (x[:, 132:195], x[:, 195:]):
        h = interp_missing(h).reshape(n, 21, 3)[:, :, :2]      # position relative to body
        rel = h - h[:, :1]                                      # shape relative to wrist
        size = np.linalg.norm(rel[:, 9], axis=1)[:, None, None]
        size[size < 1e-3] = 1
        parts += [h.reshape(n, -1), (rel / size).reshape(n, -1)]
    f = np.concatenate(parts + [pres], 1)
    vel = np.diff(f, axis=0, prepend=f[:1])
    return np.concatenate([f, vel], 1).astype(np.float32)

def augment(x):
    m = rng.integers(int(0.75 * T), T + 1); s = rng.integers(0, T - m + 1)
    x = x[np.linspace(s, s + m - 1, T).astype(int)]                       # temporal crop
    pose = x[:, :132].reshape(T, 33, 4).copy()
    lh = x[:, 132:195].reshape(T, 21, 3).copy(); rh = x[:, 195:].reshape(T, 21, 3).copy()
    ml, mr = np.abs(lh).sum((1, 2)) > 0, np.abs(rh).sum((1, 2)) > 0
    a = np.deg2rad(rng.uniform(-10, 10))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * rng.uniform(0.9, 1.1)
    sh = rng.normal(0, 0.05, 2)
    for arr in (pose, lh, rh): arr[:, :, :2] = arr[:, :, :2] @ R.T + sh   # rotate/scale/shift
    lh[~ml] = 0; rh[~mr] = 0                                              # keep missing = missing
    drop = rng.random(T) < 0.1; lh[drop] = 0; rh[drop] = 0                # simulated detection dropout
    return np.concatenate([pose.reshape(T, -1), lh.reshape(T, -1), rh.reshape(T, -1)], 1)

feats_set = lambda Xs: np.stack([build(a) for a in Xs])
K = 10
Ftr, Fva = feats_set(X[tr]), feats_set(X[va])
Fa = np.concatenate([Ftr] + [feats_set([augment(a) for a in X[tr]]) for _ in range(K)])
Ya = tf.keras.utils.to_categorical(np.concatenate([y[tr]] * (K + 1)), n_cls)
Yva = tf.keras.utils.to_categorical(y[va], n_cls)
print("train:", Fa.shape, "| val:", Fva.shape)

norm = L.Normalization(axis=-1); norm.adapt(Fa)
inp = L.Input((T, Fa.shape[-1]))
h = norm(inp)
h = L.Dense(96, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
h = L.Dropout(0.4)(h)
h = L.Bidirectional(L.GRU(64, return_sequences=True, dropout=0.2))(h)
h = L.Concatenate()([L.GlobalAveragePooling1D()(h), L.GlobalMaxPooling1D()(h)])
h = L.Dropout(0.5)(h)
model = tf.keras.Model(inp, L.Dense(n_cls, activation="softmax")(h))
model.compile("adam", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(5, name="top5")])

cb = [tf.keras.callbacks.EarlyStopping("val_accuracy", patience=15, restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=5)]
hist = model.fit(Fa, Ya, validation_data=(Fva, Yva), epochs=100, batch_size=64, callbacks=cb, verbose=2)
print("best val acc:", max(hist.history["val_accuracy"]), "| best val top5:", max(hist.history["val_top5"]))
model.save(f"{OUT}/isl_gru_v2.keras")
torch.save(model.state_dict(), 'isl_model.pth')

import numpy as np, json, pandas as pd
from collections import Counter

classes = json.load(open(f"{OUT}/classes.json"))
idx = pd.read_csv(f"{OUT}/index.csv", index_col=0)

p = model.predict(Fva, verbose=0)
pred, true = p.argmax(1), y[va]
ok = pred == true

# 1. Is hand detection the bottleneck?
hr = np.array([(np.abs(np.load(f"{OUT}/{i:04d}.npy")[:, 132:]).sum(1) > 0).mean() for i in va])
for lo, hi in [(0, .6), (.6, .8), (.8, 1.01)]:
    m = (hr >= lo) & (hr < hi)
    print(f"hand_rate {lo}-{min(hi,1):.1f}: n={m.sum():2d}  acc={ok[m].mean():.2f}")

# 2. Where does the right answer rank?
rank = (-p).argsort(1)
r = np.array([list(rank[i]).index(true[i]) + 1 for i in range(len(true))])
print("top-1/3/5/10:", [(r <= k).mean().round(2) for k in (1, 3, 5, 10)], "| median rank:", np.median(r))

# 3. Most common confusions
conf = Counter((classes[t], classes[q]) for t, q in zip(true[~ok], pred[~ok]))
for (t, q), c in conf.most_common(12):
    print(f"{t!r}  ->  {q!r}")

# 4. Do file names hint at sessions/signers? (3 example classes)
for c in classes[:3]:
    print(c, "|", sorted(os.path.basename(x) for x in idx[idx.label == c].path))

import os, cv2, numpy as np, pandas as pd

def grp(r):
    b = os.path.basename(r.path).lower()
    if b.startswith("mvi_"): return "A_MVI"
    return "C_fullname" if b.startswith(r.label.lower().strip()[:12]) else "B_keyword"
idx["group"] = idx.apply(grp, axis=1)
idx["hand_rate"] = [(np.abs(np.load(f"{OUT}/{i:04d}.npy")[:, 132:]).sum(1) > 0).mean() for i in idx.index]

print(idx.group.value_counts(), "\n")
print(idx.groupby("group").hand_rate.mean().round(2), "\n")

# val accuracy by group (ok = pred == true from the previous cell)
g = idx.group.values[va]
for k in sorted(set(g)):
    m = g == k
    print(f"val group {k}: n={m.sum():2d}  acc={ok[m].mean():.2f}")

# same camera or setup? check video properties of 3 clips per group
for k, sub in idx.groupby("group"):
    for pth in sub.path.iloc[:3]:
        c = cv2.VideoCapture(pth)
        print(k, int(c.get(3)), "x", int(c.get(4)), "| fps", round(c.get(5), 1), "| frames", int(c.get(7)))
        c.release()

# label problems
low = idx.label.str.lower().str.strip().value_counts()
print("\nclasses with <=2 clips:", dict(low[low <= 2]))
print("case-only duplicates:", idx.label.nunique() - idx.label.str.lower().str.strip().nunique())

import numpy as np, tensorflow as tf
from tensorflow.keras import layers as L

def run(train_ids, test_ids, epochs=40, K=6, seed=0):
    global rng
    rng = np.random.default_rng(seed)
    train_ids = np.array(train_ids)
    seen = set(y[train_ids])
    test_ids = np.array([i for i in test_ids if y[i] in seen])      # only classes the model has seen
    Fa = np.concatenate([feats_set(X[train_ids])] + [feats_set([augment(a) for a in X[train_ids]]) for _ in range(K)])
    Ya = tf.keras.utils.to_categorical(np.concatenate([y[train_ids]] * (K + 1)), n_cls)
    Fte = feats_set(X[test_ids])

    norm = L.Normalization(axis=-1); norm.adapt(Fa)
    inp = L.Input((T, Fa.shape[-1])); h = norm(inp)
    h = L.Dense(96, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = L.Dropout(0.4)(h)
    h = L.Bidirectional(L.GRU(64, return_sequences=True, dropout=0.2))(h)
    h = L.Concatenate()([L.GlobalAveragePooling1D()(h), L.GlobalMaxPooling1D()(h)])
    h = L.Dropout(0.5)(h)
    m = tf.keras.Model(inp, L.Dense(n_cls, activation="softmax")(h))
    m.compile("adam", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
    m.fit(Fa, Ya, epochs=epochs, batch_size=64, verbose=0)

    p = m.predict(Fte, verbose=0); t = y[test_ids]
    top5 = np.mean([t[i] in (-p[i]).argsort()[:5] for i in range(len(t))])
    return len(t), round(float((p.argmax(1) == t).mean()), 3), round(float(top5), 3)

g = idx.group.values
A, B, C = (np.where(g == k)[0] for k in ("A_MVI", "B_keyword", "C_fullname"))
AB = np.r_[A, B]

# within-session baseline: hold out one clip per class from B
r = np.random.default_rng(1)
hold = np.array([r.choice(s.index.values) for _, s in idx.iloc[B].groupby("label") if len(s) >= 3])
Btr = np.setdiff1d(B, hold)

print("(n_test, top1, top5)")
print("within B (holdout):   ", run(Btr, hold))
print("train A+B -> test C:  ", run(AB, C))
print("train C   -> test A+B:", run(C, AB))

import cv2, numpy as np, pandas as pd

def frame_aspect(path):                     # H/W of the frame MediaPipe actually received
    cap = cv2.VideoCapture(path); ok, f = cap.read(); cap.release()
    return f.shape[0] / f.shape[1] if ok else 0.5625
asp = np.array([frame_aspect(p) for p in idx.path])
print(pd.Series(asp.round(2)).value_counts())

def normalize_fixed(seq, a):
    n = len(seq)
    pose = seq[:, :132].reshape(n, 33, 4).copy()
    lh = seq[:, 132:195].reshape(n, 21, 3).copy()
    rh = seq[:, 195:].reshape(n, 21, 3).copy()
    for arr in (pose, lh, rh): arr[:, :, 1] *= a          # y into width units: x and y now same scale
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True); scale[scale < 1e-3] = 1.0
    pose[:, :, :2] = (pose[:, :, :2] - center[:, None]) / scale[:, None]
    for h in (lh, rh):
        present = np.abs(h).sum((1, 2)) > 0
        h[present, :, :2] = (h[present, :, :2] - center[present, None]) / scale[present, None]
    return np.concatenate([pose.reshape(n, -1), lh.reshape(n, -1), rh.reshape(n, -1)], 1)

X_old = X
X = np.stack([resample(normalize_fixed(np.load(f"{OUT}/{i:04d}.npy"), asp[i]), T)
              for i in range(len(idx))]).astype(np.float32)
np.save(f"{OUT}/X_aspectfixed.npy", X)

print("(n_test, top1, top5)")
print("within B (holdout):   ", run(Btr, hold))
print("train A+B -> test C:  ", run(AB, C))
print("train C   -> test A+B:", run(C, AB))

import numpy as np, tensorflow as tf
from tensorflow.keras import layers as L

def run(train_ids, test_ids, epochs=40, K=6, seed=0):
    global rng
    rng = np.random.default_rng(seed)
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    seen = set(int(c) for c in y[train_ids])
    keep = np.array([int(y[i]) in seen for i in test_ids], dtype=bool)
    print(f"   train clips: {len(train_ids)} | test clips: {len(test_ids)} | "
          f"test clips with a seen class: {keep.sum()} | y dtype: {y.dtype}, y len: {len(y)}")
    test_ids = test_ids[keep]
    if len(test_ids) == 0:
        return "no test clips with seen classes"

    Fa = np.concatenate([feats_set(X[train_ids])] +
                        [feats_set([augment(a) for a in X[train_ids]]) for _ in range(K)])
    Ya = tf.keras.utils.to_categorical(np.concatenate([y[train_ids]] * (K + 1)), n_cls)
    Fte = feats_set(X[test_ids])

    norm = L.Normalization(axis=-1); norm.adapt(Fa)
    inp = L.Input((T, Fa.shape[-1])); h = norm(inp)
    h = L.Dense(96, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = L.Dropout(0.4)(h)
    h = L.Bidirectional(L.GRU(64, return_sequences=True, dropout=0.2))(h)
    h = L.Concatenate()([L.GlobalAveragePooling1D()(h), L.GlobalMaxPooling1D()(h)])
    h = L.Dropout(0.5)(h)
    m = tf.keras.Model(inp, L.Dense(n_cls, activation="softmax")(h))
    m.compile("adam", tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
    m.fit(Fa, Ya, epochs=epochs, batch_size=64, verbose=0)

    p = m.predict(Fte, verbose=0); t = y[test_ids]
    top5 = np.mean([t[i] in (-p[i]).argsort()[:5] for i in range(len(t))])
    return len(t), round(float((p.argmax(1) == t).mean()), 3), round(float(top5), 3)

print("(n_test, top1, top5)")
print("A+B -> C landscape:", run(AB, C_land))
print("A+B -> C portrait: ", run(AB, C_port))
print("landscape -> portrait:", run(land, port))

import pandas as pd
d = pd.DataFrame({"label": idx.label.values, "landscape": asp < 1})
cnt = d.groupby("label").landscape.agg(["sum", "count"]).rename(columns={"sum": "landscape_clips", "count": "all_clips"})
print(cnt.sort_values("landscape_clips", ascending=False).head(40).to_string())

SENTENCES = [
    "help me", "hi how are you", "congratulations", "you are welcome",
    "take care of yourself", "you can do it", "how old are you", "are you free today",
    "do not worry", "i am crying", "i am feeling cold", "i am feeling bored",
    "speak softly", "go and sleep", "serve the food", "bring water for me",
    "comb your hair", "how dare you", "do me a favour", "he came by train",
]

import subprocess, os
from IPython.display import Video, display

def show(label, n=4):
    rows = idx[(idx.label == label) & (asp < 1)]          # landscape clips only
    src = rows.path.iloc[n]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=480:-2",
                    "-vcodec", "libx264", "-an", "/content/ref.mp4"])
    print(label, "|", os.path.basename(src))
    display(Video("/content/ref.mp4", embed=True, width=480))

show("hi how are you")          # change the label, and n=1,2... for other signers/takes

import os, glob, numpy as np, cv2
from google.colab import drive
drive.mount('/content/drive')

REC  = "/content/drive/MyDrive/my_recordings"
MYKP = "/content/drive/MyDrive/my_keypoints"; os.makedirs(MYKP, exist_ok=True)

def frame_aspect(path):
    cap = cv2.VideoCapture(path); ok, f = cap.read(); cap.release()
    return f.shape[0] / f.shape[1]

def segments(hand, merge_gap=8, min_len=15):
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

for label in sorted(os.listdir(REC)):
    d = os.path.join(REC, label)
    if not os.path.isdir(d): continue
    for f in sorted(glob.glob(f"{d}/*")):
        if not f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")): continue
        raw = process(f, stride=2)                      # same extractor as the dataset
        hand = np.abs(raw[:, 132:]).sum(1) > 0
        asp = frame_aspect(f)
        segs = segments(hand)
        print(f"\n{label!r} / {os.path.basename(f)}: {len(raw)} frames, aspect {asp:.2f}, "
              f"hands in {hand.mean():.2f} of frames -> found {len(segs)} sign(s)")
        for k, (a, b) in enumerate(segs):
            lo, hi = max(a - 3, 0), b + 4
            np.savez(f"{MYKP}/{label}__{os.path.splitext(os.path.basename(f))[0]}__{k}.npz",
                     raw=raw[lo:hi], aspect=asp, label=label)       # re-running overwrites, no duplicates
            print(f"   sign {k}: frames={hi-lo:3d}  hands={hand[lo:hi].mean():.2f}")

import os
REC = "/content/drive/MyDrive/my_recordings"
print("exists:", os.path.exists(REC))
for root, dirs, files in os.walk(REC):
    print(root, "| dirs:", dirs, "| files:", files)

import os, shutil
REC = "/content/drive/MyDrive/my_recordings"
src = os.path.join(REC, "MyDrive my_recordings help me clip01.mp4")
dst_dir = os.path.join(REC, "help me")
os.makedirs(dst_dir, exist_ok=True)
shutil.move(src, os.path.join(dst_dir, "clip01.mp4"))

for root, dirs, files in os.walk(REC):
    print(root, "| dirs:", dirs, "| files:", files)

import os, glob, numpy as np, cv2

REC  = "/content/drive/MyDrive/my_recordings"
MYKP = "/content/drive/MyDrive/my_keypoints"; os.makedirs(MYKP, exist_ok=True)

def frame_aspect(path):
    cap = cv2.VideoCapture(path); ok, f = cap.read(); cap.release()
    return f.shape[0] / f.shape[1]

def segments(hand, merge_gap=8, min_len=15):
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

for label in sorted(os.listdir(REC)):
    d = os.path.join(REC, label)
    if not os.path.isdir(d): continue
    for f in sorted(glob.glob(f"{d}/*")):
        if not f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")): continue
        raw = process(f, stride=2)
        hand = np.abs(raw[:, 132:]).sum(1) > 0
        asp = frame_aspect(f)
        segs = segments(hand)
        print(f"\n{label!r} / {os.path.basename(f)}: {len(raw)} frames, aspect {asp:.2f}, "
              f"hands in {hand.mean():.2f} of frames -> found {len(segs)} sign(s)")
        for k, (a, b) in enumerate(segs):
            lo, hi = max(a - 3, 0), b + 4
            np.savez(f"{MYKP}/{label}__{os.path.splitext(os.path.basename(f))[0]}__{k}.npz",
                     raw=raw[lo:hi], aspect=asp, label=label)
            print(f"   sign {k}: frames={hi-lo:3d}  hands={hand[lo:hi].mean():.2f}")











import difflib, pandas as pd

MY = ["help me", "good morning", "hi how are you", "thank you"]   # EDIT: your sentences

labels = list(idx.label.unique())
low = {l.lower().strip(): l for l in labels}
cnt = pd.DataFrame({"label": idx.label.values, "landscape": asp < 1}).groupby("label").landscape.agg(["count", "sum"])

for s in MY:
    key = s.lower().strip()
    if key in low:
        l = low[key]
        print(f"IN DATASET   {s!r:30} -> use folder name {l!r} | clips: {cnt.loc[l,'count']} (landscape {int(cnt.loc[l,'sum'])})")
    else:
        near = difflib.get_close_matches(key, list(low), n=3, cutoff=0.5)
        print(f"NOT IN       {s!r:30} -> new class, record 12+ clips | similar in dataset: {[low[n] for n in near]}")

check_clip("/content/drive/MyDrive/my_recordings/hi how are you/clip01.mp4")

import glob, json, numpy as np, tensorflow as tf
OUT, MYKP = "/content/drive/MyDrive/isl_keypoints", "/content/drive/MyDrive/my_keypoints"
classes = json.load(open(f"{OUT}/classes.json"))
model = tf.keras.models.load_model(f"{OUT}/isl_gru_v2.keras")      # dataset-only model

for f in sorted(glob.glob(f"{MYKP}/*.npz")):
    d = np.load(f, allow_pickle=True); lab = str(d["label"])
    x = build(resample(normalize(d["raw"]), 32))                    # same functions the model was trained with
    p = model.predict(x[None], verbose=0)[0]; top = p.argsort()[::-1][:3]
    print(f"{lab!r:20} -> {[(classes[i], round(float(p[i]), 2)) for i in top]}")

import os
for root, dirs, files in os.walk("/content/drive/MyDrive/my_recordings"):
    print(root, "|", dirs, "|", files)

import os, glob, numpy as np, cv2
REC  = "/content/drive/MyDrive/my_recordings"
MYKP = "/content/drive/MyDrive/my_keypoints"; os.makedirs(MYKP, exist_ok=True)

def frame_aspect(path):
    cap = cv2.VideoCapture(path); ok, f = cap.read(); cap.release()
    return f.shape[0] / f.shape[1]

def segments(hand, merge_gap=8, min_len=15):
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

for label in sorted(os.listdir(REC)):
    d = os.path.join(REC, label)
    if not os.path.isdir(d): continue
    for f in sorted(glob.glob(f"{d}/*")):
        if not f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")): continue
        raw = process(f, stride=2)
        if raw.ndim != 2: print("COULD NOT READ:", label, f); continue
        hand = np.abs(raw[:, 132:]).sum(1) > 0
        asp = frame_aspect(f); segs = segments(hand)
        print(f"{label!r} / {os.path.basename(f)}: {len(raw)} frames, hands {hand.mean():.2f} -> found {len(segs)} sign(s)")
        for k, (a, b) in enumerate(segs):
            lo, hi = max(a - 3, 0), b + 4
            np.savez(f"{MYKP}/{label}__{os.path.splitext(os.path.basename(f))[0]}__{k}.npz",
                     raw=raw[lo:hi], aspect=asp, label=label)

import glob, json, numpy as np, tensorflow as tf
OUT, MYKP = "/content/drive/MyDrive/isl_keypoints", "/content/drive/MyDrive/my_keypoints"
classes = json.load(open(f"{OUT}/classes.json"))
model = tf.keras.models.load_model(f"{OUT}/isl_gru_v2.keras")

for f in sorted(glob.glob(f"{MYKP}/*.npz")):
    d = np.load(f, allow_pickle=True); lab = str(d["label"])
    x = build(resample(normalize(d["raw"]), 32))
    p = model.predict(x[None], verbose=0)[0]; top = p.argsort()[::-1][:3]
    print(f"{lab!r:20} -> {[(classes[i], round(float(p[i]), 2)) for i in top]}")

from google.colab import files
OUT = "/content/drive/MyDrive/isl_keypoints"
files.download(f"{OUT}/isl_gru_v2.keras")
files.download(f"{OUT}/classes.json")

import json, numpy as np, tensorflow as tf
from IPython.display import JSON
from google.colab import output

OUT = "/content/drive/MyDrive/isl_keypoints"
classes = json.load(open(f"{OUT}/classes.json"))
model = tf.keras.models.load_model(f"{OUT}/isl_gru_v2.keras")      # dataset-only model for now
DEMO = ["help me", "hi how are you"]                               # None = all 101 sentences
mask = np.ones(len(classes)) if DEMO is None else np.array([float(c in DEMO) for c in classes])
model.predict(np.zeros((1, 32, 368), np.float32), verbose=0)       # warm-up so the first sign isn't slow

def isl_predict(payload):
    try:
        raw = np.array(json.loads(payload), dtype=np.float32)       # frames x 258, same layout as training
        hand = np.abs(raw[:, 132:]).sum(1) > 0
        if hand.sum() < 5: return JSON({"error": "hands barely detected"})
        a, b = np.where(hand)[0][[0, -1]]
        clip = raw[max(a - 3, 0): b + 4]                            # same trimming as your recordings
        x = build(resample(normalize(clip), 32))
        p = model.predict(x[None], verbose=0)[0] * mask
        p = p / p.sum()
        top = p.argsort()[::-1][:3]
        return JSON({"top": [[classes[i], float(p[i])] for i in top], "frames": int(len(clip))})
    except Exception as e:
        return JSON({"error": str(e)})

output.register_callback("notebook.isl_predict", isl_predict)
print("ready")

from IPython.display import display, Javascript

JS = r"""
(async () => {
const V_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const POSE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const HAND = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';
const PAD = 6, END_PAUSE = 550, MIN_SIGN = 800, MAX_SIGN = 8000, CONF_MIN = 0.5;

const root = document.createElement('div');
root.style.cssText = 'position:relative;width:640px;max-width:100%;font-family:monospace;background:#111';
const video = document.createElement('video'); video.autoplay = true; video.muted = true; video.playsInline = true;
video.style.cssText = 'width:100%;display:block;transform:scaleX(-1)';          // mirror for display only
const canvas = document.createElement('canvas');
canvas.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;transform:scaleX(-1)';
const panel = document.createElement('div');
panel.style.cssText = 'position:absolute;left:8px;top:8px;background:rgba(0,0,0,.7);color:#fff;padding:6px 10px;font-size:13px;border:1px solid #2ecc71;border-radius:4px;white-space:pre';
const result = document.createElement('div');
result.style.cssText = 'position:absolute;left:8px;right:8px;bottom:8px;background:rgba(0,0,0,.7);color:#fff;padding:6px 10px;font-size:16px;white-space:pre;border-radius:4px';
const btn = document.createElement('button'); btn.textContent = 'Stop camera';
root.append(video, canvas, panel, result); document.body.append(root, btn);
let stopped = false;

try {
  panel.textContent = 'Loading MediaPipe...';
  const V = await import(V_URL + '/vision_bundle.mjs');
  const fileset = await V.FilesetResolver.forVisionTasks(V_URL + '/wasm');
  const mk = async (cls, url, extra) => {
    for (const d of ['GPU', 'CPU']) {
      try { return await cls.createFromOptions(fileset, {baseOptions: {modelAssetPath: url, delegate: d}, runningMode: 'VIDEO', ...extra}); }
      catch (e) { console.log(d, e); }
    }
    throw new Error('could not load ' + url);
  };
  const pose = await mk(V.PoseLandmarker, POSE, {});
  const hands = await mk(V.HandLandmarker, HAND, {numHands: 2});

  const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 360}}});
  video.srcObject = stream; await video.play();
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const ASP = (video.videoHeight / video.videoWidth).toFixed(2);
  google.colab.output.setIframeHeight(root.offsetHeight + 60, true);
  btn.onclick = () => { stopped = true; stream.getTracks().forEach(t => t.stop()); panel.textContent = 'stopped'; };

  const ctx = canvas.getContext('2d'), du = new V.DrawingUtils(ctx);
  let state = 'idle', seg = [], pre = [], tStart = 0, tLast = 0, lastIdx = 0, lastT = -1, n = 0, t0 = performance.now(), fps = 0;

  async function send(s) {
    result.textContent = 'thinking...';
    const sub = s.filter((_, i) => i % 2 === 0).map(r => r.map(x => Math.round(x * 1e4) / 1e4));   // stride 2, like training
    const r = await google.colab.kernel.invokeFunction('notebook.isl_predict', [JSON.stringify(sub)], {});
    const out = r.data['application/json'];
    if (out.error) { result.textContent = 'error: ' + out.error; return; }
    const [lab, conf] = out.top[0];
    result.textContent = (conf >= CONF_MIN ? lab.toUpperCase() : 'not sure') + '\n' +
      out.top.map(t => t[0] + '  ' + t[1].toFixed(2)).join('\n');
  }

  function loop() {
    if (stopped) return;
    requestAnimationFrame(loop);
    if (video.currentTime === lastT) return; lastT = video.currentTime;
    const now = performance.now();
    const pr = pose.detectForVideo(video, now), hr = hands.detectForVideo(video, now);
    const hd = hr.handedness || hr.handednesses;

    const v = new Array(258).fill(0);                                    // same layout as training: pose 132 | left 63 | right 63
    const P = pr.landmarks && pr.landmarks[0];
    if (P) P.forEach((p, i) => { v[i*4] = p.x; v[i*4+1] = p.y; v[i*4+2] = p.z; v[i*4+3] = p.visibility ?? 0; });
    let L = null, R = null;
    hr.landmarks.forEach((lm, i) => {
      const side = hd[i][0].categoryName, off = side === 'Left' ? 132 : 195;
      lm.forEach((p, j) => { v[off+j*3] = p.x; v[off+j*3+1] = p.y; v[off+j*3+2] = p.z; });
      if (side === 'Left') L = lm; else R = lm;
    });

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hr.landmarks.forEach(lm => {
      du.drawConnectors(lm, V.HandLandmarker.HAND_CONNECTIONS, {color: '#ff3cff', lineWidth: 2});
      du.drawLandmarks(lm, {color: '#ff3cff', radius: 3});
    });
    if (P) du.drawLandmarks([0, 11, 12, 13, 14, 15, 16].map(i => P[i]), {color: '#2ecc71', radius: 4});

    const has = hr.landmarks.length > 0;
    if (state === 'idle') {
      pre.push(v); if (pre.length > PAD) pre.shift();
      if (has) { state = 'signing'; seg = pre.slice(); tStart = tLast = now; lastIdx = seg.length - 1; }
    } else {
      seg.push(v);
      if (has) { tLast = now; lastIdx = seg.length - 1; }
      if (now - tLast > END_PAUSE || now - tStart > MAX_SIGN) {
        if (tLast - tStart >= MIN_SIGN) send(seg.slice(0, lastIdx + 1 + PAD));
        state = 'idle'; seg = []; pre = [];
      }
    }

    if (++n % 20 === 0) { fps = 20000 / (now - t0); t0 = now; }
    const f = h => h ? 'x=' + h[8].x.toFixed(2) + ' y=' + h[8].y.toFixed(2) : '[NOT DETECTED]';
    panel.textContent = 'ISL telemetry  ' + fps.toFixed(0) + ' FPS  aspect ' + ASP + '\nRH: ' + f(R) + '\nLH: ' + f(L) +
      '\nstate: ' + (state === 'signing' ? 'SIGNING ' + ((now - tStart) / 1000).toFixed(1) + 's' : 'ready (hands down)');
  }
  loop();
} catch (e) { panel.textContent = 'ERROR: ' + e.message; console.error(e); }
})();
"""
display(Javascript(JS))

!pip install -q huggingface_hub
from huggingface_hub import login
login()      # paste the write token

import os, shutil

OUT  = "/content/drive/MyDrive/isl_keypoints"
REPO = "ankit79/isl-sentence-gru"            # EDIT
pkg  = "/content/hf_upload"; os.makedirs(pkg, exist_ok=True)

src = f"{OUT}/isl_gru_v2_keep.keras" if os.path.exists(f"{OUT}/isl_gru_v2_keep.keras") else f"{OUT}/isl_gru_v2.keras"
shutil.copy(src, f"{pkg}/isl_gru_v2.keras")
shutil.copy(f"{OUT}/classes.json", f"{pkg}/classes.json")

open(f"{pkg}/preprocessing.py", "w").write(r'''
import numpy as np

def resample(seq, T=32):
    return seq[np.linspace(0, len(seq) - 1, T).astype(int)]

def normalize(seq):
    """raw (N,258): pose 33x4 | left hand 21x3 | right hand 21x3 -> body-relative, shoulder-width scaled."""
    n = len(seq)
    pose = seq[:, :132].reshape(n, 33, 4).copy()
    lh = seq[:, 132:195].reshape(n, 21, 3).copy()
    rh = seq[:, 195:].reshape(n, 21, 3).copy()
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True); scale[scale < 1e-3] = 1.0
    pose[:, :, :2] = (pose[:, :, :2] - center[:, None]) / scale[:, None]
    for h in (lh, rh):
        present = np.abs(h).sum((1, 2)) > 0
        h[present, :, :2] = (h[present, :, :2] - center[present, None]) / scale[present, None]
    return np.concatenate([pose.reshape(n, -1), lh.reshape(n, -1), rh.reshape(n, -1)], 1)

def interp_missing(h):
    pres = np.abs(h).sum(1) > 0
    if pres.all() or not pres.any(): return h
    t = np.arange(len(h)); out = h.copy()
    for f in range(h.shape[1]): out[:, f] = np.interp(t, t[pres], h[pres, f])
    return out

def build(x):
    """(32,258) normalized -> (32,368) model input."""
    n = len(x)
    pose = x[:, :132].reshape(n, 33, 4)[:, [0, 11, 12, 13, 14, 15, 16], :2].reshape(n, -1)
    pres = np.stack([np.abs(x[:, 132:195]).sum(1) > 0, np.abs(x[:, 195:]).sum(1) > 0], 1).astype(np.float32)
    parts = [pose]
    for h in (x[:, 132:195], x[:, 195:]):
        h = interp_missing(h).reshape(n, 21, 3)[:, :, :2]
        rel = h - h[:, :1]
        size = np.linalg.norm(rel[:, 9], axis=1)[:, None, None]; size[size < 1e-3] = 1
        parts += [h.reshape(n, -1), (rel / size).reshape(n, -1)]
    f = np.concatenate(parts + [pres], 1)
    return np.concatenate([f, np.diff(f, axis=0, prepend=f[:1])], 1).astype(np.float32)

def sign_segment(raw, pad=3):
    """Trim idle frames: keep from first to last frame where a hand is detected."""
    hand = np.abs(raw[:, 132:]).sum(1) > 0
    if hand.sum() < 5: return None
    a, b = np.where(hand)[0][[0, -1]]
    return raw[max(a - pad, 0): b + pad + 1]

def predict_probs(model, raw_keypoints):
    clip = sign_segment(raw_keypoints)
    if clip is None: return None
    return model.predict(build(resample(normalize(clip), 32))[None], verbose=0)[0]
''')

open(f"{pkg}/README.md", "w").write("""---
library_name: keras
tags: [sign-language, indian-sign-language, mediapipe, gru]
---
# ISL sentence classifier (GRU on MediaPipe keypoints)

101-way classifier over fixed Indian Sign Language sentences, trained on the ISL-CSLRT dataset
(Kaggle: drblack00/isl-csltr-indian-sign-language-dataset).

**Input:** (1, 32, 368). Keypoints per frame from MediaPipe Pose + Hand landmarkers (258 values: pose 33x4,
left hand 21x3, right hand 21x3), frames at stride 2 from video resized to 640 px wide, then
`preprocessing.normalize -> resample(32) -> build`. **Output:** softmax over `classes.json`.

## Results (honest)
| Test | Top-1 | Top-5 |
|---|---|---|
| Random split, 98 val clips | ~36% | ~66% |
| Within one recording setup (holdout) | ~58% | ~77% |
| Across recording setups (different camera/signers) | 1-9% | 12-31% |

The model mostly recognises the recording setup it was trained on and generalises poorly to new cameras and signers.
Fine-tune on your own setup before relying on it. Closed set of 101 sentences, no open-vocabulary translation.
Research/demo use only, not for real communication.
""")
print(os.listdir(pkg))

from huggingface_hub import HfApi, create_repo
create_repo(REPO, private=True, exist_ok=True)
HfApi().upload_folder(folder_path=pkg, repo_id=REPO, commit_message="ISL GRU v2")
print("https://huggingface.co/" + REPO)

from huggingface_hub import snapshot_download
import sys, json, tensorflow as tf, glob, numpy as np
d = snapshot_download("ankit79/isl-sentence-gru")      # private repo: run login() first
sys.path.insert(0, d); import preprocessing as pp
model = tf.keras.models.load_model(f"{d}/isl_gru_v2.keras")
classes = json.load(open(f"{d}/classes.json"))

# Load actual raw keypoints of shape (frames, 258)
f = sorted(glob.glob("/content/drive/MyDrive/my_keypoints/*.npz"))[0]
raw_keypoints = np.load(f, allow_pickle=True)["raw"]

probs = pp.predict_probs(model, raw_keypoints)    # raw_keypoints: (frames, 258) from MediaPipe
print(classes[probs.argmax()], probs.max())

import glob, numpy as np
f = sorted(glob.glob("/content/drive/MyDrive/my_keypoints/*.npz"))[0]
d = np.load(f, allow_pickle=True)
raw_keypoints = d["raw"]                 # (frames, 258)
print("clip:", f.split("/")[-1], "| true label:", d["label"], "| shape:", raw_keypoints.shape)

probs = pp.predict_probs(model, raw_keypoints)
top = probs.argsort()[::-1][:3]
print([(classes[i], round(float(probs[i]), 2)) for i in top])