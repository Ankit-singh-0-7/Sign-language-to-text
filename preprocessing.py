import numpy as np

# Default to False to match the exact normalization used when building dataset.npz in sign Model.ipynb
FILL_MISSING_POSE = False


def resample(seq, T=32):
    """Resample keypoint sequence to a fixed temporal length T=32."""
    if len(seq) == 0:
        return np.zeros((T, 258), dtype=np.float32)
    idx = np.linspace(0, len(seq) - 1, T).astype(int)
    return np.asarray(seq, dtype=np.float32)[idx]


def normalize(seq, fill_missing_pose=FILL_MISSING_POSE):
    """
    raw (N, 258): pose 33x4 | left hand 21x3 | right hand 21x3
    Normalized relative to mid-shoulder, scaled by shoulder width.

    Note: Setting fill_missing_pose=False reproduces the exact normalization
    used during training in `sign Model.ipynb`.
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n == 0:
        return np.zeros((0, 258), dtype=np.float32)

    pose = seq[:, :132].reshape(n, 33, 4).copy()
    lh = seq[:, 132:195].reshape(n, 21, 3).copy()
    rh = seq[:, 195:].reshape(n, 21, 3).copy()

    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm(ls - rs, axis=1)

    if fill_missing_pose:
        valid = (scale >= 1e-3) & ((np.abs(ls).sum(1) + np.abs(rs).sum(1)) > 0)
        if valid.any():
            idx = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
            idx[idx < 0] = np.argmax(valid)
            c_fill, s_fill = center[idx], scale[idx][:, None]
        else:
            c_fill, s_fill = np.zeros_like(center), np.ones((n, 1), dtype=np.float32)
    else:
        # Exact Colab training behavior: tiny scales become 1.0 and every frame is normalized
        scale = scale.copy()
        scale[scale < 1e-3] = 1.0
        valid = np.ones(n, dtype=bool)
        c_fill, s_fill = center, scale[:, None]

    pose[valid, :, :2] = (pose[valid, :, :2] - c_fill[valid, None]) / s_fill[valid, None]
    for h in (lh, rh):
        present = np.abs(h).sum((1, 2)) > 0
        h[present, :, :2] = (h[present, :, :2] - c_fill[present, None]) / s_fill[present, None]

    return np.concatenate([pose.reshape(n, -1), lh.reshape(n, -1), rh.reshape(n, -1)], axis=1).astype(np.float32)


def interp_missing(h):
    """Interpolate missing hand detections over time."""
    pres = np.abs(h).sum(1) > 0
    if pres.all() or not pres.any():
        return h
    t = np.arange(len(h))
    out = h.copy()
    for f in range(h.shape[1]):
        out[:, f] = np.interp(t, t[pres], h[pres, f])
    return out


def build(x):
    """
    Transform (32, 258) normalized keypoints into (32, 368) model input features:
      - 7 pose joints x 2D (nose, shoulders, elbows, wrists) = 14
      - Left hand global normalized (42) + wrist-relative scaled (42) = 84
      - Right hand global normalized (42) + wrist-relative scaled (42) = 84
      - Hand presence flags = 2
      - First-order temporal velocity (np.diff) = 184
    Total = 368 features per frame.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    pose = x[:, :132].reshape(n, 33, 4)[:, [0, 11, 12, 13, 14, 15, 16], :2].reshape(n, -1)
    pres = np.stack(
        [np.abs(x[:, 132:195]).sum(1) > 0, np.abs(x[:, 195:]).sum(1) > 0], axis=1
    ).astype(np.float32)

    parts = [pose]
    for h in (x[:, 132:195], x[:, 195:]):
        h_interp = interp_missing(h).reshape(n, 21, 3)[:, :, :2]
        rel = h_interp - h_interp[:, :1]
        size = np.linalg.norm(rel[:, 9], axis=1)[:, None, None]
        size[size < 1e-3] = 1.0
        parts += [h_interp.reshape(n, -1), (rel / size).reshape(n, -1)]

    f = np.concatenate(parts + [pres], axis=1)
    vel = np.diff(f, axis=0, prepend=f[:1])
    return np.concatenate([f, vel], axis=1).astype(np.float32)


def sign_segment(raw, pad=3, min_hand_frames=5):
    """
    Optional idle-frame trimmer: keeps from first to last frame where a hand is detected.
    Note: Disabled by default in predict_probs() for pre-recorded clips because
    isl_gru_v2.keras was trained on full un-trimmed clips resampled to 32 frames.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if len(raw) == 0:
        return None
    hand = np.abs(raw[:, 132:]).sum(1) > 0
    if hand.sum() < min_hand_frames:
        return None
    a, b = np.where(hand)[0][[0, -1]]
    return raw[max(a - pad, 0) : b + pad + 1]


def extract_frame_keypoints(pose_res, hand_res, flip_handedness=False):
    """
    Convert MediaPipe PoseLandmarkerResult and HandLandmarkerResult into a (258,) vector.
    Matches the exact `feats(pr, hr)` function from `sign Model.ipynb`.
    """
    if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
        pose = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in pose_res.pose_landmarks[0]],
            dtype=np.float32,
        ).ravel()
    else:
        pose = np.zeros(132, dtype=np.float32)

    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    if hand_res.hand_landmarks and hand_res.handedness:
        for lms, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32).ravel()
            label = hd[0].category_name
            if flip_handedness:
                label = "Right" if label == "Left" else "Left"
            if label == "Left":
                lh = arr
            else:
                rh = arr

    return np.concatenate([pose, lh, rh])


def predict_probs(
    model,
    raw_keypoints,
    trim_idle=False,
    fill_missing_pose=FILL_MISSING_POSE,
    min_hand_frames=3,
):
    """
    Predict class probabilities from raw keypoints array of shape (N, 258).

    Fixes for Colab vs. Streamlit accuracy drop:
      1. `trim_idle=False` by default so full video sequences are normalized first and
         then uniformly resampled to T=32, matching `resample(normalize(raw), 32)`
         in `sign Model.ipynb`.
      2. `fill_missing_pose=False` by default to match the training feature distribution.
    """
    raw = np.asarray(raw_keypoints, dtype=np.float32)
    if len(raw) == 0:
        return None

    # Require at least a few frames with detected hands so blank videos don't trigger random guesses
    hand_frames = int((np.abs(raw[:, 132:]).sum(1) > 0).sum())
    if hand_frames < min_hand_frames:
        return None

    clip = sign_segment(raw, min_hand_frames=min_hand_frames) if trim_idle else raw
    if clip is None or len(clip) == 0:
        return None

    # Exact Colab order: normalize(clip) -> resample(..., 32) -> build(...)
    normed = normalize(clip, fill_missing_pose=fill_missing_pose)
    sampled = resample(normed, T=32)
    features = build(sampled)
    return model.predict(features[None], verbose=0)[0]
