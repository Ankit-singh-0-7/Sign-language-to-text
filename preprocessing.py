import numpy as np

# When pose detection fails on a frame, reuse the nearest earlier valid shoulder
# center/scale for that frame's hands instead of leaving them in raw image units.
# Set to False to reproduce the original behaviour exactly (use this if your model
# was trained with the original normalize() and you do not want to retrain).
FILL_MISSING_POSE = True


def resample(seq, T=32):
    """Resample keypoint sequence to a fixed temporal length T."""
    if len(seq) == 0:
        return np.zeros((T, 258), dtype=np.float32)
    return seq[np.linspace(0, len(seq) - 1, T).astype(int)]


def normalize(seq, fill_missing_pose=FILL_MISSING_POSE):
    """
    raw (N,258): pose 33x4 | left hand 21x3 | right hand 21x3
    Normalized relative to mid-shoulder, scaled by shoulder width.
    """
    n = len(seq)
    pose = seq[:, :132].reshape(n, 33, 4).copy()
    lh = seq[:, 132:195].reshape(n, 21, 3).copy()
    rh = seq[:, 195:].reshape(n, 21, 3).copy()

    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1)

    if fill_missing_pose:
        # A frame is valid when both shoulders were detected and are far enough apart.
        valid = (scale >= 1e-3) & ((np.abs(ls).sum(1) + np.abs(rs).sum(1)) > 0)
        if valid.any():
            # Index of the most recent valid frame at or before each frame;
            # frames before the first valid one use the first valid frame.
            idx = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
            idx[idx < 0] = np.argmax(valid)
            c_fill, s_fill = center[idx], scale[idx][:, None]
        else:
            c_fill, s_fill = np.zeros_like(center), np.ones((n, 1))
    else:
        # Original behaviour: tiny scales become 1.0 and every frame is normalized.
        scale = scale.copy()
        scale[scale < 1e-3] = 1.0
        valid = np.ones(n, dtype=bool)
        c_fill, s_fill = center, scale[:, None]

    pose[valid, :, :2] = (pose[valid, :, :2] - c_fill[valid, None]) / s_fill[valid, None]
    for h in (lh, rh):
        present = np.abs(h).sum((1, 2)) > 0
        h[present, :, :2] = (h[present, :, :2] - c_fill[present, None]) / s_fill[present, None]

    return np.concatenate([pose.reshape(n, -1), lh.reshape(n, -1), rh.reshape(n, -1)], 1)


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
    Transform (32,258) normalized into (32,368) model input feature representation.
    Extracts key pose joints (shoulders, elbows, wrists, nose), hand poses relative to wrist,
    hand scale normalization, presence indicators, and velocity deltas.
    """
    n = len(x)
    pose = x[:, :132].reshape(n, 33, 4)[:, [0, 11, 12, 13, 14, 15, 16], :2].reshape(n, -1)
    pres = np.stack([np.abs(x[:, 132:195]).sum(1) > 0, np.abs(x[:, 195:]).sum(1) > 0], 1).astype(np.float32)
    parts = [pose]
    for h in (x[:, 132:195], x[:, 195:]):
        h = interp_missing(h).reshape(n, 21, 3)[:, :, :2]
        rel = h - h[:, :1]
        size = np.linalg.norm(rel[:, 9], axis=1)[:, None, None]
        size[size < 1e-3] = 1.0
        parts += [h.reshape(n, -1), (rel / size).reshape(n, -1)]
    f = np.concatenate(parts + [pres], 1)
    return np.concatenate([f, np.diff(f, axis=0, prepend=f[:1])], 1).astype(np.float32)


def sign_segment(raw, pad=3):
    """Trim idle frames: keep from first to last frame where a hand is detected."""
    hand = np.abs(raw[:, 132:]).sum(1) > 0
    if hand.sum() < 5:
        return None
    a, b = np.where(hand)[0][[0, -1]]
    return raw[max(a - pad, 0): b + pad + 1]


def predict_probs(model, raw_keypoints):
    """Predict class probabilities from raw keypoints array (frames, 258)."""
    clip = sign_segment(raw_keypoints)
    if clip is None:
        return None
    features = build(resample(normalize(clip), 32))
    return model.predict(features[None], verbose=0)[0]
