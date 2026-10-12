# 🤟 Indian Sign Language (ISL) Sentence Recognizer

An end-to-end Computer Vision & Deep Learning web application deployed with **Streamlit** for continuous Indian Sign Language (ISL) sentence recognition. Powered by **MediaPipe 3D Landmark Vision Tasks** and a **Bidirectional GRU neural network**.

---

## 🌟 Key Features

- **Real-Time Video Gesture Analysis**: Upload sign language videos (`.mp4`, `.mov`, `.avi`, `.webm`) or test instantly with bundled sample clips.
- **3D Landmark Feature Extraction**: Tracks 33 upper-body pose joints and 42 dual-hand landmarks per frame using MediaPipe Tasks.
- **Invariant Geometric Normalization**: Body-centric mid-shoulder centering and shoulder-width normalization, invariant to distance from camera.
- **Bidirectional GRU Temporal Classifier**: 64 hidden units with Spatial Dropout and Dual Pooling (Global Average + Global Max Pooling) classifying 101 continuous sentences.
- **Constrained Vocabulary Filtering**: Interactive presets (Greetings, Help/Emergency, Custom Selection) to maximize inference accuracy on closed domains.
- **Visual Telemetry & Skeleton Overlay**: Real-time display of hand detection rates, active signing windows, confidence meters, and landmark tracking skeletons.

---

## 🚀 Quick Start (Local Run)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud (Free)

Deploying this app to the public web takes less than 2 minutes:

1. **Push to GitHub**:
   Ensure all changes are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Deploy ISL Sentence Recognizer Streamlit App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
   - Click **"New app"** (or **"Create app"**).

3. **Configure Deployment Settings**:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Choose your custom subdomain, e.g. `isl-recognizer.streamlit.app`)

4. **Click Deploy!**:
   - Streamlit Cloud automatically reads `packages.txt` (installs `ffmpeg`, `libgl1`) and `requirements.txt` (installs `tensorflow-cpu`, `mediapipe`, `streamlit`).
   - The app automatically initializes the weights from `isl_gru_v2.keras` and downloads the MediaPipe vision models on first boot!

---

## 🏗️ Architecture & Pipeline

```text
[Input Sign Video]
       │
       ▼
[MediaPipe Pose & Hand Landmarkers (stride=2, 640px)]
       │
       ▼ (258 features/frame: Pose 33x4, Left Hand 21x3, Right Hand 21x3)
[Active Gesture Segmentation (idle frame trimming)]
       │
       ▼
[Mid-Shoulder Normalization + 32-Frame Resampling]
       │
       ▼ (32 x 368 temporal representation with velocities)
[Bi-Directional GRU + Global Pooling]
       │
       ▼
[Top-5 Sentence Softmax Probabilities]
```

---

## 📊 Dataset & Model Details

- **Dataset:** ISL-CSLRT (Indian Sign Language Continuous Sentence Level Corpus)
- **Classes:** 101 continuous Indian Sign Language sentences
- **Model File:** `isl_gru_v2.keras` (~975 KB)
- **Vocabulary File:** `classes.json` (101 unique gloss sentences)

---

## 📝 License
Built for educational, accessibility, and research demonstrations.
