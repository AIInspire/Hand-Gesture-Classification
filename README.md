
# 👋 Hand Gesture Classification using MediaPipe & HaGRID Dataset

## 🔍 Overview
This project focuses on **classifying hand gestures** using hand landmark data extracted via **MediaPipe** from the **HaGRID (Hand Gesture Recognition Image Dataset)**.

Using a variety of machine learning models trained on x, y, z coordinates of hand keypoints, we build a real-time or recorded-video-based hand gesture recognizer capable of predicting **18 predefined gestures** with high accuracy.

---

## 📁 Project Structure

```
.
├── model.pkl                  # Trained ML model (best one after tuning)
├── label_encoder.pkl          # LabelEncoder for inverse transforming predicted classes
├── hand_landmarks.csv         # HaGRID-based landmark dataset (x, y, z coordinates + labels)
├── gesture_video.mp4          # Input video for inference
├── output.mp4                 # Output video with predicted gestures
├── gesture_classification.ipynb  # Google Colab notebook (main workflow)
└── README.md
```

---

## 🎯 Project Objectives

- Preprocess and normalize hand landmark data.
- Train and compare multiple ML classifiers for gesture recognition.
- Tune hyperparameters for best model performance.
- Apply the best-performing model on recorded video using MediaPipe.
- Overlay predicted gestures on video frames using OpenCV.
- Export the final annotated video.

---

## 🧾 Dataset Details

- **Source:** [HaGRID Dataset]
- **Gestures:** 18 types such as `like`, `dislike`, `no`, `stop`, `peace`, `call`, etc.
- **Format:** CSV file with:
  - 21 landmarks per hand × (x, y, z) = 63 features
  - 1 label column for gesture class

---

## ⚙️ Technologies Used

- 🧠 **Machine Learning:** `scikit-learn`, `xgboost`
- 💻 **Preprocessing:** `NumPy`, `Pandas`
- 📹 **MediaPipe** for real-time landmark extraction
- 🖼 **OpenCV** for image processing & video overlay
- 🧪 **Evaluation:** Accuracy, Precision, Recall, F1-score
- 📓 **Platform:** Google Colab

---

## 🧠 Machine Learning Models & Results

### 📉 Before Hyperparameter Tuning:

| Model                   | Accuracy  | Precision | Recall   | F1-score |
|------------------------|-----------|-----------|----------|----------|
| Logistic Regression     | 0.7996    | 0.7985    | 0.7996   | 0.7978   |
| Decision Tree           | 0.8536    | 0.8558    | 0.8536   | 0.8542   |
| Random Forest           | 0.9387    | 0.9397    | 0.9387   | 0.9388   |
| Support Vector Machine  | 0.8359    | 0.8571    | 0.8359   | 0.8386   |
| K-Nearest Neighbors     | 0.9081    | 0.9097    | 0.9081   | 0.9084   |
| XGBoost Classifier      | 0.9699    | 0.9704    | 0.9699   | 0.9699   |

### 🚀 After Hyperparameter Tuning:

| Model                   | Accuracy  | Precision | Recall   | F1-score |
|------------------------|-----------|-----------|----------|----------|
| Logistic Regression     | 0.9307    | 0.9307    | 0.9307   | 0.9304   |
| Decision Tree           | 0.8681    | 0.8690    | 0.8681   | 0.8681   |
| Random Forest           | 0.9527    | 0.9534    | 0.9527   | 0.9529   |
| Support Vector Machine  | **0.9808**| **0.9810**| **0.9808**| **0.9808** |
| K-Nearest Neighbors     | 0.9325    | 0.9331    | 0.9325   | 0.9327   |
| XGBoost Classifier      | 0.9782    | 0.9784    | 0.9782   | 0.9782   |

✅ **Best Model:** Support Vector Machine (SVC) after tuning

---

## 🔁 Normalization & Preprocessing

- Re-centered all (x, y) coordinates using the wrist point as origin
- Scaled coordinates using the distance to the mid-finger tip
- Z-coordinates used directly (already normalized)
- Applied smoothing using **mode of predictions over a rolling window**

---

## 🎥 Video Demo

> A recorded video is used as input and processed frame-by-frame:
> - MediaPipe extracts hand landmarks
> - The best model predicts the gesture
> - OpenCV overlays the prediction
> - Final video is saved with gesture labels

🎬 **Output Video:** [`output.mp4`](#)

📎 **Demo Sample:** [https://drive.google.com/file/d/1q6TI4d9Br_06sElh-FZQEbUmZkOWUFOo/view?usp=sharing](#) 

---

## 📌 How to Use

1. Clone this repo or upload files to Google Colab.
2. Upload:
   - `model.pkl` and `label_encoder.pkl`
   - Your input video (e.g., `gesture_video.mp4`)
3. Run the notebook `gesture_classification.ipynb`.
4. View and download `output.mp4` — the video with gesture predictions.

---

## 🙋‍♀️ Author

- **Omnia Abdulnabi**
