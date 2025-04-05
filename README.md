# 🤖 Hand Gesture Classification Using MediaPipe & HaGRID Dataset

This project focuses on classifying 18 different hand gestures using landmark coordinates extracted via **MediaPipe** from the **HaGRID dataset**. A trained machine learning model then classifies the gestures in **real-time using a webcam**.

---

## 📁 Project Structure

```plaintext
Hand-Gesture-Classification/
│
├── Dataset/
│   └── haGRID_landmarks.csv          # Preprocessed landmark data from MediaPipe
│
├── best_model.pkl                    # Final trained model
├── label_encoder.pkl                 # Label encoder to map class numbers to labels
│
├── Data_Preparation.ipynb           # Data cleaning, preprocessing, and normalization
├── Training_and_Evaluation.ipynb    # Training and tuning of multiple ML models
├── Implementation_using_Mediapipe.ipynb   # Real-time implementation using MediaPipe & webcam
│
├── README.md                         # Project documentation (you’re here!)

