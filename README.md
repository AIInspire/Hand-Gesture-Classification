
# Hand Gesture Classification

## Project Overview

This project focuses on classifying hand gestures using machine learning models trained on normalized hand landmark data. The goal is to accurately predict hand gestures from landmark features extracted using MediaPipe.

## Dataset

- Training and testing data are stored in the `Data/` directory as CSV files: `train_df.csv` and `test_df.csv`.
- Each dataset contains extracted hand landmarks and corresponding gesture labels.

## Models Trained

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

All models were trained and evaluated on the hand landmark features, and results were logged using **MLflow** for experiment tracking.

## Results

| Model               | Accuracy  |
|---------------------|-----------|
| Logistic Regression  | 0.7933    |
| Random Forest       | 0.9405    |
| **XGBoost**         | **0.9727**|

**Best Model: XGBoost**

The XGBoost classifier achieved the highest accuracy of 97.27% on the test set, outperforming Logistic Regression and Random Forest models. Due to its superior accuracy and robustness, XGBoost was selected as the best-performing model.

## Usage Instructions

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run training and logging script:**

   ```bash
   python train.py
   ```

3. **Launch MLflow UI to view experiments and runs:**

   From the project root directory:

   ```bash
   mlflow ui --backend-store-uri file:///$(pwd)/.mlruns
   ```

   - Open your browser at [http://localhost:5000](http://localhost:5000) to inspect experiment runs, metrics, and artifacts.

## Notes

- The training script logs model parameters, metrics, and artifacts including the trained model and label encoder to MLflow.
- To reproduce experiments, ensure your dataset is correctly placed in the `Data/` folder.
- The project uses normalized hand landmarks for better model performance.

---

Feel free to explore the runs in MLflow UI to review model details and metrics.

---

*Created by [Your Name] | Date: 2025-05-31*
