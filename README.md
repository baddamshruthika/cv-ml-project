# CV Prediction System (ML-Based)

Predicts current from Cyclic Voltammetry parameters for
supercapacitor materials using Machine Learning.

## Based on
Ravichandran et al., ACS Omega 2024, 9, 33459–33470

## Materials
- BFO (Bismuth Ferrite)
- BFZO (Zinc-substituted BFO)
- BFCO (Cobalt-substituted BFO)

## Models Used
- ANN (TensorFlow/Keras) — 6→100→80→1
- Random Forest (scikit-learn)
- XGBoost
- Meta-Model (Manual Stacking + RidgeCV)

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add your dataset
Place FINAL_CV_DATASET.xlsx inside the data/ folder

### 3. Train models
python train_and_save.py

### 4. Launch app
streamlit run app.py

## Project Structure
cv-ml-project/
├── data/               ← place Excel file here (not uploaded)
├── models/             ← generated after training (not uploaded)
├── app.py              ← Streamlit web app
├── train_and_save.py   ← training script
├── requirements.txt    ← dependencies
└── README.md

## Results
| Model | Test R²  | Accuracy |
|-------|----------|----------|
| ANN   | ~0.977   | ~97.7%   |
| RF    | ~0.975   | ~97.5%   |
| XGB   | ~0.977   | ~97.7%   |
| Meta  | ~0.977   | ~97.7%   |
