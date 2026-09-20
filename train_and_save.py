"""
train_and_save.py
Complete training script using TensorFlow for ANN
Run: python train_and_save.py
"""

import os


import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# CONFIGURATION
# ============================================================
EXCEL_FILE = "data/CV_DATASET (2).xlsx"
PREDICTORS = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
TARGET     = "Current"

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading data...")
print("=" * 55)

if not os.path.exists(EXCEL_FILE):
    print(f"ERROR: File not found: {EXCEL_FILE}")
    print("Place FINAL_CV_DATASET.xlsx inside the data/ folder")
    exit()

data = pd.read_excel(EXCEL_FILE)
print(f"Loaded: {data.shape[0]:,} rows x {data.shape[1]} columns")
print(f"Columns: {list(data.columns)}")

missing = [c for c in PREDICTORS + [TARGET] if c not in data.columns]
if missing:
    print(f"ERROR - Missing columns: {missing}")
    exit()

# ============================================================
# STEP 2: PREPROCESSING
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Preprocessing...")
print("=" * 55)

data = data[PREDICTORS + [TARGET]].dropna().reset_index(drop=True)
print(f"Rows after NaN removal: {len(data):,}")

train_data, test_data, train_target, test_target = train_test_split(
    data[PREDICTORS], data[TARGET],
    test_size=0.2, random_state=123
)

train_target = np.array(train_target)
test_target  = np.array(test_target)

print(f"Train: {len(train_data):,} | Test: {len(test_data):,}")

# StandardScaler for ANN
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled  = scaler.transform(test_data)
print("Scaling complete")

os.makedirs("models", exist_ok=True)

# ============================================================
# STEP 3: TRAIN RANDOM FOREST
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Training Random Forest...")
print("=" * 55)

rf_model = RandomForestRegressor(
    n_estimators = 100,
    max_depth    = 11,
    random_state = 123,
    n_jobs       = 1       # n_jobs=1 avoids Windows parallel issues
)
rf_model.fit(train_data, train_target)

rf_pred_test  = rf_model.predict(test_data)
rf_pred_train = rf_model.predict(train_data)

rf_rmse  = np.sqrt(mean_squared_error(test_target,  rf_pred_test))
rf_mse   = mean_squared_error(test_target,  rf_pred_test)
rf_r2    = r2_score(test_target,  rf_pred_test)
rf_rmset = np.sqrt(mean_squared_error(train_target, rf_pred_train))
rf_r2t   = r2_score(train_target, rf_pred_train)

print(f"RF Test  -> RMSE: {rf_rmse:.6f} | R2: {rf_r2:.6f} | Acc: {rf_r2*100:.2f}%")
print(f"RF Train -> RMSE: {rf_rmset:.6f} | R2: {rf_r2t:.6f}")

feat_imp = pd.Series(rf_model.feature_importances_, index=PREDICTORS)
print("\nFeature Importance:")
for feat, imp in feat_imp.sort_values(ascending=False).items():
    print(f"  {feat:15s}: {imp*100:.1f}%")

joblib.dump(rf_model, "models/rf_model.pkl")
print("Saved: models/rf_model.pkl")

# ============================================================
# STEP 4: TRAIN XGBOOST
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Training XGBoost...")
print("=" * 55)

xgb_model = XGBRegressor(
    objective    = 'reg:squarederror',
    eta          = 0.1,
    max_depth    = 6,
    n_estimators = 200,
    random_state = 123,
    n_jobs       = 1,
    verbosity    = 0
)
xgb_model.fit(train_data, train_target)

xgb_pred_test  = xgb_model.predict(test_data)
xgb_pred_train = xgb_model.predict(train_data)

xgb_rmse  = np.sqrt(mean_squared_error(test_target,  xgb_pred_test))
xgb_mse   = mean_squared_error(test_target,  xgb_pred_test)
xgb_r2    = r2_score(test_target,  xgb_pred_test)
xgb_rmset = np.sqrt(mean_squared_error(train_target, xgb_pred_train))
xgb_r2t   = r2_score(train_target, xgb_pred_train)

print(f"XGB Test  -> RMSE: {xgb_rmse:.6f} | R2: {xgb_r2:.6f} | Acc: {xgb_r2*100:.2f}%")
print(f"XGB Train -> RMSE: {xgb_rmset:.6f} | R2: {xgb_r2t:.6f}")

cv_kf = KFold(n_splits=10, shuffle=True, random_state=123)
cv_scores_xgb = cross_val_score(
    xgb_model, data[PREDICTORS], data[TARGET],
    scoring='r2', cv=cv_kf, n_jobs=1
)
print("\n10-Fold CV R2 Scores:")
for i, score in enumerate(cv_scores_xgb, 1):
    print(f"  Fold {i:2d}: {score:.6f}")
print(f"Average CV R2: {cv_scores_xgb.mean():.6f}")

joblib.dump(xgb_model, "models/xgb_model.pkl")
print("Saved: models/xgb_model.pkl")

# ============================================================
# STEP 5: TRAIN ANN (TensorFlow)
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Training ANN (TensorFlow)...")
print("=" * 55)

def build_ann():
    """
    Architecture from paper Table 2:
    Input(6) -> Dense(100, ReLU) -> Dense(80, ReLU) -> Dense(1, Linear)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu',
                              input_shape=(len(PREDICTORS),)),
        tf.keras.layers.Dense(80,  activation='relu'),
        tf.keras.layers.Dense(1,   activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    return model

ann_model = build_ann()
ann_model.summary()

# Train — no validation_split to avoid TF multiprocessing issues
history = ann_model.fit(
    train_data_scaled, train_target,
    epochs          = 100,
    batch_size      = 32,
    validation_data = (test_data_scaled, test_target),
    verbose         = 1
)

ann_pred_test  = ann_model.predict(test_data_scaled).flatten()
ann_pred_train = ann_model.predict(train_data_scaled).flatten()

ann_rmse  = np.sqrt(mean_squared_error(test_target,  ann_pred_test))
ann_mse   = mean_squared_error(test_target,  ann_pred_test)
ann_r2    = r2_score(test_target,  ann_pred_test)
ann_rmset = np.sqrt(mean_squared_error(train_target, ann_pred_train))
ann_r2t   = r2_score(train_target, ann_pred_train)

print(f"\nANN Test  -> RMSE: {ann_rmse:.6f} | R2: {ann_r2:.6f} | Acc: {ann_r2*100:.2f}%")
print(f"ANN Train -> RMSE: {ann_rmset:.6f} | R2: {ann_r2t:.6f}")

# 4-fold CV for ANN
print("\nANN 4-Fold Cross-Validation:")
cv_scores_ann = []
kf4 = KFold(n_splits=4, shuffle=True, random_state=123)
for fold, (tr_idx, val_idx) in enumerate(kf4.split(train_data_scaled), 1):
    xtr  = train_data_scaled[tr_idx]
    xval = train_data_scaled[val_idx]
    ytr  = train_target[tr_idx]
    yval = train_target[val_idx]
    m = build_ann()
    m.fit(xtr, ytr, epochs=50, batch_size=32, verbose=0)
    score = r2_score(yval, m.predict(xval).flatten())
    cv_scores_ann.append(score)
    print(f"  Fold {fold}: R2 = {score:.6f}")
print(f"Mean CV R2: {np.mean(cv_scores_ann):.6f}")

# Save ANN in Keras format
ann_model.save("models/ann_model.keras")
print("Saved: models/ann_model.keras")

# ============================================================
# STEP 6: META-MODEL (MANUAL STACKING — no StackingRegressor)
# Avoids all Windows joblib + TensorFlow DLL crash issues
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Training Meta-Model (Manual Stacking)...")
print("=" * 55)

# 6a: Collect base model predictions
print("Getting base model predictions...")

ann_train_pred = ann_model.predict(train_data_scaled).flatten()
ann_test_pred  = ann_model.predict(test_data_scaled).flatten()

rf_train_pred  = rf_model.predict(train_data.values)
rf_test_pred   = rf_model.predict(test_data.values)

xgb_train_pred = xgb_model.predict(train_data.values)
xgb_test_pred  = xgb_model.predict(test_data.values)

print("  ANN predictions  : done")
print("  RF  predictions  : done")
print("  XGB predictions  : done")

# 6b: Stack into new feature matrix
X_meta_train = np.column_stack([ann_train_pred, rf_train_pred, xgb_train_pred])
X_meta_test  = np.column_stack([ann_test_pred,  rf_test_pred,  xgb_test_pred])

print(f"  Meta train shape : {X_meta_train.shape}")
print(f"  Meta test  shape : {X_meta_test.shape}")

# 6c: Train RidgeCV as final estimator (same as paper)
print("Training RidgeCV meta-learner...")
meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta_model.fit(X_meta_train, train_target)
print(f"Best alpha: {meta_model.alpha_}")

# 6d: Evaluate
stacking_pred_test  = meta_model.predict(X_meta_test)
stacking_pred_train = meta_model.predict(X_meta_train)

stacking_rmse  = np.sqrt(mean_squared_error(test_target,  stacking_pred_test))
stacking_mse   = mean_squared_error(test_target,  stacking_pred_test)
stacking_r2    = r2_score(test_target,  stacking_pred_test)
stacking_rmset = np.sqrt(mean_squared_error(train_target, stacking_pred_train))
stacking_r2t   = r2_score(train_target, stacking_pred_train)

print(f"\nMeta Test  -> RMSE: {stacking_rmse:.6f} | R2: {stacking_r2:.6f} | Acc: {stacking_r2*100:.2f}%")
print(f"Meta Train -> RMSE: {stacking_rmset:.6f} | R2: {stacking_r2t:.6f}")

joblib.dump(meta_model, "models/meta_model.pkl")
print("Saved: models/meta_model.pkl")

# ============================================================
# STEP 7: SAVE REMAINING ARTIFACTS
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Saving remaining artifacts...")
print("=" * 55)

joblib.dump(scaler,     "models/scaler.pkl")
print("Saved: models/scaler.pkl")

joblib.dump(PREDICTORS, "models/feature_columns.pkl")
print("Saved: models/feature_columns.pkl")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("ALL MODELS TRAINED AND SAVED")
print("=" * 55)
print(f"{'Model':<8} {'Test R2':>10} {'Test RMSE':>12} {'Accuracy':>10}")
print("-" * 45)
print(f"{'ANN':<8} {ann_r2:>10.6f} {ann_rmse:>12.6f} {ann_r2*100:>9.2f}%")
print(f"{'RF':<8} {rf_r2:>10.6f} {rf_rmse:>12.6f} {rf_r2*100:>9.2f}%")
print(f"{'XGB':<8} {xgb_r2:>10.6f} {xgb_rmse:>12.6f} {xgb_r2*100:>9.2f}%")
print(f"{'Meta':<8} {stacking_r2:>10.6f} {stacking_rmse:>12.6f} {stacking_r2*100:>9.2f}%")
print("=" * 55)
print("\nModels saved in models/ folder:")
print("  ann_model.keras  rf_model.pkl")
print("  xgb_model.pkl    meta_model.pkl")
print("  scaler.pkl       feature_columns.pkl")
print("\nNext step -> run:  streamlit run app.py")