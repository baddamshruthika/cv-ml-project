"""
app.py — Streamlit CV Prediction App
Uses TensorFlow ANN + RF + XGB + Manual Stacking Meta-Model
Run: streamlit run app.py
"""

import os


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "CV Prediction System",
    page_icon  = "🔬",
    layout     = "wide"
)

# ============================================================
# LOAD ALL MODELS
# ============================================================
@st.cache_resource
def load_models():
    required_files = {
        "ANN"             : "models/ann_model.keras",
        "Random Forest"   : "models/rf_model.pkl",
        "XGBoost"         : "models/xgb_model.pkl",
        "Meta Model"      : "models/meta_model.pkl",
        "Scaler"          : "models/scaler.pkl",
        "Feature Columns" : "models/feature_columns.pkl",
    }

    missing = [name for name, path in required_files.items()
               if not os.path.exists(path)]
    if missing:
        st.error(f"Missing model files: {missing}\n\nRun: python train_and_save.py first.")
        return None, None, None, None, None, None

    ann    = tf.keras.models.load_model("models/ann_model.keras")
    rf     = joblib.load("models/rf_model.pkl")
    xgb    = joblib.load("models/xgb_model.pkl")
    meta   = joblib.load("models/meta_model.pkl")   # RidgeCV
    scaler = joblib.load("models/scaler.pkl")
    feats  = joblib.load("models/feature_columns.pkl")

    return ann, rf, xgb, meta, scaler, feats

ann_model, rf_model, xgb_model, meta_model, scaler, FEATURE_COLS = load_models()

# ============================================================
# PREDICTION HELPER
# Handles scaling + manual stacking correctly
# ============================================================
def predict_current(inp_array, model_choice):
    """
    inp_array: shape (1, 6) — raw unscaled features
    Returns: single float prediction
    """
    inp_scaled = scaler.transform(inp_array)

    if model_choice == "ANN":
        return ann_model.predict(inp_scaled, verbose=0).flatten()[0]

    elif model_choice == "Random Forest":
        return rf_model.predict(inp_array)[0]

    elif model_choice == "XGBoost":
        return xgb_model.predict(inp_array)[0]

    elif model_choice == "Meta (Stacking)":
        # Manual stacking: collect base predictions then feed to RidgeCV
        ann_p  = ann_model.predict(inp_scaled, verbose=0).flatten()
        rf_p   = rf_model.predict(inp_array)
        xgb_p  = xgb_model.predict(inp_array)
        X_meta = np.column_stack([ann_p, rf_p, xgb_p])
        return meta_model.predict(X_meta)[0]


def predict_batch(X_raw, model_choice):
    """Batch prediction for CSV upload."""
    X_scaled = scaler.transform(X_raw)

    if model_choice == "ANN":
        return ann_model.predict(X_scaled, verbose=0).flatten()

    elif model_choice == "Random Forest":
        return rf_model.predict(X_raw)

    elif model_choice == "XGBoost":
        return xgb_model.predict(X_raw)

    elif model_choice == "Meta (Stacking)":
        ann_p  = ann_model.predict(X_scaled, verbose=0).flatten()
        rf_p   = rf_model.predict(X_raw)
        xgb_p  = xgb_model.predict(X_raw)
        X_meta = np.column_stack([ann_p, rf_p, xgb_p])
        return meta_model.predict(X_meta)

# ============================================================
# HEADER
# ============================================================
st.title("🔬 CV Prediction System (ML-Based)")
st.markdown(
    "Predict Current from Cyclic Voltammetry Parameters — "
    "*Ravichandran et al., ACS Omega 2024*"
)

if ann_model is None:
    st.stop()

# ============================================================
# SIDEBAR — INPUT PARAMETERS
# ============================================================
st.sidebar.header("⚙️ Input Parameters")

material = st.sidebar.selectbox(
    "Material",
    ["BFO", "BFZO", "BFCO"],
    help="BFO = Bismuth Ferrite | BFZO = Zinc substituted | BFCO = Cobalt substituted"
)

potential = st.sidebar.slider(
    "Potential (V)",
    min_value = -1.0,
    max_value =  1.0,
    value     =  0.0,
    step      =  0.001,
    format    = "%.3f"
)

scan_rate = st.sidebar.selectbox(
    "Scan Rate (mV/s)",
    [10, 20, 30, 40, 50, 60]
)

oxidation_label = st.sidebar.selectbox(
    "Reaction Type",
    ["Oxidation (1)", "Reduction (0)"]
)
oxidation_val = 1 if "Oxidation" in oxidation_label else 0

zn_co_conc = st.sidebar.number_input(
    "Zn/Co Concentration (mmol)",
    min_value = 0.0,
    max_value = 5.0,
    value     = 1.5,
    step      = 0.5
)

model_choice = st.sidebar.selectbox(
    "Model",
    ["Meta (Stacking)", "ANN", "Random Forest", "XGBoost"]
)

# Encode material
zn = 1 if material == "BFZO" else 0
co = 1 if material == "BFCO" else 0

# Build input array — order must match PREDICTORS exactly
# ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
def make_input():
    return np.array([[potential, oxidation_val, zn_co_conc, scan_rate, zn, co]])

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Single Prediction",
    "📈 CV Curve",
    "🔁 Compare All Models",
    "📂 Batch CSV"
])

# ── TAB 1: SINGLE PREDICTION ─────────────────────────────────
with tab1:
    st.subheader("⚡ Single Point Prediction")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Input:**")
        input_df = pd.DataFrame({
            "Parameter" : ["Material", "Potential (V)", "Scan Rate (mV/s)",
                           "Reaction", "Zn/Co Conc (mmol)", "Zn flag", "Co flag"],
            "Value"     : [material, f"{potential:.3f}", scan_rate,
                           oxidation_label, zn_co_conc, zn, co]
        })
        st.dataframe(input_df, hide_index=True)

    with col2:
        if st.button("🔮 Predict Current", key="single_pred"):
            inp = make_input()
            pred = predict_current(inp, model_choice)
            st.success(f"### Predicted Current")
            st.metric(
                label = f"{model_choice}",
                value = f"{pred:.8f} A"
            )

# ── TAB 2: CV CURVE ──────────────────────────────────────────
with tab2:
    st.subheader("📈 Full CV Curve Prediction")
    st.write("Generates predicted current across full potential range −1.0 to 1.0 V")

    if st.button("🎨 Generate CV Curve", key="cv_curve"):
        potentials = np.linspace(-1.0, 1.0, 300)

        ox_currents  = []
        red_currents = []

        progress = st.progress(0, text="Generating CV curve...")
        total = len(potentials)

        for i, p in enumerate(potentials):
            inp_ox  = np.array([[p, 1, zn_co_conc, scan_rate, zn, co]])
            inp_red = np.array([[p, 0, zn_co_conc, scan_rate, zn, co]])
            ox_currents.append(predict_current(inp_ox,  model_choice))
            red_currents.append(predict_current(inp_red, model_choice))
            progress.progress((i + 1) / total,
                              text=f"Generating... {i+1}/{total}")

        progress.empty()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(potentials, ox_currents,  'b-',  lw=2, label='Oxidation')
        ax.plot(potentials, red_currents, 'r--', lw=2, label='Reduction')
        ax.set_xlabel("Potential (V)", fontsize=12)
        ax.set_ylabel("Predicted Current (A)", fontsize=12)
        ax.set_title(
            f"Predicted CV Curve — {material} | {scan_rate} mV/s | {model_choice}",
            fontsize=13
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Specific capacitance
        v_window = 2.0   # -1 to 1 = 2V window
        area = np.trapz(np.abs(ox_currents), potentials)
        mass = zn_co_conc if zn_co_conc > 0 else 1.0
        csp  = area / (2 * (scan_rate / 1000) * mass * v_window)
        st.info(f"**Estimated Specific Capacitance:** `{csp:.5f} F/g`")

        # Download CV data
        cv_df = pd.DataFrame({
            "Potential"           : potentials,
            "Predicted_Oxidation" : ox_currents,
            "Predicted_Reduction" : red_currents
        })
        csv_cv = cv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CV Data",
            csv_cv, "cv_curve.csv", "text/csv"
        )

# ── TAB 3: COMPARE ALL MODELS ────────────────────────────────
with tab3:
    st.subheader("🔁 Compare All Models — Same Input")

    if st.button("📊 Compare Models", key="compare"):
        inp = make_input()
        inp_scaled = scaler.transform(inp)

        pred_ann  = ann_model.predict(inp_scaled, verbose=0).flatten()[0]
        pred_rf   = rf_model.predict(inp)[0]
        pred_xgb  = xgb_model.predict(inp)[0]

        # Manual stacking meta
        X_meta    = np.column_stack([[pred_ann], [pred_rf], [pred_xgb]])
        pred_meta = meta_model.predict(X_meta)[0]

        results = pd.DataFrame({
            "Model"             : ["ANN", "Random Forest", "XGBoost", "Meta (Stacking)"],
            "Predicted Current" : [pred_ann, pred_rf, pred_xgb, pred_meta],
        })

        st.dataframe(
            results.style.format({"Predicted Current": "{:.8f}"}),
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
        bars = ax.bar(results["Model"], results["Predicted Current"], color=colors)
        ax.set_ylabel("Predicted Current (A)")
        ax.set_title(
            f"Model Comparison — {material} | Potential={potential:.3f}V | SR={scan_rate}"
        )
        for bar, val in zip(bars, results["Predicted Current"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.6f}",
                ha='center', va='bottom', fontsize=8
            )
        plt.xticks(rotation=10)
        plt.tight_layout()
        st.pyplot(fig)

        # Performance metrics from paper
        st.write("**Paper Reported Accuracies:**")
        paper_df = pd.DataFrame({
            "Model"    : ["ANN", "RF", "XGB", "Meta"],
            "Test R²"  : [0.97725, 0.97541, 0.97714, 0.97737],
            "Test RMSE": [0.001111, 0.001155, 0.001114, 0.001108],
            "Accuracy" : ["97.73%", "97.54%", "97.71%", "97.74%"]
        })
        st.dataframe(paper_df, hide_index=True)

# ── TAB 4: BATCH CSV ─────────────────────────────────────────
with tab4:
    st.subheader("📂 Batch Prediction from CSV")

    st.write("**Your CSV must have these exact column names:**")
    st.code("Potential, OXIDATION, Zn/Co_Conc, SCAN_RATE, ZN, CO")

    st.write("**Example CSV format:**")
    example = pd.DataFrame({
        "Potential" : [0.3,  -0.3, 0.5],
        "OXIDATION" : [1,    0,    1  ],
        "Zn/Co_Conc": [1.5,  2.5,  3.5],
        "SCAN_RATE" : [10,   20,   30 ],
        "ZN"        : [1,    0,    0  ],
        "CO"        : [0,    1,    0  ],
    })
    st.dataframe(example, hide_index=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Uploaded:** {len(df_up)} rows")
        st.dataframe(df_up.head(10))

        if FEATURE_COLS:
            missing_cols = [c for c in FEATURE_COLS if c not in df_up.columns]
        else:
            missing_cols = [c for c in
                ["Potential","OXIDATION","Zn/Co_Conc","SCAN_RATE","ZN","CO"]
                if c not in df_up.columns]

        if missing_cols:
            st.error(f"Missing columns in your CSV: {missing_cols}")
        else:
            feat_cols = FEATURE_COLS or \
                ["Potential","OXIDATION","Zn/Co_Conc","SCAN_RATE","ZN","CO"]
            X_batch = df_up[feat_cols].values

            preds = predict_batch(X_batch, model_choice)
            df_up["Predicted_Current"] = preds

            st.write("**Results (first 10 rows):**")
            st.dataframe(df_up.head(10))

            csv_out = df_up.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Results CSV",
                csv_out,
                "cv_predictions.csv",
                "text/csv"
            )
