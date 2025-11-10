# ==========================================================
# 📊 CTR Prediction Dashboard (Fixed: excludes non-model pickles)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle, os, warnings
from types import SimpleNamespace

# ----------------------------------------------------------
# 🧹 Suppress warnings
# ----------------------------------------------------------
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 🧱 Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="CTR Prediction Dashboard", layout="wide")
st.title("📈 CTR Prediction Model Comparison Dashboard")
st.markdown("Compare, visualize, and interpret pre-trained CTR prediction models with ROC, PR curves, confusion matrices, and feature importance.")

# ----------------------------------------------------------
# 📦 Load Pickles: models vs other saved data
# ----------------------------------------------------------
model_dir = "saved_models"

def load_saved_pickles(path):
    """
    Loads .pkl files but separates 'models' (objects with predict_proba/predict)
    from other saved objects (dicts/arrays).
    Returns (models_dict, other_objects_dict)
    """
    models = {}
    other = {}
    if not os.path.exists(path):
        return models, other

    for fname in os.listdir(path):
        if not fname.endswith(".pkl"):
            continue
        full = os.path.join(path, fname)
        try:
            with open(full, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            # couldn't unpickle — skip
            continue

        key = fname.replace(".pkl", "")
        # Heuristic: treat as model if has predict_proba or predict method
        if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
            models[key] = obj
        else:
            other[key] = obj
    return models, other

models, other_pickles = load_saved_pickles(model_dir)

if not models:
    st.warning("❗ No valid model pickles found in 'saved_models'. Make sure your model .pkl files are present.")
else:
    st.sidebar.success("✅ Models Loaded Successfully!")
    st.sidebar.write("Models:")
    st.sidebar.write(list(models.keys()))

# ----------------------------------------------------------
# 📊 Load other saved data (CSV/npz) regardless
# ----------------------------------------------------------
perf_file = os.path.join(model_dir, "model_performance.csv")
roc_file = os.path.join(model_dir, "roc_data.npz")
pr_file  = os.path.join(model_dir, "pr_data.npz")
cm_file  = os.path.join(model_dir, "confusion_data.pkl")  # optional single file
feature_file = os.path.join(model_dir, "train_features.txt")

# If confusion_data.pkl exists as a standalone file (not inside other_pickles), load it:
confusion_pickle_data = None
if os.path.exists(cm_file):
    try:
        with open(cm_file, "rb") as f:
            confusion_pickle_data = pickle.load(f)
    except Exception:
        confusion_pickle_data = None

# Also check other_pickles for relevant saved-data objects (safe fallback)
if confusion_pickle_data is None and "confusion_data" in other_pickles:
    confusion_pickle_data = other_pickles["confusion_data"]

# ----------------------------------------------------------
# ⚡ Show precomputed performance table (from CSV)
# ----------------------------------------------------------
if os.path.exists(perf_file):
    st.subheader("⚡ Model Performance (K-Fold Cross Validation)")
    try:
        perf_df = pd.read_csv(perf_file)
        st.dataframe(perf_df.style.highlight_max(axis=0, color="lightgreen"))
    except Exception as e:
        st.warning(f"Could not read {perf_file}: {e}")
else:
    st.info("No precomputed model_performance.csv found in 'saved_models'.")

# ----------------------------------------------------------
# 📈 ROC–AUC Curves (from training)
# ----------------------------------------------------------
if os.path.exists(roc_file):
    try:
        st.subheader("📈 ROC–AUC Curve Comparison (From Training)")
        roc_data = np.load(roc_file, allow_pickle=True)
        plt.figure(figsize=(8, 6))
        for name in roc_data.files:
            d = roc_data[name].item()
            # d expected to be dict with 'fpr','tpr','auc'
            plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{name} (AUC={d['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.title("ROC–AUC Curve Comparison")
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not plot ROC curves: {e}")

# ----------------------------------------------------------
# 📉 Precision–Recall Curves (from training)
# ----------------------------------------------------------
if os.path.exists(pr_file):
    try:
        st.subheader("📉 Precision–Recall Curves (From Training)")
        pr_data = np.load(pr_file, allow_pickle=True)
        plt.figure(figsize=(8, 6))
        for name in pr_data.files:
            d = pr_data[name].item()
            # d expected to have 'precision','recall'
            plt.plot(d["recall"], d["precision"], lw=2, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve Comparison")
        plt.legend(loc="lower left")
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not plot PR curves: {e}")

# ----------------------------------------------------------
# 🧩 Confusion Matrices + Precision & Recall (from saved file)
# ----------------------------------------------------------
if confusion_pickle_data is not None:
    st.subheader("🧩 Confusion Matrices with Precision & Recall")
    cols = st.columns(2)
    idx = 0
    try:
        for name, d in confusion_pickle_data.items():
            cm = d.get("matrix")
            pr = d.get("precision")
            rc = d.get("recall")
            with cols[idx % 2]:
                st.markdown(f"### {name}")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.write(f"**Precision:** {pr:.3f} | **Recall:** {rc:.3f}")
            idx += 1
    except Exception as e:
        st.warning(f"Could not render confusion matrices: {e}")

# ----------------------------------------------------------
# 🏆 Feature Importance Viewer (Filtered)
# ----------------------------------------------------------
st.subheader("🏆 Feature Importance Viewer")

# Only include valid ML models (we already filtered when loading)
valid_model_names = list(models.keys())

# If no models, small message
if not valid_model_names:
    st.info("No ML models loaded to display feature importances.")
else:
    selected_model_name = st.selectbox("Select a model to view feature importance:", valid_model_names)

    # read training features if available
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            train_features = [line.strip() for line in f.readlines()]
    else:
        train_features = []

    if selected_model_name:
        model = models[selected_model_name]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # if train_features missing or length mismatch, create generic names
            if not train_features or len(train_features) != len(importances):
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            else:
                feature_names = train_features

            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance")
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title=f"Feature Importance – {selected_model_name}"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning(f"{selected_model_name} does not support feature importance visualization.")

# ----------------------------------------------------------
# 📂 Upload Dataset for Prediction
# ----------------------------------------------------------
st.sidebar.header("📥 Upload Dataset for Predictions")
uploaded_file = st.sidebar.file_uploader("Upload your preprocessed dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")

    # Load train features (if exists) to align columns
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            train_features = [line.strip() for line in f.readlines()]
    else:
        train_features = list(data.columns)

    # Clean dataset: drop unseen columns and add missing ones (fill 0)
    X = data.copy()
    dropped = []
    for col in list(X.columns):
        if col not in train_features:
            X.drop(columns=col, inplace=True)
            dropped.append(col)
    for col in train_features:
        if col not in X.columns:
            X[col] = 0
    X = X[train_features]

    if dropped:
        st.warning(f"Dropped unseen columns from uploaded data: {dropped}")

    # Encode categorical columns
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # ----------------------------------------------------------
    # 🎯 Interactive Single-Row Prediction (All valid models)
    # ----------------------------------------------------------
    st.subheader("🎯 Predict CTR on Custom Input")
    input_data = {}
    for col in train_features:
        if np.issubdtype(X[col].dtype, np.number):
            input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()), key=f"in_{col}")
        else:
            input_data[col] = st.selectbox(f"Select {col}", X[col].unique(), key=f"in_sel_{col}")

    if st.button("🔮 Predict CTR Using All Models"):
        input_df = pd.DataFrame([input_data])

        # encode categorical values same as training encoding (best-effort)
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        predictions = {}
        binary_outputs = {}
        for name, model in models.items():
            try:
                # use predict_proba if available, else predict (and convert to prob-like value)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_df)[0, 1]
                else:
                    # fallback: model.predict -> treat as deterministic prob (0 or 1)
                    pred = model.predict(input_df)[0]
                    prob = float(pred)
                predictions[name] = prob
                binary_outputs[name] = int(prob >= 0.5)
            except Exception as e:
                predictions[name] = None
                binary_outputs[name] = None
                st.warning(f"{name} could not predict: {e}")

        # Display model outputs
        pred_df = pd.DataFrame(predictions.items(), columns=["Model", "Predicted CTR Probability"])
        bin_df = pd.DataFrame(binary_outputs.items(), columns=["Model", "Predicted Class (0/1)"])

        st.write("### 🔍 Model Probabilities")
        st.dataframe(pred_df.style.highlight_max(axis=0, color="lightblue"))

        st.write("### 🧮 Predicted Classes")
        st.dataframe(bin_df.style.highlight_max(axis=0, color="lightgreen"))

        # Combined output (average of available probs)
        valid_probs = [v for v in predictions.values() if v is not None]
        if valid_probs:
            combined_pred = np.mean(valid_probs)
            final_class = int(combined_pred >= 0.5)
            st.success(f"✅ **Combined CTR Probability:** {combined_pred:.3f}")
            st.info(f"🧩 **Final Predicted Class:** {final_class}")

else:
    st.info("👈 Upload a dataset to make predictions.")
