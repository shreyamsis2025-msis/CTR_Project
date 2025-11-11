# ==========================================================
# 📊 CTR Prediction Dashboard (Final Decoded Dropdown Version)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle, os, warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 🧱 Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="CTR Prediction Dashboard", layout="wide")
st.title("📈 CTR Prediction Model Comparison Dashboard")
st.markdown("Visualize model performance and make CTR predictions with pre-trained models.")

# ----------------------------------------------------------
# 📦 Load Models
# ----------------------------------------------------------
model_dir = "saved_models"

def load_models():
    models = {}
    if not os.path.exists(model_dir):
        st.error("❌ 'saved_models' directory not found. Please run ctr_prediction.ipynb first.")
        return models
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl") and "confusion" not in filename.lower() and "data" not in filename.lower() and "encoder" not in filename.lower():
            model_name = filename.replace(".pkl", "")
            with open(os.path.join(model_dir, filename), "rb") as file:
                models[model_name] = pickle.load(file)
    return models

models = load_models()
if not models:
    st.stop()
else:
    st.sidebar.success("✅ Models Loaded Successfully!")
    st.sidebar.write(list(models.keys()))

# ----------------------------------------------------------
# 📊 Load Supporting Files
# ----------------------------------------------------------
perf_file = os.path.join(model_dir, "model_performance.csv")
roc_file = os.path.join(model_dir, "roc_data.npz")
pr_file = os.path.join(model_dir, "pr_data.npz")
cm_file = os.path.join(model_dir, "confusion_data.pkl")
feature_file = os.path.join(model_dir, "train_features.txt")
encoder_file = os.path.join(model_dir, "encoders.pkl")

# Load label encoders
label_encoders = {}
if os.path.exists(encoder_file):
    with open(encoder_file, "rb") as f:
        label_encoders = pickle.load(f)
    # st.sidebar.info("")

# ----------------------------------------------------------
# ⚡ Model Performance
# ----------------------------------------------------------
if os.path.exists(perf_file):
    st.subheader("⚡ Model Performance (K-Fold Cross Validation)")
    perf_df = pd.read_csv(perf_file)
    st.dataframe(perf_df.style.highlight_max(axis=0, color="lightgreen"))

# ----------------------------------------------------------
# 📈 ROC–AUC Curves
# ----------------------------------------------------------
if os.path.exists(roc_file):
    st.subheader("📈 ROC–AUC Curve Comparison (From Training)")
    roc_data = np.load(roc_file, allow_pickle=True)
    plt.figure(figsize=(8, 6))
    for name in roc_data.files:
        d = roc_data[name].item()
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{name} (AUC={d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC–AUC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)

# ----------------------------------------------------------
# 📉 Precision–Recall Curves
# ----------------------------------------------------------
if os.path.exists(pr_file):
    st.subheader("📉 Precision–Recall Curves (From Training)")
    pr_data = np.load(pr_file, allow_pickle=True)
    plt.figure(figsize=(8, 6))
    for name in pr_data.files:
        d = pr_data[name].item()
        plt.plot(d["recall"], d["precision"], lw=2, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.grid(True)
    st.pyplot(plt)

# ----------------------------------------------------------
# 🧩 Confusion Matrices + Precision & Recall
# ----------------------------------------------------------
if os.path.exists(cm_file):
    st.subheader("🧩 Confusion Matrices with Precision & Recall")
    with open(cm_file, "rb") as f:
        cm_data = pickle.load(f)
    cols = st.columns(2)
    idx = 0
    for name, d in cm_data.items():
        cm = d["matrix"]
        pr, rc = d["precision"], d["recall"]
        with cols[idx % 2]:
            st.markdown(f"### {name}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            st.write(f"**Precision:** {pr:.3f} | **Recall:** {rc:.3f}")
        idx += 1

# ----------------------------------------------------------
# 🏆 Feature Importance Viewer (Filtered)
# ----------------------------------------------------------
st.subheader("🏆 Feature Importance Viewer")
valid_model_names = [name for name in models.keys() if "data" not in name.lower() and "confusion" not in name.lower()]
selected_model_name = st.selectbox("Select a model to view feature importance:", valid_model_names)

if os.path.exists(feature_file):
    with open(feature_file, "r") as f:
        train_features = [line.strip() for line in f.readlines()]
else:
    train_features = []

if selected_model_name:
    model = models[selected_model_name]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": train_features, "Importance": importances}).sort_values(by="Importance")
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

    # Load training features
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            train_features = [line.strip() for line in f.readlines()]
    else:
        train_features = list(data.columns)

    X = data.copy()
    for col in X.columns:
        if col not in train_features:
            X.drop(columns=col, inplace=True)
            st.warning(f"Dropped unseen column: {col}")
    for col in train_features:
        if col not in X.columns:
            X[col] = 0
    X = X[train_features]

    # ----------------------------------------------------------
    # 🎯 CTR Prediction (Decode Encoded Variables)
    # ----------------------------------------------------------
    st.subheader("🎯 Predict CTR on Custom Input")

    categorical_cols = ["City", "Gender", "Country"]
    input_data = {}

    for col in X.columns:
        if col in categorical_cols and col in label_encoders:
            encoder = label_encoders[col]
            options = encoder.classes_
            input_data[col] = st.selectbox(f"Select {col}", options)
        elif np.issubdtype(X[col].dtype, np.number):
            input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        else:
            unique_vals = sorted(X[col].astype(str).unique().tolist())
            input_data[col] = st.selectbox(f"Select {col}", unique_vals)

    if st.button("🔮 Predict CTR Using All Models"):
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns using saved encoders
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])

        predictions, binary_outputs = {}, {}
        for name, model in models.items():
            try:
                prob = model.predict_proba(input_df)[0, 1]
                predictions[name] = prob
                binary_outputs[name] = int(prob >= 0.5)
            except Exception as e:
                predictions[name] = None
                binary_outputs[name] = None
                st.warning(f"{name} could not predict: {e}")

        # Display results
        pred_df = pd.DataFrame(predictions.items(), columns=["Model", "Predicted CTR Probability"])
        bin_df = pd.DataFrame(binary_outputs.items(), columns=["Model", "Predicted Class (0/1)"])
        st.write("### 🔍 Model Probabilities")
        st.dataframe(pred_df.style.highlight_max(axis=0, color="lightblue"))
        st.write("### 🧮 Predicted Classes")
        st.dataframe(bin_df.style.highlight_max(axis=0, color="lightgreen"))

        valid_probs = [v for v in predictions.values() if v is not None]
        if valid_probs:
            combined_pred = np.mean(valid_probs)
            final_class = int(combined_pred >= 0.5)
            st.success(f"✅ **Combined CTR Probability:** {combined_pred:.3f}")
            st.info(f"🧩 **Final Predicted Class:** {final_class}")
else:
    st.info("👈 Upload a dataset to make predictions.")
