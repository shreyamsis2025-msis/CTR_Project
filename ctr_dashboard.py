# ==========================================================
# 📊 CTR Prediction Dashboard (Final Version)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle, os, warnings

# ----------------------------------------------------------
# 🧹 Suppress warnings
# ----------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ----------------------------------------------------------
# 🧱 Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="CTR Prediction Dashboard", layout="wide")
st.title("📈 CTR Prediction Model Comparison Dashboard")
st.markdown("Compare pre-trained CTR prediction models, view feature importances, and make live predictions!")

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
        if filename.endswith(".pkl"):
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
# 📊 Load Precomputed Model Performance
# ----------------------------------------------------------
perf_file = os.path.join(model_dir, "model_performance.csv")
feature_file = os.path.join(model_dir, "train_features.txt")
roc_file = os.path.join(model_dir, "roc_data.npz")

if os.path.exists(perf_file):
    st.subheader("⚡ Model Performance (K-Fold CV Results)")
    perf_df = pd.read_csv(perf_file)
    st.dataframe(perf_df.style.highlight_max(axis=0, color="lightgreen"))
else:
    st.warning("⚠️ No precomputed performance file found. Run ctr_prediction.ipynb to generate it.")

# ----------------------------------------------------------
# 📈 ROC–AUC Curves (Loaded from Training)
# ----------------------------------------------------------
if os.path.exists(roc_file):
    st.subheader("📈 ROC–AUC Curve Comparison (From Training)")
    roc_data = np.load(roc_file, allow_pickle=True)
    plt.figure(figsize=(8, 6))
    for name in roc_data.files:
        d = roc_data[name].item()
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{name} (AUC={d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC–AUC Curve Comparison (From Training)")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)
else:
    st.info("ℹ️ ROC data not found. Please re-run ctr_prediction.ipynb to generate it.")

# ----------------------------------------------------------
# 📊 Feature Importance Section
# ----------------------------------------------------------
st.subheader("🏆 Feature Importance Viewer")
selected_model_name = st.selectbox("Select a model to view feature importance:", list(models.keys()))
if os.path.exists(feature_file):
    with open(feature_file, "r") as f:
        train_features = [line.strip() for line in f.readlines()]
else:
    train_features = []

if selected_model_name:
    model = models[selected_model_name]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": train_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

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
        st.warning(f"⚠️ {selected_model_name} does not provide feature importances.")

# ----------------------------------------------------------
# 📂 Upload Dataset for Prediction
# ----------------------------------------------------------
st.sidebar.header("📥 Upload Dataset for Predictions")
uploaded_file = st.sidebar.file_uploader("Upload your preprocessed dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")

    # Load training feature list
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            train_features = [line.strip() for line in f.readlines()]
    else:
        train_features = list(data.columns)
        st.warning("⚠️ Training feature list not found. Proceeding with uploaded columns.")

    # Drop unseen columns automatically
    X = data.copy()
    for col in X.columns:
        if col not in train_features:
            X.drop(columns=col, inplace=True)
            st.warning(f"Dropped unseen column: {col}")

    # Add missing training features
    for col in train_features:
        if col not in X.columns:
            X[col] = 0

    X = X[train_features]

    # Encode categorical columns
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    encoder_mappings = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoder_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # ----------------------------------------------------------
    # 🎯 Custom CTR Prediction
    # ----------------------------------------------------------
    st.subheader("🎯 Predict CTR on Custom Input")
    input_data = {}
    for col in train_features:
        if np.issubdtype(X[col].dtype, np.number):
            input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        else:
            unique_vals = X[col].unique().tolist()
            input_data[col] = st.selectbox(f"Select {col}", unique_vals)

    if st.button("🔮 Predict CTR Using All Models"):
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        predictions = {}
        binary_outputs = {}
        for name, model in models.items():
            try:
                prob = model.predict_proba(input_df)[0, 1]
                pred_class = int(prob >= 0.5)
                predictions[name] = prob
                binary_outputs[name] = pred_class
            except Exception as e:
                predictions[name] = None
                binary_outputs[name] = None
                st.warning(f"{name} could not predict: {e}")

        # Display probabilities
        pred_df = pd.DataFrame(predictions.items(), columns=["Model", "Predicted CTR Probability"])
        st.write("### 🔍 Model Probabilities")
        st.dataframe(pred_df.style.highlight_max(axis=0, color="lightblue"))

        # Display binary output (0 or 1)
        bin_df = pd.DataFrame(binary_outputs.items(), columns=["Model", "Predicted Class (0=No, 1=Yes)"])
        st.write("### 🧮 Final Predicted Classes")
        st.dataframe(bin_df.style.highlight_max(axis=0, color="lightgreen"))

        # Combined Output (average)
        valid_probs = [v for v in predictions.values() if v is not None]
        if valid_probs:
            combined_pred = np.mean(valid_probs)
            final_class = int(combined_pred >= 0.5)
            st.success(f"✅ **Combined CTR Probability:** {combined_pred:.3f}")
            st.info(f"🧩 **Final Predicted Class:** {final_class}")

else:
    st.info("👈 Upload a dataset to make predictions.")
