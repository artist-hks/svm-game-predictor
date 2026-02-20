import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import shap

from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Video Game Sales Predictor",
    page_icon="🎮",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ---------------- HEADER ----------------
st.title("🎮 Video Game Sales Predictor Pro")
st.caption("Advanced ML dashboard with explainability")

# ---------------- SIDEBAR (REAL-TIME SLIDERS) ----------------
st.sidebar.header("🎯 Regional Sales Input")

na_sales = st.sidebar.slider("NA Sales", 0.0, 10.0, 0.5, 0.1)
eu_sales = st.sidebar.slider("EU Sales", 0.0, 10.0, 0.3, 0.1)
jp_sales = st.sidebar.slider("JP Sales", 0.0, 10.0, 0.1, 0.1)
other_sales = st.sidebar.slider("Other Sales", 0.0, 10.0, 0.05, 0.05)

features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)

pred = model.predict(features_scaled)[0]
proba = model.predict_proba(features_scaled)[0]
confidence = np.max(proba) * 100

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction",
    "📈 Feature Importance",
    "📊 Model Comparison",
    "🧠 SHAP Explainability"
])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.subheader("Prediction Result")

    labels = {
        0: ("📉 Low Sales", "#ef4444"),
        1: ("📊 Medium Sales", "#f59e0b"),
        2: ("🚀 High Sales", "#22c55e")
    }

    text, color = labels[pred]

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:14px;
            background:#111827;
            border:1px solid #2d3748;
            text-align:center;">
            <h2 style="color:{color}; margin-bottom:0;">{text}</h2>
            <p style="color:#9aa0a6;">Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": ["Low", "Medium", "High"],
        "Probability": proba
    })

    fig_prob = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        color="Probability",
        text="Probability"
    )
    fig_prob.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_prob, use_container_width=True)

# ============================================================
# TAB 2 — FEATURE IMPORTANCE (Permutation for SVM)
# ============================================================
with tab2:
    st.subheader("Feature Importance (Permutation)")

    st.info("Permutation importance approximates feature impact for SVM.")

    # dummy small sample for speed
    try:
        X_sample = scaler.transform(
            np.random.rand(200, 4)
        )
        y_sample = model.predict(X_sample)

        perm = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=42
        )

        importance_df = pd.DataFrame({
            "Feature": ["NA", "EU", "JP", "Other"],
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Importance"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.warning("Feature importance could not be computed.")

# ============================================================
# TAB 3 — MODEL COMPARISON
# ============================================================
with tab3:
    st.subheader("Model Comparison")

    st.caption("Quick comparison using synthetic sample")

    try:
        # synthetic data for demo comparison
        X_demo = np.random.rand(500, 4)
        y_demo = model.predict(scaler.transform(X_demo))

        nb = GaussianNB().fit(X_demo, y_demo)
        knn = KNeighborsClassifier().fit(X_demo, y_demo)

        acc_svm = accuracy_score(y_demo, model.predict(scaler.transform(X_demo)))
        acc_nb = accuracy_score(y_demo, nb.predict(X_demo))
        acc_knn = accuracy_score(y_demo, knn.predict(X_demo))

        comp_df = pd.DataFrame({
            "Model": ["SVM", "Naive Bayes", "KNN"],
            "Accuracy": [acc_svm, acc_nb, acc_knn]
        })

        fig_cmp = px.bar(
            comp_df,
            x="Model",
            y="Accuracy",
            color="Accuracy"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    except Exception:
        st.warning("Model comparison unavailable.")

# ============================================================
# TAB 4 — SHAP EXPLAINABILITY
# ============================================================
with tab4:
    st.subheader("SHAP Explainability")

    st.caption("Local explanation of the current prediction")

    try:
        background = shap.sample(
            scaler.transform(np.random.rand(200, 4)),
            50
        )

        explainer = shap.KernelExplainer(
            model.predict_proba,
            background
        )

        shap_values = explainer.shap_values(features_scaled, nsamples=100)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()

        fig_shap = shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            matplotlib=True
        )

        st.pyplot(bbox_inches='tight')

    except Exception:
        st.warning("SHAP visualization unavailable (first load may be slow).")

# ---------------- FOOTER ----------------
st.caption("Built by HKS • ML + UI/UX • Advanced Edition")