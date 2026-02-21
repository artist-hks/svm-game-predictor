import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ---------- SESSION ANALYTICS ----------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "best_acc_pct" not in st.session_state:
    st.session_state.best_acc_pct = "—"
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Video Game Sales Predictor",
    page_icon="🎮",
    layout="wide"
)

# ---------------- PREMIUM GLOBAL CSS ----------------
st.markdown("""
<style>
:root {
    --bg-main: #0b1220;
    --card-bg: rgba(17, 25, 40, 0.75);
    --accent-green: #22c55e;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
}

/* ---- Animated Gradient Glow ---- */
.hero-container {
    position: relative;
    padding: 30px 34px;
    border-radius: 22px;
    background: linear-gradient(135deg, #020617 0%, #0f172a 60%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    margin-bottom: 1.4rem;
}

.hero-container::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: linear-gradient(
        120deg,
        rgba(34,197,94,0.25),
        rgba(59,130,246,0.25),
        rgba(168,85,247,0.25),
        rgba(34,197,94,0.25)
    );
    filter: blur(38px);
    opacity: 0.55;
    animation: heroGlow 8s linear infinite;
    z-index: 0;
}

@keyframes heroGlow {
    0% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.05); }
    100% { transform: rotate(360deg) scale(1); }
}

.hero-title,
.hero-subtitle,
.hero-badges {
    position: relative;
    z-index: 2;
}

/* ---- Glass Navbar ---- */
.glass-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-radius: 14px;
    background: rgba(17, 25, 40, 0.55);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

.nav-left {
    font-weight: 600;
    letter-spacing: 0.3px;
}

.nav-right {
    font-size: 12px;
    color: #9aa0a6;
}

/* ---- Status Chips ---- */
.status-chip {
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    margin-left: 6px;
}

/* ---- Netflix Hero Header ---- */
.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

.hero-subtitle {
    color: #9aa0a6;
    font-size: 15px;
    margin-bottom: 14px;
}

.hero-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.hero-badge {
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 12px;
}

/* ---- Main container ---- */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---- Glass Card ---- */
.glass-card {
    padding: 24px;
    border-radius: 18px;
    background: rgba(17, 25, 40, 0.78);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow:
        0 10px 30px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ---- Custom Metrics Layout ---- */
.metric-row {
    display: flex;
    gap: 12px;
}

.metric-box {
    flex: 1;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
}

.metric-label {
    font-size: 13px;
    color: #9aa0a6;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
}

.metric-total {
    margin-top: 16px;
    background: linear-gradient(145deg, rgba(99,102,241,0.1) 0%, rgba(99,102,241,0.02) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-total .metric-value {
    color: #818cf8;
    font-size: 36px;
}

.metric-total .metric-label {
    color: #9aa0a6;
}

/* ---- Section spacing ---- */
.section-gap {
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* ---- Tabs Styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(17, 25, 40, 0.4);
    border-radius: 8px 8px 0 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: none;
    padding: 10px 16px;
    color: #9aa0a6;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(17, 25, 40, 0.9);
    color: #fff !important;
    border-top: 2px solid var(--accent-green) !important;
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background-color: rgba(17, 25, 40, 0.5);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    # Direct loading - No training inside the app! 🚀
    base_model = joblib.load("svm_model.pkl")
    calibrated_model = joblib.load("calibrated_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return base_model, calibrated_model, scaler

model, calibrated_model, scaler = load_assets()

# ----- LOAD DATASET  -----
@st.cache_data
def load_dataset():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna(subset=["Global_Sales"])
    return df

df_games = load_dataset()

# ---------- DRIFT BASELINE ----------
@st.cache_data
def compute_training_baseline(df):
    base = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna()
    return {
        "mean": base.mean(),
        "std": base.std()
    }

drift_baseline = compute_training_baseline(df_games)

# ---------------- MODEL COMPARISON DATA ----------------
@st.cache_resource
def compute_cv_scores(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            use_label_encoder=False, random_state=42
        )
    }
    results = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std()}
    return results

@st.cache_resource
def train_comparison_models(X_train, y_train):
    timings = {}

    start = time.perf_counter()
    svm = SVC(probability=True, random_state=42).fit(X_train, y_train)
    timings["SVM"] = time.perf_counter() - start

    start = time.perf_counter()
    nb = GaussianNB().fit(X_train, y_train)
    timings["Naive Bayes"] = time.perf_counter() - start

    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    timings["KNN"] = time.perf_counter() - start

    start = time.perf_counter()
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    timings["Decision Tree"] = time.perf_counter() - start

    start = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
        random_state=42
    ).fit(X_train, y_train)
    timings["XGBoost"] = time.perf_counter() - start

    return svm, nb, knn, dt, xgb, timings

@st.cache_data
def prepare_model_data(df):
    temp = df.dropna(subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]).copy()
    temp["Sales_Class"] = pd.qcut(temp["Global_Sales"], q=3, labels=[0, 1, 2])
    X = temp[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = temp["Sales_Class"].astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = prepare_model_data(df_games)

# ---------------- MEMORY SAFE SIMILARITY ----------------
@st.cache_resource
def prepare_similarity_engine(df):
    try:
        sim_df = df[["Name", "Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].dropna().reset_index(drop=True)
        features = sim_df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
        nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        nn_model.fit(features)
        return sim_df, nn_model
    except Exception:
        return None, None

sim_games, nn_model = prepare_similarity_engine(df_games)

# ---------------- HEADER ----------------
st.toast("Model ready", icon="🤖")

best_acc_display = st.session_state.get("best_acc_pct", "—")
latency_display = st.session_state.get("last_latency", 0)

st.markdown(f"""
<div class="glass-nav">
    <div class="nav-left">⚡ ML Ops Dashboard</div>
    <div class="nav-right">
        <span class="status-chip">Best Acc: {best_acc_display}</span>
        <span class="status-chip">Latency: {latency_display:.1f} ms</span>
        <span class="status-chip">Models: 5+</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎮 Video Game Sales Intelligence</div>
    <div class="hero-subtitle">
        Production-grade ML dashboard for sales prediction, explainability,
        and interactive analytics.
    </div>
    <div class="hero-badges">
        <div class="hero-badge">SVM Primary</div>
        <div class="hero-badge">XGBoost Benchmark</div>
        <div class="hero-badge">SHAP Enabled</div>
        <div class="hero-badge">Drift Monitor</div>
        <div class="hero-badge">Calibrated Probs</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (REAL-TIME SLIDERS) ----------------
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.caption("Adjust inputs and explore insights")
st.sidebar.divider()
st.sidebar.header("🎯 Regional Sales Input")

na_sales = st.sidebar.slider("NA Sales", 0.0, 10.0, 0.5, 0.1)
eu_sales = st.sidebar.slider("EU Sales", 0.0, 10.0, 0.3, 0.1)
jp_sales = st.sidebar.slider("JP Sales", 0.0, 10.0, 0.1, 0.1)
other_sales = st.sidebar.slider("Other Sales", 0.0, 10.0, 0.05, 0.05)

use_calibrated = st.sidebar.toggle("Use calibrated probabilities", value=True)

features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)

st.sidebar.markdown("---")
st.sidebar.caption(f"Session predictions: {st.session_state.prediction_count}")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🎯 Prediction",
    "📈 Feature Importance",
    "📊 Model Comparison",
    "🧠 SHAP Explainability",
    "📊 Analytics Dashboard",
    "🎮 Game Recommender",
    "🧪 Model Diagnostics",
    "🎛️ What-If Simulator",
    "🧭 Drift Monitor",
    "📘 About Model"
])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Current Input Snapshot")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">NA Sales</div>
                <div class="metric-value">{na_sales:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">EU Sales</div>
                <div class="metric-value">{eu_sales:.2f}</div>
            </div>
        </div>
        <div class="metric-row" style="margin-top: 12px;">
            <div class="metric-box">
                <div class="metric-label">JP Sales</div>
                <div class="metric-value">{jp_sales:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Other Sales</div>
                <div class="metric-value">{other_sales:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        total_sales = na_sales + eu_sales + jp_sales + other_sales
        st.markdown(f"""
        <div class="metric-total">
            <div class="metric-label">🌍 Total Regional Sales</div>
            <div class="metric-value">{total_sales:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom: 0;'>🤖 Model Prediction</h3>", unsafe_allow_html=True)

        with st.spinner("Running ML inference..."):
            start_inf = time.perf_counter()
            active_model = calibrated_model if use_calibrated else model
            pred = active_model.predict(features_scaled)[0]
            proba = active_model.predict_proba(features_scaled)[0]
            confidence = float(np.max(proba) * 100)

        latency_ms = (time.perf_counter() - start_inf) * 1000
        st.session_state.last_latency = latency_ms
        st.session_state.prediction_count += 1

        labels = {
            0: ("📉 Low Sales", "var(--accent-red)"),
            1: ("📊 Medium Sales", "var(--accent-orange)"),
            2: ("🚀 High Sales", "var(--accent-green)")
        }

        text, color = labels.get(pred, ("Unknown", "#9aa0a6"))

        st.markdown(
            f"""
            <div style="text-align:center; padding: 20px 10px;">
                <h2 style="color:{color}; margin-bottom: 8px; font-size: 32px;">{text}</h2>
                <div style="font-size: 56px; font-weight: 800; margin-top: 10px; line-height: 1;">
                    {confidence:.2f}%
                </div>
                <div style="color: #9aa0a6; font-size: 14px; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px;">
                    Model confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — FEATURE IMPORTANCE
# ============================================================
with tab2:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Feature Importance (Permutation)")
    st.info("Permutation importance approximates feature impact for SVM.")

    try:
        sample_df = df_games[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].dropna().sample(300, random_state=42)
        X_sample = scaler.transform(sample_df)
        y_sample = model.predict(X_sample)

        perm = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)

        importance_df = pd.DataFrame({
            "Feature": ["NA", "EU", "JP", "Other"],
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(importance_df, x="Feature", y="Importance", color="Importance", color_continuous_scale="viridis")
        fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning("Feature importance could not be computed.")

# ============================================================
# TAB 3 — MODEL COMPARISON
# ============================================================
with tab3:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Real Model Comparison")
    
    X_full = pd.concat([X_train_cmp, X_test_cmp], axis=0)
    y_full = pd.concat([y_train_cmp, y_test_cmp], axis=0)
    cv_results = compute_cv_scores(X_full, y_full)
    
    svm_cmp, nb_cmp, knn_cmp, dt_cmp, xgb_cmp, timings = train_comparison_models(X_train_cmp, y_train_cmp)

    preds = {
        "SVM": svm_cmp.predict(X_test_cmp),
        "Naive Bayes": nb_cmp.predict(X_test_cmp),
        "KNN": knn_cmp.predict(X_test_cmp),
        "Decision Tree": dt_cmp.predict(X_test_cmp),
        "XGBoost": xgb_cmp.predict(X_test_cmp),
    }

    acc_data = [
        {"Model": "SVM", "Accuracy": accuracy_score(y_test_cmp, preds["SVM"]), "Training Time (s)": timings["SVM"]},
        {"Model": "Naive Bayes", "Accuracy": accuracy_score(y_test_cmp, preds["Naive Bayes"]), "Training Time (s)": timings["Naive Bayes"]},
        {"Model": "KNN", "Accuracy": accuracy_score(y_test_cmp, preds["KNN"]), "Training Time (s)": timings["KNN"]},
        {"Model": "Decision Tree", "Accuracy": accuracy_score(y_test_cmp, preds["Decision Tree"]), "Training Time (s)": timings["Decision Tree"]},
        {"Model": "XGBoost", "Accuracy": accuracy_score(y_test_cmp, preds["XGBoost"]), "Training Time (s)": timings["XGBoost"]},
    ]

    acc_df = pd.DataFrame(acc_data)
    best_acc_value = acc_df["Accuracy"].max()
    st.session_state.best_acc_pct = f"{best_acc_value*100:.2f}%"

    best_acc_model = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    fastest_model = acc_df.sort_values("Training Time (s)").iloc[0]["Model"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid var(--accent-green);">
            <div style="color: #9aa0a6; font-size: 14px; text-transform: uppercase;">🏆 Most Accurate Model</div>
            <div style="font-size: 24px; font-weight: 700; margin-top: 8px;">{best_acc_model}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid var(--accent-orange);">
            <div style="color: #9aa0a6; font-size: 14px; text-transform: uppercase;">⚡ Fastest Training Model</div>
            <div style="font-size: 24px; font-weight: 700; margin-top: 8px;">{fastest_model}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 Accuracy Comparison")
    fig_acc = px.bar(acc_df, x="Model", y="Accuracy", color="Accuracy", color_continuous_scale="viridis")
    fig_acc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("#### 🧪 Cross-Validation Stability")
    cv_df = pd.DataFrame([{"Model": name, "CV Mean Accuracy": vals["mean"], "CV Std": vals["std"]} for name, vals in cv_results.items()])
    fig_cv = px.bar(cv_df, x="Model", y="CV Mean Accuracy", error_y="CV Std", color="CV Mean Accuracy", title="5-Fold Stratified CV Accuracy", color_continuous_scale="viridis")
    fig_cv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("#### ⏱️ Training Time Comparison")
    fig_time = px.bar(acc_df, x="Model", y="Training Time (s)", color="Training Time (s)", color_continuous_scale="magma")
    fig_time.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔥 Confusion Matrices")
    cols = st.columns(len(preds))
    for i, (name, p) in enumerate(preds.items()):
        cm = confusion_matrix(y_test_cmp, p)
        with cols[i]:
            fig, ax = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(name, color="white", fontsize=10)
            ax.set_xlabel("Predicted", color="white", fontsize=8)
            ax.set_ylabel("Actual", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=8)
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            st.pyplot(fig)

# ============================================================
# TAB 4 — SHAP EXPLAINABILITY
# ============================================================
with tab4:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🧠 SHAP Explainability")
    st.caption("Local explanation of the current prediction")

    try:
        @st.cache_data
        def get_shap_background(df):
            sample = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna().sample(50, random_state=42)
            return scaler.transform(sample)

        background = get_shap_background(df_games)

        @st.cache_resource
        def get_explainer(_background):
            return shap.KernelExplainer(model.predict_proba, _background)

        explainer = get_explainer(background)

        @st.cache_data
        def compute_shap_values(input_features):
            return explainer.shap_values(input_features, nsamples=50)

        with st.spinner("Analyzing feature impact with SHAP..."):
            shap_values = compute_shap_values(features_scaled)

        if isinstance(shap_values, list):
            class_idx = int(pred)
            shap_for_class = shap_values[class_idx][0]
        else:
            shap_for_class = shap_values[0]

        feature_names = ["NA", "EU", "JP", "Other"]
        shap_for_class = np.array(shap_for_class).flatten()
        min_len = min(len(feature_names), len(shap_for_class))

        shap_df = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Impact": np.abs(shap_for_class[:min_len])
        }).sort_values("Impact", ascending=False)

        st.markdown("#### 📊 Feature Impact Breakdown")
        fig_shap = px.bar(shap_df, x="Feature", y="Impact", color="Impact", title=f"Impact on Predicting Class: {labels[pred][0]}", color_continuous_scale="viridis")
        fig_shap.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.error("SHAP failed to compute.")
        st.caption(f"Error details: {str(e)}")

# ============================================================
# TAB 5 — ANALYTICS DASHBOARD
# ============================================================
with tab5:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Video Game Sales Analytics")

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Global Sales Distribution")
        fig_hist = px.histogram(df_games, x="Global_Sales", nbins=50, color_discrete_sequence=["#3b82f6"])
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Top Publishers")
        top_pub = df_games.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_pub = px.bar(top_pub, x="Publisher", y="Global_Sales", color="Global_Sales", color_continuous_scale="viridis")
        fig_pub.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pub, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    colC, colD = st.columns(2)

    with colC:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎮 Platform-wise Sales")
        plat_sales = df_games.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_plat = px.bar(plat_sales, x="Platform", y="Global_Sales", color="Global_Sales", color_continuous_scale="viridis")
        fig_plat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_plat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colD:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Sales Correlation")
        corr = df_games[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 6 — GAME RECOMMENDER SYSTEM
# ============================================================
with tab6:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🎮 Video Game Recommender")
    st.caption("Find top games based on platform, genre, and year.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_platform = st.selectbox("Select Platform", ["All"] + sorted(df_games["Platform"].dropna().unique()))
    with col2:
        selected_genre = st.selectbox("Select Genre", ["All"] + sorted(df_games["Genre"].dropna().unique()))
    with col3:
        year_min, year_max = int(df_games["Year"].min()), int(df_games["Year"].max())
        year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    filtered_df = df_games.copy()
    if selected_platform != "All": filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]
    if selected_genre != "All": filtered_df = filtered_df[filtered_df["Genre"] == selected_genre]
    filtered_df = filtered_df[(filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])]

    st.markdown("#### 🏆 Top Recommended Games")
    top_games = filtered_df.sort_values("Global_Sales", ascending=False).head(10)[["Name", "Platform", "Genre", "Year", "Global_Sales"]]

    if len(top_games) == 0:
        st.warning("No games match the current filters.")
    else:
        st.dataframe(top_games, use_container_width=True)
        fig_top = px.bar(top_games, x="Name", y="Global_Sales", color="Global_Sales", color_continuous_scale="viridis")
        fig_top.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧠 Similar Game Finder (Advanced)")
    if sim_games is not None and nn_model is not None:
        selected_game = st.selectbox("Select a game to find similar ones", sim_games["Name"].values)
        try:
            idx = sim_games[sim_games["Name"] == selected_game].index[0]
            distances, indices = nn_model.kneighbors([sim_games.loc[idx, ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]]], n_neighbors=11)
            similar_games = sim_games.iloc[indices[0][1:]][["Name", "Platform", "Genre", "Global_Sales"]]
            st.success("Top similar games:")
            st.dataframe(similar_games, use_container_width=True)
        except Exception as e:
            st.error("Could not compute similar games.")
    else:
        st.error("Similarity engine failed to initialize.")

# ============================================================
# TAB 7 — MODEL DIAGNOSTICS
# ============================================================
with tab7:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🧪 Model Diagnostics")

    best_model_name = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    model_map = {"SVM": svm_cmp, "Naive Bayes": nb_cmp, "KNN": knn_cmp, "Decision Tree": dt_cmp, "XGBoost": xgb_cmp}
    best_model = model_map[best_model_name]

    st.success(f"Analyzing best model: **{best_model_name}**")

    st.markdown("#### 📉 Multiclass ROC Curves")
    try:
        y_test_bin = label_binarize(y_test_cmp, classes=[0, 1, 2])
        y_proba = best_model.predict_proba(X_test_cmp)
        roc_data = []
        for i, class_name in enumerate(["Low", "Medium", "High"]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_data.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Class": class_name}))
        
        fig_roc = px.line(pd.concat(roc_data), x="FPR", y="TPR", color="Class", title="ROC Curve (One-vs-Rest)")
        fig_roc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception:
        st.warning("ROC could not be computed.")

    st.markdown("#### 📊 Precision & Recall")
    y_pred_best = best_model.predict(X_test_cmp)
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall"],
        "Score": [precision_score(y_test_cmp, y_pred_best, average="macro"), recall_score(y_test_cmp, y_pred_best, average="macro")]
    })
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("#### 🧮 Test Set Class Distribution")
    dist_df = y_test_cmp.value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]
    fig_dist = px.bar(dist_df, x="Class", y="Count", color="Count", color_continuous_scale="viridis")
    fig_dist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 🚨 Misclassification Insight")
    misclassified = (y_test_cmp != y_pred_best).sum()
    total = len(y_test_cmp)
    st.metric("Misclassified Samples", f"{misclassified} / {total}", delta=f"{(misclassified/total)*100:.2f}% error", delta_color="inverse")

# ============================================================
# TAB 8 — WHAT-IF SIMULATOR
# ============================================================
with tab8:
    @st.fragment
    def what_if_simulator():
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("### 🎛️ What-If Simulator")
        st.caption("Interactively explore how regional sales affect predictions. (Runs independently)")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sim_na = st.slider("NA Sales (Sim)", 0.0, 10.0, 1.0, 0.1, key="sim_na")
            sim_eu = st.slider("EU Sales (Sim)", 0.0, 10.0, 1.0, 0.1, key="sim_eu")
        with c2:
            sim_jp = st.slider("JP Sales (Sim)", 0.0, 10.0, 0.5, 0.1, key="sim_jp")
            sim_other = st.slider("Other Sales (Sim)", 0.0, 10.0, 0.3, 0.1, key="sim_other")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sim_features = np.array([[sim_na, sim_eu, sim_jp, sim_other]])
        sim_scaled = scaler.transform(sim_features)
        sim_pred = model.predict(sim_scaled)[0]
        sim_proba = model.predict_proba(sim_scaled)[0]
        class_labels = ["Low", "Medium", "High"]

        st.markdown("#### 🔮 Simulated Prediction")
        color_map = {"Low": "var(--accent-red)", "Medium": "var(--accent-orange)", "High": "var(--accent-green)"}
        pred_label = class_labels[sim_pred]
        
        st.markdown(f"<h3 style='color: {color_map[pred_label]}; border: 1px solid {color_map[pred_label]}; padding: 16px; border-radius: 12px; text-align: center; background: rgba(17, 25, 40, 0.5);'>Predicted Class: {pred_label}</h3>", unsafe_allow_html=True)

        st.markdown("#### 📉 Decision Sensitivity Curve")
        sweep_vals = np.linspace(0, 10, 60)
        sweep_features = np.zeros((len(sweep_vals), 4))
        sweep_features[:, 0] = sweep_vals
        sweep_features[:, 1] = sim_eu
        sweep_features[:, 2] = sim_jp
        sweep_features[:, 3] = sim_other
        
        sweep_scaled = scaler.transform(sweep_features)
        sweep_probs = model.predict_proba(sweep_scaled)[:, 2] 

        fig_boundary = px.line(pd.DataFrame({"NA_Sales": sweep_vals, "High_Sales_Prob": sweep_probs}), x="NA_Sales", y="High_Sales_Prob", title="How NA Sales influences High Sales probability")
        fig_boundary.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_boundary, use_container_width=True)

        st.markdown("#### 📊 Class Probabilities")
        fig_prob_sim = px.bar(pd.DataFrame({"Class": class_labels, "Probability": sim_proba}), x="Class", y="Probability", color="Class", color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"})
        fig_prob_sim.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_prob_sim, use_container_width=True)

    what_if_simulator()

# ============================================================
# TAB 9 — DRIFT MONITOR
# ============================================================
with tab9:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🧭 Model Drift Monitor")
    st.caption("Compare live inputs against training distribution.")

    current_input = pd.Series({"NA_Sales": na_sales, "EU_Sales": eu_sales, "JP_Sales": jp_sales, "Other_Sales": other_sales})
    baseline_mean = drift_baseline["mean"]
    baseline_std = drift_baseline["std"]

    drift_scores = ((current_input - baseline_mean) / (baseline_std + 1e-6)).abs()
    drift_df = pd.DataFrame({"Feature": drift_scores.index, "Drift Score (|z|)": drift_scores.values}).sort_values("Drift Score (|z|)", ascending=False)

    max_drift = drift_scores.max()
    if max_drift < 1: st.success("✅ Input is within normal training range.")
    elif max_drift < 2.5: st.warning("⚠️ Moderate drift detected. Monitor recommended.")
    else: st.error("🚨 High drift! Model may be unreliable.")

    st.markdown("#### 📊 Feature Drift Scores")
    fig_drift = px.bar(drift_df, x="Feature", y="Drift Score (|z|)", color="Drift Score (|z|)", color_continuous_scale="Reds")
    fig_drift.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig_drift, use_container_width=True)

    st.markdown("#### 🔍 Detailed Comparison")
    st.dataframe(pd.DataFrame({"Current": current_input, "Training Mean": baseline_mean, "Training Std": baseline_std}), use_container_width=True)

# ============================================================
# TAB 10 — ABOUT MODEL
# ============================================================
with tab10:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📘 About Model — Full Technical Documentation")
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-bottom:0;">🎮 Video Game Sales Intelligence System</h3>
        <p style="color:#9aa0a6;">
        A production-oriented machine learning dashboard designed to demonstrate
        disciplined end-to-end ML engineering including preprocessing, multi-model
        benchmarking, calibration, explainability, and drift awareness.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🎯 1. Problem Formulation (Detailed)", expanded=True):
        st.markdown("""
        The core objective is to estimate the **global commercial performance tier** of a video game.
        We frame this as a **multi-class classification problem** (Low, Medium, High) to improve robustness to extreme values and interpretability.
        """)

    with st.expander("📊 2. Dataset Deep Dive"):
        st.markdown("""
        Trained on the **vgsales dataset**. Features used: NA_Sales, EU_Sales, JP_Sales, Other_Sales.
        Target is quantile-binned Global_Sales to mitigate class imbalance.
        """)

    with st.expander("🧠 3. Data Preprocessing Pipeline"):
        st.markdown("""
        - **Cleaning**: Removal of missing rows.
        - **Splitting**: Stratified train-test split.
        - **Scaling**: StandardScaler normalization (critical for SVM/KNN).
        """)

    with st.expander("🤖 4. Model Architecture & Rationale"):
        st.markdown("""
        - **SVM (Primary)**: Strong performance on normalized tabular data.
        - **XGBoost**: Gradient boosting benchmark.
        - **KNN / Naive Bayes / Decision Tree**: Baselines for comparison.
        """)

    with st.expander("🧪 5. Evaluation Methodology"):
        st.markdown("Uses hold-out accuracy, stratified 5-fold CV, confusion matrices, and multiclass ROC.")

    with st.expander("📉 6. Probability Calibration"):
        st.markdown("Supports Platt scaling via CalibratedClassifierCV to improve probability reliability.")

    with st.expander("🔍 7. Explainability Framework"):
        st.markdown("Includes permutation feature importance and SHAP local explanations.")

    with st.expander("🧭 8. Drift Monitoring Strategy"):
        st.markdown("A lightweight drift detector compares live user inputs against the training distribution using z-score distance.")

    st.markdown("---")
    st.caption("Model Version: v1.0 • Production ML System • Built by HKS")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by HKS • Machine Learning • UI/UX")