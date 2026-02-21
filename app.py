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
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

# ---------- SESSION ANALYTICS ----------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Video Game Sales Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PREMIUM GLOBAL CSS ----------------
st.markdown("""
<style>
/* Base Theme Adjustments */
:root {
    --bg-main: #09090b;
    --card-bg: rgba(24, 24, 27, 0.65);
    --card-border: rgba(255, 255, 255, 0.08);
    --accent-primary: #6366f1;
    --accent-green: #10b981;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --text-muted: #a1a1aa;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Container Padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* ---- Animated Gradient Glow Hero ---- */
.hero-container {
    position: relative;
    padding: 40px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(9,9,11,0.9) 0%, rgba(24,24,27,0.9) 100%);
    border: 1px solid var(--card-border);
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
}

.hero-container::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(99,102,241,0.15) 0%, transparent 50%);
    animation: pulseGlow 10s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}

@keyframes pulseGlow {
    0% { transform: scale(0.8); opacity: 0.5; }
    100% { transform: scale(1.2); opacity: 1; }
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(to right, #fff, #a1a1aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
    max-width: 600px;
    line-height: 1.6;
}

.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-badge {
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 0.8rem;
    font-weight: 500;
    color: #e4e4e7;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
}

/* ---- Glass Navbar ---- */
.glass-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    border-radius: 16px;
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.nav-left {
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-right {
    display: flex;
    gap: 12px;
    align-items: center;
}

.status-chip {
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e4e4e7;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ---- Glass Cards ---- */
.glass-card {
    padding: 28px;
    border-radius: 20px;
    background: var(--card-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--card-border);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    border-color: rgba(255,255,255,0.12);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-title-sub {
    font-size: 0.9rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-left: auto;
}

/* ---- Metrics ---- */
.metric-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 16px;
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.03);
    margin-bottom: 12px;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
}

.metric-total {
    background: linear-gradient(145deg, rgba(99,102,241,0.1) 0%, rgba(99,102,241,0.02) 100%);
    border-color: rgba(99,102,241,0.2);
}
.metric-total .metric-value {
    color: #818cf8;
}
</style>
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

use_calibrated = st.sidebar.toggle(
    "Use calibrated probabilities",
    value=True
)

features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Session predictions: {st.session_state.prediction_count}"
)

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
    
    # ====== MAIN TWO COLUMN LAYOUT ======
    left_col, right_col = st.columns([1, 1.2], gap="large")

    # =====================================================
    # LEFT — INPUT SUMMARY CARD
    # =====================================================
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Current Input Snapshot")
        st.markdown("<br>", unsafe_allow_html=True)

        # Custom metric layout
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

    # =====================================================
    # RIGHT — HERO PREDICTION CARD
    # =====================================================
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
# TAB 2 — FEATURE IMPORTANCE (Permutation for SVM)
# ============================================================
with tab2:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Feature Importance (Permutation)")

    st.info("Permutation importance approximates feature impact for SVM.")

    # dummy small sample for speed
    try:
        sample_df = df_games[
        ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
        ].dropna().sample(300, random_state=42)

        X_sample = scaler.transform(sample_df)
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
            color="Importance",
            color_continuous_scale="viridis"
        )
        fig_imp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )
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
    
    # ----- Train models with timing -----
    svm_cmp, nb_cmp, knn_cmp, dt_cmp, xgb_cmp, timings = train_comparison_models(
        X_train_cmp, y_train_cmp
    )

    svm_time = timings["SVM"]
    nb_time = timings["Naive Bayes"]
    knn_time = timings["KNN"]
    dt_time = timings["Decision Tree"]
    xgb_time = timings["XGBoost"]

    # probability predictions
    svm_proba = svm_cmp.predict_proba(X_test_cmp)
    nb_proba = nb_cmp.predict_proba(X_test_cmp)
    knn_proba = knn_cmp.predict_proba(X_test_cmp)
    dt_proba = dt_cmp.predict_proba(X_test_cmp)
    xgb_proba = xgb_cmp.predict_proba(X_test_cmp)
    
    # ----- Binarize labels for multi-class ROC -----
    y_test_bin = label_binarize(y_test_cmp, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    # ----- Predictions -----
    preds = {
        "SVM": svm_cmp.predict(X_test_cmp),
        "Naive Bayes": nb_cmp.predict(X_test_cmp),
        "KNN": knn_cmp.predict(X_test_cmp),
        "Decision Tree": dt_cmp.predict(X_test_cmp),
        "XGBoost": xgb_cmp.predict(X_test_cmp),
    }

    # ----- Accuracy table -----
    acc_data = [
        {
            "Model": "SVM",
            "Accuracy": accuracy_score(y_test_cmp, svm_cmp.predict(X_test_cmp)),
            "Training Time (s)": svm_time
        },
        {
            "Model": "Naive Bayes",
            "Accuracy": accuracy_score(y_test_cmp, nb_cmp.predict(X_test_cmp)),
            "Training Time (s)": nb_time
        },
        {
            "Model": "KNN",
            "Accuracy": accuracy_score(y_test_cmp, knn_cmp.predict(X_test_cmp)),
            "Training Time (s)": knn_time
        },
        {
            "Model": "Decision Tree",
            "Accuracy": accuracy_score(y_test_cmp, dt_cmp.predict(X_test_cmp)),
            "Training Time (s)": dt_time
        },
        {
            "Model": "XGBoost",
            "Accuracy": accuracy_score(y_test_cmp, xgb_cmp.predict(X_test_cmp)),
            "Training Time (s)": xgb_time
        },
    ]

    acc_df = pd.DataFrame(acc_data)
    best_acc_value = acc_df["Accuracy"].max()
    best_acc_pct = f"{best_acc_value*100:.2f}%"
    st.session_state.best_acc_pct = best_acc_pct

    # ---------- SUMMARY CARDS ----------
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

    fig_acc = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale="viridis"
    )
    fig_acc.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff"
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("#### 🧪 Cross-Validation Stability")

    cv_df = pd.DataFrame([
        {
            "Model": name,
            "CV Mean Accuracy": vals["mean"],
            "CV Std": vals["std"]
        }
        for name, vals in cv_results.items()
    ])

    fig_cv = px.bar(
        cv_df,
        x="Model",
        y="CV Mean Accuracy",
        error_y="CV Std",
        color="CV Mean Accuracy",
        title="5-Fold Stratified CV Accuracy",
        color_continuous_scale="viridis"
    )
    fig_cv.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff"
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("#### ⏱️ Training Time Comparison")

    fig_time = px.bar(
        acc_df,
        x="Model",
        y="Training Time (s)",
        color="Training Time (s)",
        color_continuous_scale="magma"
    )
    fig_time.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff"
    )
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
        # 1. Properly Cache Background Data (Only @st.cache_data)
        @st.cache_data
        def get_shap_background(df):
            sample = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna().sample(50, random_state=42)
            return scaler.transform(sample)

        background = get_shap_background(df_games)

        # 2. Cache the Heavy Explainer Object (@st.cache_resource)
        @st.cache_resource
        def get_explainer(_background):
            # We use a wrapper function so it only initializes once
            return shap.KernelExplainer(model.predict_proba, _background)

        explainer = get_explainer(background)

        # 3. Cache the SHAP Computation based on user input
        @st.cache_data
        def compute_shap_values(input_features):
            # It will only recalculate if input_features change
            return explainer.shap_values(input_features, nsamples=50)

        with st.spinner("Analyzing feature impact with SHAP..."):
            shap_values = compute_shap_values(features_scaled)

        # ---------- handle multiclass safely ----------
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
        
        fig_shap = px.bar(
            shap_df,
            x="Feature",
            y="Impact",
            color="Impact",
            title=f"Impact on Predicting Class: {labels[pred][0]}",
            color_continuous_scale="viridis"
        )
        fig_shap.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )

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

    # ---------- Global Sales Distribution ----------
    with colA:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Global Sales Distribution")

        fig_hist = px.histogram(
            df_games,
            x="Global_Sales",
            nbins=50,
            color_discrete_sequence=["#3b82f6"]
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Top Publishers ----------
    with colB:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Top Publishers")

        top_pub = (
            df_games.groupby("Publisher")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig_pub = px.bar(
            top_pub,
            x="Publisher",
            y="Global_Sales",
            color="Global_Sales",
            color_continuous_scale="viridis"
        )
        fig_pub.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_pub, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colC, colD = st.columns(2)

    # ---------- Platform Sales ----------
    with colC:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎮 Platform-wise Sales")

        plat_sales = (
            df_games.groupby("Platform")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig_plat = px.bar(
            plat_sales,
            x="Platform",
            y="Global_Sales",
            color="Global_Sales",
            color_continuous_scale="viridis"
        )
        fig_plat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_plat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Correlation Heatmap ----------
    with colD:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Sales Correlation")

        corr = df_games[
            ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
        ].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 6 — GAME RECOMMENDER SYSTEM
# ============================================================
with tab6:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🎮 Video Game Recommender")

    st.caption("Find top games based on platform, genre, and year.")

    # ---------- FILTERS ----------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        platform_list = sorted(df_games["Platform"].dropna().unique())
        selected_platform = st.selectbox(
            "Select Platform",
            ["All"] + platform_list
        )

    with col2:
        genre_list = sorted(df_games["Genre"].dropna().unique())
        selected_genre = st.selectbox(
            "Select Genre",
            ["All"] + genre_list
        )

    with col3:
        year_min = int(df_games["Year"].min())
        year_max = int(df_games["Year"].max())

        year_range = st.slider(
            "Select Year Range",
            year_min,
            year_max,
            (year_min, year_max)
        )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- FILTER DATA ----------
    filtered_df = df_games.copy()

    if selected_platform != "All":
        filtered_df = filtered_df[
            filtered_df["Platform"] == selected_platform
        ]

    if selected_genre != "All":
        filtered_df = filtered_df[
            filtered_df["Genre"] == selected_genre
        ]

    filtered_df = filtered_df[
        (filtered_df["Year"] >= year_range[0]) &
        (filtered_df["Year"] <= year_range[1])
    ]

    # ---------- TOP GAMES ----------
    st.markdown("#### 🏆 Top Recommended Games")

    top_games = (
        filtered_df
        .sort_values("Global_Sales", ascending=False)
        .head(10)
        [["Name", "Platform", "Genre", "Year", "Global_Sales"]]
    )

    if len(top_games) == 0:
        st.warning("No games match the current filters.")
        st.caption("Try widening platform, genre, or year range.")
    else:
        st.dataframe(top_games, use_container_width=True)
        
        # ---------- OPTIONAL CHART ----------
        fig_top = px.bar(
            top_games,
            x="Name",
            y="Global_Sales",
            color="Global_Sales",
            color_continuous_scale="viridis"
        )
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧠 Similar Game Finder (Advanced)")
    if sim_games is None or nn_model is None:
        st.error("Similarity engine failed to initialize.")
        st.stop()

    selected_game = st.selectbox(
        "Select a game to find similar ones",
        sim_games["Name"].values
    )

    try:
        idx = sim_games[sim_games["Name"] == selected_game].index[0]

        distances, indices = nn_model.kneighbors(
            [sim_games.loc[idx, ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]]],
            n_neighbors=11
        )

        game_indices = indices[0][1:]

        similar_games = sim_games.iloc[game_indices][[
            "Name",
            "Platform",
            "Genre",
            "Global_Sales"
        ]]

        st.success("Top similar games:")
        st.dataframe(similar_games, use_container_width=True)

    except Exception as e:
        st.error("Could not compute similar games.")
        st.caption(str(e))

# ============================================================
# TAB 7 — MODEL DIAGNOSTICS
# ============================================================
with tab7:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🧪 Model Diagnostics")

    # ---------- choose best model ----------
    best_model_name = acc_df.sort_values(
        "Accuracy", ascending=False
    ).iloc[0]["Model"]

    model_map = {
        "SVM": svm_cmp,
        "Naive Bayes": nb_cmp,
        "KNN": knn_cmp,
        "Decision Tree": dt_cmp,
        "XGBoost": xgb_cmp
    }

    best_model = model_map[best_model_name]

    st.success(f"Analyzing best model: **{best_model_name}**")

    # =====================================================
    # ROC CURVES (MULTICLASS)
    # =====================================================
    st.markdown("#### 📉 Multiclass ROC Curves")

    try:
        y_test_bin = label_binarize(y_test_cmp, classes=[0, 1, 2])
        y_proba = best_model.predict_proba(X_test_cmp)

        roc_data = []

        for i, class_name in enumerate(["Low", "Medium", "High"]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            temp_df = pd.DataFrame({
                "FPR": fpr,
                "TPR": tpr,
                "Class": class_name
            })
            roc_data.append(temp_df)

        roc_df = pd.concat(roc_data)

        fig_roc = px.line(
            roc_df,
            x="FPR",
            y="TPR",
            color="Class",
            title="ROC Curve (One-vs-Rest)"
        )
        fig_roc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.warning("ROC could not be computed.")

    # =====================================================
    # PRECISION / RECALL TABLE
    # =====================================================
    st.markdown("#### 📊 Precision & Recall")

    y_pred_best = best_model.predict(X_test_cmp)

    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall"],
        "Score": [
            precision_score(y_test_cmp, y_pred_best, average="macro"),
            recall_score(y_test_cmp, y_pred_best, average="macro")
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================
    st.markdown("#### 🧮 Test Set Class Distribution")

    dist_df = y_test_cmp.value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]

    fig_dist = px.bar(
        dist_df,
        x="Class",
        y="Count",
        color="Count",
        color_continuous_scale="viridis"
    )
    fig_dist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # =====================================================
    # MISCLASSIFICATION COUNT
    # =====================================================
    st.markdown("#### 🚨 Misclassification Insight")

    misclassified = (y_test_cmp != y_pred_best).sum()
    total = len(y_test_cmp)

    st.metric(
        "Misclassified Samples",
        f"{misclassified} / {total}",
        delta=f"{(misclassified/total)*100:.2f}% error",
        delta_color="inverse"
    )

# ============================================================
# TAB 8 — WHAT-IF SIMULATOR (FRAGMENTED FOR ZERO LATENCY)
# ============================================================
with tab8:
    @st.fragment
    def what_if_simulator():
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("### 🎛️ What-If Simulator")
        st.caption("Interactively explore how regional sales affect predictions. (Runs independently)")

        # ---------- input sliders ----------
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

        # ---------- prediction ----------
        sim_features = np.array([[sim_na, sim_eu, sim_jp, sim_other]])
        sim_scaled = scaler.transform(sim_features)

        sim_pred = model.predict(sim_scaled)[0]
        sim_proba = model.predict_proba(sim_scaled)[0]

        class_labels = ["Low", "Medium", "High"]

        st.markdown("#### 🔮 Simulated Prediction")
        
        # Dynamic coloring based on prediction
        color_map = {"Low": "var(--accent-red)", "Medium": "var(--accent-orange)", "High": "var(--accent-green)"}
        pred_label = class_labels[sim_pred]
        
        st.markdown(
            f"<h3 style='color: {color_map[pred_label]}; border: 1px solid {color_map[pred_label]}; padding: 16px; border-radius: 12px; text-align: center; background: rgba(17, 25, 40, 0.5);'>"
            f"Predicted Class: {pred_label}</h3>", 
            unsafe_allow_html=True
        )

        # =====================================================
        # DECISION SWEEP (NA axis)
        # =====================================================
        st.markdown("#### 📉 Decision Sensitivity Curve")

        sweep_vals = np.linspace(0, 10, 60)
        
        # Fast batch prediction for the sweep instead of loop
        sweep_features = np.zeros((len(sweep_vals), 4))
        sweep_features[:, 0] = sweep_vals
        sweep_features[:, 1] = sim_eu
        sweep_features[:, 2] = sim_jp
        sweep_features[:, 3] = sim_other
        
        sweep_scaled = scaler.transform(sweep_features)
        sweep_probs = model.predict_proba(sweep_scaled)[:, 2] # Probabilities for High Sales

        sweep_df = pd.DataFrame({
            "NA_Sales": sweep_vals,
            "High_Sales_Prob": sweep_probs
        })

        fig_boundary = px.line(
            sweep_df,
            x="NA_Sales",
            y="High_Sales_Prob",
            title="How NA Sales influences High Sales probability"
        )
        # Optimize chart rendering
        fig_boundary.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

        # =====================================================
        # CLASS PROBABILITIES
        # =====================================================
        st.markdown("#### 📊 Class Probabilities")

        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": sim_proba
        })

        fig_prob_sim = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            color="Class",
            color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"}
        )
        fig_prob_sim.update_layout(
            showlegend=False, 
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff"
        )

        st.plotly_chart(fig_prob_sim, use_container_width=True)

    # Call the fragment function
    what_if_simulator()

# ============================================================
# TAB 9 — DRIFT MONITOR
# ============================================================
with tab9:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 🧭 Model Drift Monitor")
    st.caption("Compare live inputs against training distribution.")

    # ---------- current input vector ----------
    current_input = pd.Series({
        "NA_Sales": na_sales,
        "EU_Sales": eu_sales,
        "JP_Sales": jp_sales,
        "Other_Sales": other_sales
    })

    baseline_mean = drift_baseline["mean"]
    baseline_std = drift_baseline["std"]

    # ---------- z-score drift ----------
    drift_scores = ((current_input - baseline_mean) / (baseline_std + 1e-6)).abs()

    drift_df = pd.DataFrame({
        "Feature": drift_scores.index,
        "Drift Score (|z|)": drift_scores.values
    }).sort_values("Drift Score (|z|)", ascending=False)

    # =====================================================
    # HEALTH STATUS
    # =====================================================
    max_drift = drift_scores.max()

    if max_drift < 1:
        st.success("✅ Input is within normal training range.")
    elif max_drift < 2.5:
        st.warning("⚠️ Moderate drift detected. Monitor recommended.")
    else:
        st.error("🚨 High drift! Model may be unreliable.")

    # =====================================================
    # DRIFT BAR CHART
    # =====================================================
    st.markdown("#### 📊 Feature Drift Scores")

    fig_drift = px.bar(
        drift_df,
        x="Feature",
        y="Drift Score (|z|)",
        color="Drift Score (|z|)",
        color_continuous_scale="Reds"
    )
    fig_drift.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff"
    )

    st.plotly_chart(fig_drift, use_container_width=True)

    # =====================================================
    # RAW COMPARISON TABLE
    # =====================================================
    st.markdown("#### 🔍 Detailed Comparison")

    compare_df = pd.DataFrame({
        "Current": current_input,
        "Training Mean": baseline_mean,
        "Training Std": baseline_std
    })

    st.dataframe(compare_df, use_container_width=True)

# ============================================================
# TAB 10 — ABOUT MODEL (ULTRA PREMIUM + FULL DEPTH)
# ============================================================
with tab10:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("### 📘 About Model — Full Technical Documentation")
    st.caption("In-depth architecture, methodology, and design philosophy of the Video Game Sales Intelligence system.")

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

    # =====================================================
    # SECTION 1 — PROBLEM
    # =====================================================
    with st.expander("🎯 1. Problem Formulation (Detailed)", expanded=True):
        st.markdown("""
The core objective of this system is to estimate the **global commercial performance tier**
of a video game using regional sales signals.

### Why not regression?

Predicting exact revenue in the gaming industry is highly volatile due to:

- blockbuster outliers  
- platform lifecycle effects  
- marketing shocks  
- franchise bias  

Therefore, the task is intentionally framed as a **multi-class classification problem**
with three ordinal tiers:

- 📉 Low Sales  
- 📊 Medium Sales  
- 🚀 High Sales  

This formulation improves:

- robustness to extreme values  
- interpretability for stakeholders  
- stability across model families  
- decision-support usefulness  

Mathematically, the system learns:

f(NA, EU, JP, Other) → {Low, Medium, High}

This transforms noisy revenue prediction into a structured success categorization task.
""")

    # =====================================================
    # SECTION 2 — DATASET
    # =====================================================
    with st.expander("📊 2. Dataset Deep Dive"):
        st.markdown("""
### Dataset Source

The model is trained on the widely used **vgsales dataset**, which aggregates
historical video game performance across regions and platforms.

### Key Characteristics

The dataset contains:

- multi-region sales data  
- platform metadata  
- genre information  
- publisher details  
- global aggregated revenue  

For modeling stability and interpretability, the current system focuses on
**numerical regional sales signals**.

---

### Feature Set Used

Primary predictive inputs:

- **NA_Sales** — North America market performance  
- **EU_Sales** — European market performance  
- **JP_Sales** — Japan market performance  
- **Other_Sales** — Rest-of-world performance  

These features are continuous and moderately correlated, making the task suitable
for both margin-based and tree-based learners.

---

### Target Engineering Strategy

The raw **Global_Sales** variable is transformed using **quantile-based binning**
into three approximately balanced classes.

#### Motivation

Quantile binning helps:

- mitigate class imbalance  
- stabilize multi-class learning  
- reduce skew from blockbuster titles  
- improve cross-model comparability  

Class construction:

- bottom quantile → Low  
- middle quantile → Medium  
- top quantile → High  

This creates a balanced ordinal classification target suitable for robust evaluation.
""")

    # =====================================================
    # SECTION 3 — PREPROCESSING
    # =====================================================
    with st.expander("🧠 3. Data Preprocessing Pipeline"):
        st.markdown("""
The preprocessing pipeline follows a **production-aware architecture** to ensure
train–serve consistency.

### Step 1: Data Cleaning

- removal of rows missing critical sales fields  
- validation of numeric ranges  
- index reset after filtering  

This prevents silent data corruption during training.

---

### Step 2: Stratified Train-Test Split

The dataset is split using:

- stratified sampling  
- fixed random seed  
- preserved class proportions  

**Why stratification matters**

Without stratification:

- minority classes may vanish in test set  
- accuracy becomes misleading  
- model comparison becomes unstable  

---

### Step 3: Feature Standardization

The pipeline applies **StandardScaler normalization**.

#### Why scaling is critical

Required for:

- Support Vector Machines  
- K-Nearest Neighbors  
- distance-based methods  
- gradient-based optimization  

Without scaling:

- SVM margins distort  
- KNN distance becomes biased  
- convergence becomes unstable  

The scaler is persisted via joblib and reused during inference to guarantee
**identical feature space geometry**.
""")

    # =====================================================
    # SECTION 4 — MODELS
    # =====================================================
    with st.expander("🤖 4. Model Architecture & Rationale"):
        st.markdown("""
The system is intentionally designed as a **multi-model benchmarking environment**
to compare different inductive biases.

---

### Gaussian Naive Bayes

Role:

- probabilistic baseline  
- extremely fast training  
- low variance reference  

Assumption:

- conditional independence between features  

Used primarily as a performance floor.

---

### K-Nearest Neighbors

Role:

- distance-based learner  
- non-parametric model  

Strengths:

- captures local structure  
- simple decision boundary  

Limitations:

- sensitive to scaling  
- slower inference  

---

### Decision Tree

Role:

- interpretable nonlinear model  
- rule-based structure  

Advantages:

- human-readable splits  
- captures feature interactions  
- handles nonlinearity  

Regularized using max_depth to control overfitting.

---

### Support Vector Machine (Primary Model)

The SVM serves as the **primary deployed classifier**.

#### Why SVM?

- strong performance on normalized tabular data  
- robust margin maximization  
- stable decision boundaries  
- effective in medium-dimensional space  

Configuration:

- RBF kernel  
- probability enabled  
- standardized inputs  

---

### XGBoost

Role:

- gradient boosting benchmark  
- high-capacity nonlinear learner  

Strengths:

- captures complex feature interactions  
- strong tabular performance  
- ensemble robustness  

Used to benchmark against margin-based methods.
""")

    # =====================================================
    # SECTION 5 — EVALUATION
    # =====================================================
    with st.expander("🧪 5. Evaluation Methodology"):
        st.markdown("""
The system avoids reliance on a single metric and instead uses a
**multi-angle evaluation framework**.

### Metrics Used

- hold-out accuracy  
- stratified 5-fold cross-validation  
- confusion matrices  
- multiclass ROC (OvR)  
- macro precision & recall  
- training time benchmarking  

---

### Why Cross-Validation?

Single train-test splits can be misleading due to sampling variance.

Stratified 5-fold CV provides:

- more stable generalization estimate  
- reduced split bias  
- improved model comparison credibility  

Mean and standard deviation are both reported to capture stability.
""")

    # =====================================================
    # SECTION 6 — CALIBRATION
    # =====================================================
    with st.expander("📉 6. Probability Calibration"):
        st.markdown("""
Raw classifier probabilities are often poorly calibrated, especially for
margin-based models like SVM.

The system supports **Platt scaling via CalibratedClassifierCV**.

### Benefits

- improves probability reliability  
- reduces overconfidence  
- better threshold decisions  
- more trustworthy risk interpretation  

Users can toggle calibrated inference from the sidebar.
""")

    # =====================================================
    # SECTION 7 — EXPLAINABILITY
    # =====================================================
    with st.expander("🔍 7. Explainability Framework"):
        st.markdown("""
Model transparency is implemented through multiple complementary methods.

**Included tools:**

- permutation feature importance  
- SHAP local explanations  
- probability breakdown visualization  
- input sensitivity sweeps  

These mechanisms help answer the critical question:

> Why did the model predict this outcome?
""")

    # =====================================================
    # SECTION 8 — DRIFT
    # =====================================================
    with st.expander("🧭 8. Drift Monitoring Strategy"):
        st.markdown("""
A lightweight drift detector compares live user inputs against the
training distribution using standardized z-score distance.

### Purpose

- detect out-of-distribution queries  
- warn about reliability degradation  
- simulate real production monitoring  

This introduces **MLOps awareness** into the dashboard.
""")

    # =====================================================
    # SECTION 9 — LIMITATIONS
    # =====================================================
    with st.expander("⚠️ 9. Known Limitations"):
        st.markdown("""
No responsible ML system is complete without acknowledging constraints.

Key limitations:

- historical dataset may not reflect modern market dynamics  
- regional sales are correlated  
- quantile binning simplifies revenue distribution  
- temporal validation not yet enforced  
- marketing and release timing not modeled  

Therefore, predictions should be interpreted as **indicative classification signals**,
not precise financial forecasts.
""")

    # =====================================================
    # SECTION 10 — ROADMAP
    # =====================================================
    with st.expander("🚀 10. Future Technical Roadmap"):
        st.markdown("""
Planned production-grade enhancements:

- time-aware cross-validation  
- richer categorical encoding  
- automated drift-triggered retraining  
- expanded dataset coverage  
- reliability diagram visualization  
- real-time inference telemetry  
- model version registry  

The long-term goal is to evolve this system into a fully automated
intelligent analytics pipeline.
""")

    st.markdown("---")
    st.caption("Model Version: v1.0 • Production ML System • Built by HKS")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by HKS • Machine Learning • UI/UX")