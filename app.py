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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE CONFIG & SESSION ----------------
st.set_page_config(page_title="VG-Ops Intelligence", page_icon="🎮", layout="wide")

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

# ---------------- PREMIUM GLOBAL CSS ----------------
st.markdown("""
<style>
:root { --bg-main: #0b1220; --card-bg: rgba(17, 25, 40, 0.75); }
.hero-container {
    position: relative; padding: 30px 34px; border-radius: 22px;
    background: linear-gradient(135deg, #020617 0%, #0f172a 60%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08); overflow: hidden; margin-bottom: 1.4rem;
}
.hero-container::before {
    content: ""; position: absolute; inset: -2px;
    background: linear-gradient(120deg, rgba(34,197,94,0.25), rgba(59,130,246,0.25), rgba(168,85,247,0.25), rgba(34,197,94,0.25));
    filter: blur(38px); opacity: 0.55; animation: heroGlow 8s linear infinite; z-index: 0;
}
@keyframes heroGlow { 0% { transform: rotate(0deg) scale(1); } 50% { transform: rotate(180deg) scale(1.05); } 100% { transform: rotate(360deg) scale(1); } }
.hero-title { font-size: 42px; font-weight: 800; margin-bottom: 6px; position: relative; z-index: 2; }
.hero-subtitle { color: #9aa0a6; font-size: 15px; margin-bottom: 14px; position: relative; z-index: 2; }
.glass-nav {
    display: flex; justify-content: space-between; align-items: center; padding: 14px 18px;
    border-radius: 14px; background: rgba(17, 25, 40, 0.55); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;
}
.status-chip { padding: 5px 10px; border-radius: 999px; font-size: 12px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); margin-left: 6px; }
.glass-card { padding: 22px; border-radius: 16px; background: var(--card-bg); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); }
</style>
""", unsafe_allow_html=True)

# ---------------- CACHED DATA & MODELS ----------------
@st.cache_resource
def load_assets():
    base_model = joblib.load("svm_model.pkl")
    calibrated_model = joblib.load("calibrated_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return base_model, calibrated_model, scaler

model, calibrated_model, scaler = load_assets()

@st.cache_data
def load_dataset():
    return pd.read_csv("vgsales.csv").dropna(subset=["Global_Sales"])

df_games = load_dataset()

@st.cache_data
def compute_training_baseline(df):
    base = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna()
    return {"mean": base.mean(), "std": base.std()}

drift_baseline = compute_training_baseline(df_games)

@st.cache_data
def prepare_model_data(df):
    temp = df.dropna(subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]).copy()
    temp["Sales_Class"] = pd.qcut(temp["Global_Sales"], q=3, labels=[0, 1, 2])
    X = temp[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = temp["Sales_Class"].astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = prepare_model_data(df_games)

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
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42).fit(X_train, y_train)
    timings["XGBoost"] = time.perf_counter() - start

    return svm, nb, knn, dt, xgb, timings

from sklearn.svm import SVC
svm_cmp, nb_cmp, knn_cmp, dt_cmp, xgb_cmp, timings = train_comparison_models(X_train_cmp, y_train_cmp)

# Global Accuracy calculation for Header
best_acc_pct = f"{accuracy_score(y_test_cmp, svm_cmp.predict(X_test_cmp))*100:.2f}%"

@st.cache_resource
def prepare_similarity_engine(df):
    try:
        sim_df = df[["Name", "Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].dropna().reset_index(drop=True)
        features = sim_df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
        nn_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(features)
        return sim_df, nn_model
    except: return None, None

sim_games, nn_model = prepare_similarity_engine(df_games)

labels = {0: ("📉 Low Sales", "#ef4444"), 1: ("📊 Medium Sales", "#f59e0b"), 2: ("🚀 High Sales", "#22c55e")}

# ---------------- SIDEBAR (MANAGEMENT MENU & INPUTS) ----------------
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h2 style='color: #22c55e; margin-bottom: 0;'>🎮 VG-Ops</h2>
    <p style='color: #9aa0a6; font-size: 13px;'>Enterprise Dashboard v2.0</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "MAIN NAVIGATION",
    ["🏠 Command Center", "🔮 Prediction Studio", "⚙️ MLOps & Diagnostics", "📖 System Docs"]
)

st.sidebar.divider()
st.sidebar.markdown("### 📝 Live Input Data")
st.sidebar.caption("Adjust inputs to see real-time changes across the system.")

na_sales = st.sidebar.number_input("🇺🇸 NA Sales (M)", 0.0, 50.0, 0.5, 0.1)
eu_sales = st.sidebar.number_input("🇪🇺 EU Sales (M)", 0.0, 50.0, 0.3, 0.1)
jp_sales = st.sidebar.number_input("🇯🇵 JP Sales (M)", 0.0, 50.0, 0.1, 0.1)
other_sales = st.sidebar.number_input("🌍 Other Sales (M)", 0.0, 50.0, 0.05, 0.05)

use_calibrated = st.sidebar.toggle("Use Calibrated Probs", value=True)

# Global Inference
features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)

start_inf = time.perf_counter()
active_model = calibrated_model if use_calibrated else model
pred = active_model.predict(features_scaled)[0]
proba = active_model.predict_proba(features_scaled)[0]
latency_ms = (time.perf_counter() - start_inf) * 1000

if st.sidebar.button("Run Prediction"):
    st.session_state.prediction_count += 1
    st.toast("Inference updated globally!", icon="✅")

st.sidebar.divider()
st.sidebar.caption(f"Session Queries: {st.session_state.prediction_count}")
st.sidebar.caption("System Status: 🟢 Online")

# ---------------- GLASS NAVBAR ----------------
st.markdown(f"""
<div class="glass-nav">
    <div style="font-weight: 600;">⚡ VG-Ops Global Server</div>
    <div>
        <span class="status-chip">Best Acc: {best_acc_pct}</span>
        <span class="status-chip">Latency: {latency_ms:.1f} ms</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: COMMAND CENTER
# ============================================================
if menu == "🏠 Command Center":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Video Game Sales Intelligence</div>
        <div class="hero-subtitle">Production-grade ML dashboard for global sales forecasting, explainability, and interactive analytics.</div>
        <div style="display: flex; gap: 8px;">
            <span class="status-chip" style="margin-left:0;">SVM Primary</span>
            <span class="status-chip">XGBoost Benchmark</span>
            <span class="status-chip">SHAP Enabled</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Market Analytics Overview")
    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(df_games, x="Global_Sales", nbins=50, title="🌍 Global Sales Distribution", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        top_pub = df_games.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_pub = px.bar(top_pub, x="Publisher", y="Global_Sales", title="🏢 Top Publishers", template="plotly_dark", color="Global_Sales")
        st.plotly_chart(fig_pub, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        plat_sales = df_games.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_plat = px.bar(plat_sales, x="Platform", y="Global_Sales", title="🎮 Top Platforms", template="plotly_dark", color="Global_Sales")
        st.plotly_chart(fig_plat, use_container_width=True)
    with colD:
        corr = df_games[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="🔥 Sales Correlation Matrix", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

# ============================================================
# PAGE 2: PREDICTION STUDIO
# ============================================================
elif menu == "🔮 Prediction Studio":
    st.markdown("## 🔮 Prediction & Simulation Studio")
    
    text, color = labels[pred]
    confidence = float(np.max(proba) * 100)

    st.markdown(f"""
    <div style="padding:25px; border-radius:14px; background:#111827; border:1px solid #2d3748; text-align:center; margin-bottom: 20px;">
        <h2 style="color:{color}; margin-bottom:0;">{text}</h2>
        <p style="color:#9aa0a6; font-size:18px;">Model Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("#### 🎯 Confidence Meter")
        gauge_fig = {
            "data": [{"type": "indicator", "mode": "gauge+number", "value": confidence, "number": {"suffix": "%"},
                      "gauge": {"axis": {"range": [0, 100]}, "bar": {"color": color}, "bgcolor": "#111827",
                                "steps": [{"range": [0, 40], "color": "#7f1d1d"}, {"range": [40, 70], "color": "#78350f"}, {"range": [70, 100], "color": "#052e16"}]}}]
        }
        st.plotly_chart(gauge_fig, use_container_width=True)
    with c2:
        st.markdown("#### 📊 Probability Breakdown")
        prob_df = pd.DataFrame({"Class": ["Low", "Medium", "High"], "Probability": proba})
        fig_prob = px.bar(prob_df, x="Class", y="Probability", color="Class", text="Probability", template="plotly_dark", 
                          color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"})
        fig_prob.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_prob.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # Fragmented What-If Simulator
    @st.fragment
    def what_if_simulator():
        st.markdown("### 🎛️ Rapid What-If Simulator")
        st.caption("Test independent scenarios without reloading the global app.")
        wc1, wc2, wc3, wc4 = st.columns(4)
        sim_na = wc1.slider("🇺🇸 NA", 0.0, 10.0, float(na_sales), 0.1, key="s1")
        sim_eu = wc2.slider("🇪🇺 EU", 0.0, 10.0, float(eu_sales), 0.1, key="s2")
        sim_jp = wc3.slider("🇯🇵 JP", 0.0, 10.0, float(jp_sales), 0.1, key="s3")
        sim_oth = wc4.slider("🌍 Other", 0.0, 10.0, float(other_sales), 0.1, key="s4")

        sim_scaled = scaler.transform([[sim_na, sim_eu, sim_jp, sim_oth]])
        sim_pred = active_model.predict(sim_scaled)[0]
        
        c_map = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}
        st.markdown(f"<div style='border-left: 5px solid {c_map[sim_pred]}; padding-left: 15px; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'><b>Simulated Result:</b> {labels[sim_pred][0]}</div>", unsafe_allow_html=True)
    
    what_if_simulator()

    st.divider()
    st.markdown("### 🧠 AI Game Recommender")
    st.caption("Finding similar games based on current regional input pattern.")
    try:
        dist, indices = nn_model.kneighbors(features, n_neighbors=6)
        similar_games = sim_games.iloc[indices[0][1:]][["Name", "Platform", "Genre", "Global_Sales"]]
        st.dataframe(similar_games, use_container_width=True)
    except: st.error("Recommender offline.")

# ============================================================
# PAGE 3: MLOps & DIAGNOSTICS
# ============================================================
elif menu == "⚙️ MLOps & Diagnostics":
    st.markdown("## ⚙️ Model Diagnostics & MLOps")
    
    tabA, tabB, tabC = st.tabs(["🧭 Data Drift", "🧠 SHAP Explainability", "📊 Model Comparison"])

    with tabA:
        st.markdown("### Live Drift Monitor")
        current_input = pd.Series({"NA_Sales": na_sales, "EU_Sales": eu_sales, "JP_Sales": jp_sales, "Other_Sales": other_sales})
        drift_scores = ((current_input - drift_baseline["mean"]) / (drift_baseline["std"] + 1e-6)).abs()
        max_drift = drift_scores.max()

        if max_drift < 1: st.success("✅ Input is within normal training range.")
        elif max_drift < 2.5: st.warning("⚠️ Moderate drift detected. Monitor recommended.")
        else: st.error("🚨 High drift! Model may be unreliable for this input.")

        fig_drift = px.bar(x=drift_scores.index, y=drift_scores.values, labels={'x': 'Features', 'y': 'Drift Score (|z|)'}, template="plotly_dark", title="Z-Score Deviation from Training Data")
        st.plotly_chart(fig_drift, use_container_width=True)

    with tabB:
        st.markdown("### SHAP Feature Impact")
        @st.cache_data
        def get_shap_background(df): return scaler.transform(df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna().sample(50, random_state=42))
        @st.cache_resource
        def get_explainer(_bg): return shap.KernelExplainer(model.predict_proba, _bg)
        @st.cache_data
        def compute_shap(_features): return get_explainer(get_shap_background(df_games)).shap_values(_features, nsamples=50)

        with st.spinner("Calculating SHAP values..."):
            shap_vals = compute_shap(features_scaled)
            shap_for_class = shap_vals[int(pred)][0] if isinstance(shap_vals, list) else shap_vals[0]
            
            shap_df = pd.DataFrame({"Feature": ["NA", "EU", "JP", "Other"], "Impact": np.abs(np.array(shap_for_class).flatten())}).sort_values("Impact", ascending=False)
            fig_shap = px.bar(shap_df, x="Feature", y="Impact", color="Impact", template="plotly_dark")
            st.plotly_chart(fig_shap, use_container_width=True)

    with tabC:
        st.markdown("### Cross-Model Benchmarking")
        acc_df = pd.DataFrame([
            {"Model": "SVM", "Accuracy": accuracy_score(y_test_cmp, svm_cmp.predict(X_test_cmp)), "Time(s)": timings["SVM"]},
            {"Model": "XGBoost", "Accuracy": accuracy_score(y_test_cmp, xgb_cmp.predict(X_test_cmp)), "Time(s)": timings["XGBoost"]},
            {"Model": "Naive Bayes", "Accuracy": accuracy_score(y_test_cmp, nb_cmp.predict(X_test_cmp)), "Time(s)": timings["Naive Bayes"]},
        ])
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(acc_df, x="Model", y="Accuracy", color="Accuracy", template="plotly_dark", title="Accuracy Comparison"), use_container_width=True)
        c2.plotly_chart(px.bar(acc_df, x="Model", y="Time(s)", color="Time(s)", template="plotly_dark", title="Training Latency"), use_container_width=True)

# ============================================================
# PAGE 4: SYSTEM DOCS
# ============================================================
elif menu == "📖 System Docs":
    st.markdown("## 📖 Architecture & Documentation")
    
    with st.expander("🎯 1. Problem Formulation", expanded=True):
        st.write("Predicting exact revenue is highly volatile due to blockbuster outliers. This system frames the task as a Multi-class Classification problem (Low, Medium, High Sales) to provide robust, interpretable decision-support intelligence.")
    with st.expander("🤖 2. Model Architecture"):
        st.write("**Primary Model:** Support Vector Machine (RBF Kernel)\n**Benchmark:** XGBoost, Decision Tree, KNN, Naive Bayes\n**Calibration:** Platt Scaling via CalibratedClassifierCV.")
    with st.expander("🛡️ 3. MLOps Features"):
        st.write("Includes Live Drift Monitoring (Z-Score), SHAP Explainability, Cached inference paths, and separated training/serving scripts for zero-latency loading.")
    
    st.divider()
    st.caption("Developed by HKS | Enterprise Grade Dashboard | Built with Streamlit")