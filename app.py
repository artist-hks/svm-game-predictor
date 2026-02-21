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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Intelligence OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SESSION STATE & THEME ----------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "best_acc_pct" not in st.session_state:
    st.session_state.best_acc_pct = "—"
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# ---------- DYNAMIC THEME VARIABLES ----------
theme = st.session_state.theme
if theme == "light":
    bg_color = "#f4f7fe"
    card_bg = "#ffffff"
    text_main = "#1e293b"
    text_muted = "#64748b"
    border_color = "#e2e8f0"
    shadow = "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)"
    accent_primary = "#6366f1"
else:
    bg_color = "#0b1220"
    card_bg = "#111928"
    text_main = "#f8fafc"
    text_muted = "#9aa0a6"
    border_color = "rgba(255,255,255,0.08)"
    shadow = "none"
    accent_primary = "#818cf8"

# ---------------- PREMIUM GLOBAL CSS ----------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_main};
}}
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}}
.saas-card {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 16px;
    padding: 24px;
    box-shadow: {shadow};
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.saas-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}}
h1, h2, h3, h4, h5, h6, p, span, div {{
    color: {text_main};
    font-family: 'Inter', sans-serif;
}}
.text-muted {{
    color: {text_muted} !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;
    background-color: transparent;
    border-bottom: 1px solid {border_color};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border: none;
    padding: 12px 4px;
    color: {text_muted};
    font-weight: 500;
    font-size: 15px;
}}
.stTabs [aria-selected="true"] {{
    color: {accent_primary} !important;
    border-bottom: 3px solid {accent_primary} !important;
    background-color: transparent !important;
}}
[data-testid="stSidebar"] {{
    background-color: {card_bg};
    border-right: 1px solid {border_color};
}}
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {border_color};
}}
.theme-btn-container {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    height: 100%;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER COMPONENTS ----------------
def saas_metric_card(title, value, trend, trend_color, icon, icon_color):
    return f"""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="color: {text_muted}; font-size: 14px; font-weight: 500; margin-bottom: 8px;">{title}</div>
                <div style="font-size: 28px; font-weight: 700; color: {text_main}; line-height: 1.2;">{value}</div>
            </div>
            <div style="width: 40px; height: 40px; border-radius: 10px; background: {icon_color}15; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                {icon}
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-top: 16px; font-size: 13px; font-weight: 600; color: {trend_color};">
            <span style="background: {trend_color}15; padding: 2px 6px; border-radius: 4px; margin-right: 8px;">{trend}</span>
            <span style="color: {text_muted}; font-weight: 400;">vs last period</span>
        </div>
    </div>
    """

def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=text_main,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor=border_color, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=border_color, zeroline=False)
    )
    return fig

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    base_model = joblib.load("svm_model.pkl")
    calibrated_model = joblib.load("calibrated_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return base_model, calibrated_model, scaler

model, calibrated_model, scaler = load_assets()

@st.cache_data
def load_dataset():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna(subset=["Global_Sales"])
    return df

df_games = load_dataset()

@st.cache_data
def compute_training_baseline(df):
    base = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna()
    return {"mean": base.mean(), "std": base.std()}

drift_baseline = compute_training_baseline(df_games)

# --- SPEED OPTIMIZATION: Subsample for Benchmark ---
@st.cache_data
def prepare_model_data(df):
    temp = df.dropna(subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]).copy()
    temp["Sales_Class"] = pd.qcut(temp["Global_Sales"], q=3, labels=[0, 1, 2])
    
    # Subsample to 3000 rows to make SVM training and CV instant
    if len(temp) > 3000:
        temp = temp.sample(3000, random_state=42)
        
    X = temp[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = temp["Sales_Class"].astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = prepare_model_data(df_games)

@st.cache_resource
def compute_cv_scores(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", use_label_encoder=False, random_state=42, n_jobs=-1)
    }
    results = {}
    for name, m in models.items():
        # n_jobs=-1 uses all CPU cores for blazing fast cross-validation
        scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
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
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1).fit(X_train, y_train)
    timings["KNN"] = time.perf_counter() - start

    start = time.perf_counter()
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    timings["Decision Tree"] = time.perf_counter() - start

    start = time.perf_counter()
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", random_state=42, n_jobs=-1).fit(X_train, y_train)
    timings["XGBoost"] = time.perf_counter() - start

    return svm, nb, knn, dt, xgb, timings

@st.cache_resource
def prepare_similarity_engine(df):
    try:
        sim_df = df[["Name", "Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].dropna().reset_index(drop=True)
        features = sim_df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
        nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        nn_model.fit(features)
        return sim_df, nn_model
    except Exception:
        return None, None

sim_games, nn_model = prepare_similarity_engine(df_games)

# ---------------- TOP NAVIGATION BAR ----------------
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
        <div style="width: 48px; height: 48px; border-radius: 12px; background: {accent_primary}; display: flex; align-items: center; justify-content: center; font-size: 24px;">
            🎮
        </div>
        <div>
            <h2 style="margin: 0; font-size: 24px; font-weight: 700;">Welcome Back, Analyst</h2>
            <p style="margin: 0; color: {text_muted}; font-size: 14px;">Analyze your Data & Predictions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="theme-btn-container">', unsafe_allow_html=True)
    theme_icon = "🌙 Dark Mode" if theme == "light" else "☀️ Light Mode"
    st.button(theme_icon, on_click=toggle_theme, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(f"""
<div style="padding: 10px 0 20px 0;">
    <h3 style="margin:0; font-size: 18px;">⚙️ Control Panel</h3>
    <p style="margin:0; color: {text_muted}; font-size: 13px;">Adjust inputs & explore</p>
</div>
""", unsafe_allow_html=True)

na_sales = st.sidebar.slider("🇺🇸 NA Sales (Millions)", 0.0, 10.0, 0.5, 0.1)
eu_sales = st.sidebar.slider("🇪🇺 EU Sales (Millions)", 0.0, 10.0, 0.3, 0.1)
jp_sales = st.sidebar.slider("🇯🇵 JP Sales (Millions)", 0.0, 10.0, 0.1, 0.1)
other_sales = st.sidebar.slider("🌍 Other Sales (Millions)", 0.0, 10.0, 0.05, 0.05)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
use_calibrated = st.sidebar.toggle("Use calibrated probabilities", value=True)

features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)

st.sidebar.markdown("---")
st.sidebar.caption(f"Session predictions: {st.session_state.prediction_count}")
st.sidebar.caption(f"Last Latency: {st.session_state.last_latency:.1f} ms")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Dashboard", "Feature Impact", "Models", "SHAP", "Analytics", 
    "Recommender", "Diagnostics", "Simulator", "Drift", "Docs"
])

# ============================================================
# TAB 1 — DASHBOARD (PREDICTION)
# ============================================================
with tab1:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(saas_metric_card("NA Sales", f"${na_sales:.2f}M", "▲ 12%", "#22c55e", "🇺🇸", "#6366f1"), unsafe_allow_html=True)
    with m2: st.markdown(saas_metric_card("EU Sales", f"${eu_sales:.2f}M", "▲ 8%", "#22c55e", "🇪🇺", "#f59e0b"), unsafe_allow_html=True)
    with m3: st.markdown(saas_metric_card("JP Sales", f"${jp_sales:.2f}M", "▼ 3%", "#ef4444", "🇯🇵", "#ec4899"), unsafe_allow_html=True)
    with m4: st.markdown(saas_metric_card("Other Sales", f"${other_sales:.2f}M", "▲ 5%", "#22c55e", "🌍", "#14b8a6"), unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    c_main, c_side = st.columns([2, 1])
    
    with c_main:
        st.markdown(f'<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin-top:0;'>Sales Distribution Overview</h4>", unsafe_allow_html=True)
        
        input_df = pd.DataFrame({
            "Region": ["North America", "Europe", "Japan", "Other"],
            "Sales (Millions)": [na_sales, eu_sales, jp_sales, other_sales]
        })
        fig_input = px.bar(input_df, x="Region", y="Sales (Millions)", color="Region", 
                           color_discrete_sequence=["#6366f1", "#f59e0b", "#ec4899", "#14b8a6"])
        fig_input = apply_chart_theme(fig_input)
        fig_input.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_input, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_side:
        st.markdown(f'<div class="saas-card" style="display: flex; flex-direction: column; justify-content: center; text-align: center;">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin-top:0; color: {text_muted};'>AI Prediction Result</h4>", unsafe_allow_html=True)

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
            0: ("Low Sales Tier", "#ef4444", "📉"),
            1: ("Medium Sales Tier", "#f59e0b", "📊"),
            2: ("High Sales Tier", "#22c55e", "🚀")
        }
        text, color, icon = labels.get(pred, ("Unknown", text_muted, "❓"))

        st.markdown(f"""
            <div style="margin: 20px 0;">
                <div style="font-size: 64px; margin-bottom: 10px;">{icon}</div>
                <h2 style="color: {color}; font-size: 28px; margin: 0;">{text}</h2>
                <div style="font-size: 48px; font-weight: 800; color: {text_main}; margin-top: 16px; line-height: 1;">
                    {confidence:.1f}%
                </div>
                <div style="color: {text_muted}; font-size: 14px; margin-top: 8px; font-weight: 500;">
                    Confidence Score
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — FEATURE IMPORTANCE
# ============================================================
with tab2:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Feature Importance (Permutation)")
    st.caption("Permutation importance approximates feature impact for SVM.")

    try:
        sample_df = df_games[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].dropna().sample(300, random_state=42)
        X_sample = scaler.transform(sample_df)
        y_sample = model.predict(X_sample)

        perm = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
        importance_df = pd.DataFrame({
            "Feature": ["NA", "EU", "JP", "Other"],
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(importance_df, x="Feature", y="Importance", color="Importance", color_continuous_scale="Purples")
        fig_imp = apply_chart_theme(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning("Feature importance could not be computed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3 — MODEL COMPARISON
# ============================================================
with tab3:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
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
        <div class="saas-card" style="border-left: 4px solid #22c55e;">
            <div style="color: {text_muted}; font-size: 14px; text-transform: uppercase; font-weight: 600;">🏆 Most Accurate Model</div>
            <div style="font-size: 28px; font-weight: 700; margin-top: 8px; color: {text_main};">{best_acc_model}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid #f59e0b;">
            <div style="color: {text_muted}; font-size: 14px; text-transform: uppercase; font-weight: 600;">⚡ Fastest Training Model</div>
            <div style="font-size: 28px; font-weight: 700; margin-top: 8px; color: {text_main};">{fastest_model}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Accuracy Comparison")
        fig_acc = px.bar(acc_df, x="Model", y="Accuracy", color="Accuracy", color_continuous_scale="Purples")
        fig_acc = apply_chart_theme(fig_acc)
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with colB:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### ⏱️ Training Time Comparison")
        fig_time = px.bar(acc_df, x="Model", y="Training Time (s)", color="Training Time (s)", color_continuous_scale="Oranges")
        fig_time = apply_chart_theme(fig_time)
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 4 — SHAP EXPLAINABILITY
# ============================================================
with tab4:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 SHAP Explainability")
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

        fig_shap = px.bar(shap_df, x="Feature", y="Impact", color="Impact", title=f"Impact on Predicting Class: {labels[pred][0]}", color_continuous_scale="Teal")
        fig_shap = apply_chart_theme(fig_shap)
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.error("SHAP failed to compute.")
        st.caption(f"Error details: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 5 — ANALYTICS DASHBOARD
# ============================================================
with tab5:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Global Sales Distribution")
        fig_hist = px.histogram(df_games, x="Global_Sales", nbins=50, color_discrete_sequence=[accent_primary])
        fig_hist = apply_chart_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Top Publishers")
        top_pub = df_games.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_pub = px.bar(top_pub, x="Publisher", y="Global_Sales", color="Global_Sales", color_continuous_scale="Purples")
        fig_pub = apply_chart_theme(fig_pub)
        st.plotly_chart(fig_pub, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    colC, colD = st.columns(2)

    with colC:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🎮 Platform-wise Sales")
        plat_sales = df_games.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_plat = px.bar(plat_sales, x="Platform", y="Global_Sales", color="Global_Sales", color_continuous_scale="Purples")
        fig_plat = apply_chart_theme(fig_plat)
        st.plotly_chart(fig_plat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colD:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Sales Correlation")
        corr = df_games[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr = apply_chart_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 6 — GAME RECOMMENDER SYSTEM
# ============================================================
with tab6:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 🎮 Video Game Recommender")
    st.caption("Find top games based on platform, genre, and year.")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_platform = st.selectbox("Select Platform", ["All"] + sorted(df_games["Platform"].dropna().unique()))
    with col2:
        selected_genre = st.selectbox("Select Genre", ["All"] + sorted(df_games["Genre"].dropna().unique()))
    with col3:
        year_min, year_max = int(df_games["Year"].min()), int(df_games["Year"].max())
        year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

    filtered_df = df_games.copy()
    if selected_platform != "All": filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]
    if selected_genre != "All": filtered_df = filtered_df[filtered_df["Genre"] == selected_genre]
    filtered_df = filtered_df[(filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])]

    st.markdown("<br><h5>🏆 Top Recommended Games</h5>", unsafe_allow_html=True)
    top_games = filtered_df.sort_values("Global_Sales", ascending=False).head(10)[["Name", "Platform", "Genre", "Year", "Global_Sales"]]

    if len(top_games) == 0:
        st.warning("No games match the current filters.")
    else:
        st.dataframe(top_games, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 7 — MODEL DIAGNOSTICS
# ============================================================
with tab7:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Model Diagnostics")

    best_model_name = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    model_map = {"SVM": svm_cmp, "Naive Bayes": nb_cmp, "KNN": knn_cmp, "Decision Tree": dt_cmp, "XGBoost": xgb_cmp}
    best_model = model_map[best_model_name]

    st.success(f"Analyzing best model: **{best_model_name}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📉 Multiclass ROC Curves")
        try:
            y_test_bin = label_binarize(y_test_cmp, classes=[0, 1, 2])
            y_proba = best_model.predict_proba(X_test_cmp)
            roc_data = []
            for i, class_name in enumerate(["Low", "Medium", "High"]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_data.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Class": class_name}))
            
            fig_roc = px.line(pd.concat(roc_data), x="FPR", y="TPR", color="Class")
            fig_roc = apply_chart_theme(fig_roc)
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.warning("ROC could not be computed.")

    with c2:
        st.markdown("##### 🧮 Test Set Class Distribution")
        dist_df = y_test_cmp.value_counts().reset_index()
        dist_df.columns = ["Class", "Count"]
        fig_dist = px.bar(dist_df, x="Class", y="Count", color="Count", color_continuous_scale="Purples")
        fig_dist = apply_chart_theme(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 8 — WHAT-IF SIMULATOR
# ============================================================
with tab8:
    @st.fragment
    def what_if_simulator():
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 🎛️ What-If Simulator")
        st.caption("Interactively explore how regional sales affect predictions.")

        c1, c2 = st.columns(2)
        with c1:
            sim_na = st.slider("NA Sales (Sim)", 0.0, 10.0, 1.0, 0.1, key="sim_na")
            sim_eu = st.slider("EU Sales (Sim)", 0.0, 10.0, 1.0, 0.1, key="sim_eu")
        with c2:
            sim_jp = st.slider("JP Sales (Sim)", 0.0, 10.0, 0.5, 0.1, key="sim_jp")
            sim_other = st.slider("Other Sales (Sim)", 0.0, 10.0, 0.3, 0.1, key="sim_other")

        sim_features = np.array([[sim_na, sim_eu, sim_jp, sim_other]])
        sim_scaled = scaler.transform(sim_features)
        sim_pred = model.predict(sim_scaled)[0]
        sim_proba = model.predict_proba(sim_scaled)[0]
        class_labels = ["Low", "Medium", "High"]

        st.markdown("##### 📉 Decision Sensitivity Curve")
        sweep_vals = np.linspace(0, 10, 60)
        sweep_features = np.zeros((len(sweep_vals), 4))
        sweep_features[:, 0] = sweep_vals
        sweep_features[:, 1] = sim_eu
        sweep_features[:, 2] = sim_jp
        sweep_features[:, 3] = sim_other
        
        sweep_scaled = scaler.transform(sweep_features)
        sweep_probs = model.predict_proba(sweep_scaled)[:, 2] 

        fig_boundary = px.line(pd.DataFrame({"NA_Sales": sweep_vals, "High_Sales_Prob": sweep_probs}), x="NA_Sales", y="High_Sales_Prob")
        fig_boundary = apply_chart_theme(fig_boundary)
        st.plotly_chart(fig_boundary, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    what_if_simulator()

# ============================================================
# TAB 9 — DRIFT MONITOR
# ============================================================
with tab9:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 🧭 Model Drift Monitor")
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

    fig_drift = px.bar(drift_df, x="Feature", y="Drift Score (|z|)", color="Drift Score (|z|)", color_continuous_scale="Reds")
    fig_drift = apply_chart_theme(fig_drift)
    st.plotly_chart(fig_drift, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 10 — ABOUT MODEL
# ============================================================
with tab10:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### 📘 About Model — Full Technical Documentation")
    
    with st.expander("🎯 1. Problem Formulation (Detailed)", expanded=True):
        st.markdown("The core objective is to estimate the **global commercial performance tier** of a video game as a **multi-class classification problem** (Low, Medium, High).")

    with st.expander("📊 2. Dataset Deep Dive"):
        st.markdown("Trained on the **vgsales dataset**. Features used: NA_Sales, EU_Sales, JP_Sales, Other_Sales.")

    with st.expander("🧠 3. Data Preprocessing Pipeline"):
        st.markdown("- **Cleaning**: Removal of missing rows.\n- **Splitting**: Stratified train-test split.\n- **Scaling**: StandardScaler normalization.")

    with st.expander("🤖 4. Model Architecture & Rationale"):
        st.markdown("- **SVM (Primary)**: Strong performance on normalized tabular data.\n- **XGBoost**: Gradient boosting benchmark.")
    st.markdown('</div>', unsafe_allow_html=True)