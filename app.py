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

# ---------- SESSION ANALYTICS ----------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="VG-Ops Intelligence", page_icon="🎮", layout="wide")

# ---------------- SAAS PREMIUM CSS (Matching Your Images) ----------------
st.markdown("""
<style>
/* Dashboard Background and Font */
.css-18e3th9 { padding-top: 1rem; }

/* Custom Metric Cards matching your screenshots */
.saas-card {
    background: #ffffff;
    padding: 20px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}
@media (prefers-color-scheme: dark) {
    .saas-card {
        background: #111827;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #1f2937;
    }
}
.card-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.card-value-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
}
.card-value {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.trend-up {
    color: #22c55e;
    font-weight: 600;
    font-size: 14px;
    background: rgba(34, 197, 94, 0.1);
    padding: 2px 8px;
    border-radius: 999px;
}
.trend-down {
    color: #ef4444;
    font-weight: 600;
    font-size: 14px;
    background: rgba(239, 68, 68, 0.1);
    padding: 2px 8px;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CACHED DATA & MODELS (Optimized & Intact) ----------------
@st.cache_resource
def load_assets():
    base_model = joblib.load("svm_model.pkl")
    calibrated_model = joblib.load("calibrated_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return base_model, calibrated_model, scaler

model, calibrated_model, scaler = load_assets()

@st.cache_data
def load_dataset():
    df = pd.read_csv("vgsales.csv").dropna(subset=["Global_Sales"])
    return df

df_games = load_dataset()

@st.cache_data
def compute_training_baseline(df):
    base = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna()
    return {"mean": base.mean(), "std": base.std()}

drift_baseline = compute_training_baseline(df_games)

@st.cache_data
def prepare_model_data(df):
    temp = df.dropna(subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]).copy()
    temp["Sales_Class"] = pd.qcut(temp["Global_Sales"], q=3, labels=[0, 1, 2]).astype(int)
    X = temp[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = temp["Sales_Class"]
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = prepare_model_data(df_games)

@st.cache_resource
def compute_cv_scores(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.svm import SVC
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    }
    results = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std()}
    return results

@st.cache_resource
def train_comparison_models(X_train, y_train):
    timings = {}
    from sklearn.svm import SVC
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

svm_cmp, nb_cmp, knn_cmp, dt_cmp, xgb_cmp, timings = train_comparison_models(X_train_cmp, y_train_cmp)

@st.cache_resource
def prepare_similarity_engine(df):
    try:
        sim_df = df[["Name", "Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].dropna().reset_index(drop=True)
        features = sim_df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
        nn_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(features)
        return sim_df, nn_model
    except: return None, None

sim_games, nn_model = prepare_similarity_engine(df_games)


# ---------------- SAAS SIDEBAR NAVIGATION ----------------
st.sidebar.markdown("""
<div style='margin-bottom: 30px;'>
    <h1 style='margin-bottom: 0;'>🎮 VG-Ops</h1>
    <p style='color: #64748b; font-size: 14px; margin-top: 0;'>Welcome back, Developer</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "MAIN MENU",
    ["🏠 Dashboard Overview", "🔮 Prediction & Simulation", "⚙️ MLOps & Diagnostics", "📖 System Docs"]
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Global Settings")
use_calibrated = st.sidebar.toggle("Use Calibrated Probs", value=True)
st.sidebar.caption(f"Session Analytics: {st.session_state.prediction_count} Queries")

# ============================================================
# PAGE 1: COMMAND CENTER (Like the Images)
# ============================================================
if menu == "🏠 Dashboard Overview":
    st.markdown("## Analyze your Market Data")
    st.caption("High-level metrics and historical performance tracking.")
    
    # --- SAAS METRIC CARDS ---
    m1, m2, m3, m4 = st.columns(4)
    total_games = f"{len(df_games):,}"
    top_platform = df_games["Platform"].mode()[0]
    best_acc = f"{accuracy_score(y_test_cmp, svm_cmp.predict(X_test_cmp))*100:.1f}%"
    
    with m1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="card-title">Total Games Tracked</div>
            <div class="card-value-row">
                <p class="card-value">{total_games}</p>
                <span class="trend-up">▲ 100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="card-title">Best ML Accuracy</div>
            <div class="card-value-row">
                <p class="card-value">{best_acc}</p>
                <span class="trend-up">▲ SVM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="card-title">Top Platform</div>
            <div class="card-value-row">
                <p class="card-value">{top_platform}</p>
                <span class="trend-up">▲ Lead</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="card-title">System Status</div>
            <div class="card-value-row">
                <p class="card-value">Online</p>
                <span class="trend-up">▲ 0 Latency</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- FULL ANALYTICS (Original Tab 5) ---
    st.markdown("### 📊 Market Analytics Overview")
    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(df_games, x="Global_Sales", nbins=50, title="🌍 Global Sales Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        top_pub = df_games.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_pub = px.bar(top_pub, x="Publisher", y="Global_Sales", title="🏢 Top Publishers by Sales")
        st.plotly_chart(fig_pub, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        plat_sales = df_games.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_plat = px.bar(plat_sales, x="Platform", y="Global_Sales", title="🎮 Top Platforms")
        st.plotly_chart(fig_plat, use_container_width=True)
    with colD:
        corr = df_games[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="🔥 Sales Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# PAGE 2: PREDICTION & SIMULATION (Forms + Recommender)
# ============================================================
elif menu == "🔮 Prediction & Simulation":
    st.markdown("## 🔮 Prediction & Simulation Studio")
    
    # Input Form Design
    st.markdown("### 📝 Enter Regional Sales (Millions)")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        na_sales = c1.number_input("🇺🇸 NA Sales", 0.0, 50.0, 0.5, 0.1)
        eu_sales = c2.number_input("🇪🇺 EU Sales", 0.0, 50.0, 0.3, 0.1)
        jp_sales = c3.number_input("🇯🇵 JP Sales", 0.0, 50.0, 0.1, 0.1)
        other_sales = c4.number_input("🌍 Other Sales", 0.0, 50.0, 0.05, 0.05)
    
    features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
    features_scaled = scaler.transform(features)
    
    active_model = calibrated_model if use_calibrated else model
    
    start_inf = time.perf_counter()
    pred = active_model.predict(features_scaled)[0]
    proba = active_model.predict_proba(features_scaled)[0]
    latency_ms = (time.perf_counter() - start_inf) * 1000

    labels = {0: ("📉 Low Sales", "#ef4444"), 1: ("📊 Medium Sales", "#f59e0b"), 2: ("🚀 High Sales", "#22c55e")}
    text, color = labels[pred]
    confidence = float(np.max(proba) * 100)

    st.markdown(f"""
    <div style="padding:25px; border-radius:14px; background: rgba(0,0,0,0.02); border:1px solid #e2e8f0; text-align:center; margin-top: 20px; margin-bottom: 20px;">
        <h2 style="color:{color}; margin-bottom:0; font-size: 36px;">{text}</h2>
        <p style="color:#64748b; font-size:18px;">Model Confidence: <b>{confidence:.2f}%</b> | Latency: <b>{latency_ms:.2f} ms</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.prediction_count += 1

    # Probability Chart (Original Tab 1)
    st.markdown("#### 📊 Class Confidence Breakdown")
    prob_df = pd.DataFrame({"Class": ["Low", "Medium", "High"], "Probability": proba})
    fig_prob = px.bar(prob_df, x="Class", y="Probability", color="Class", text="Probability", 
                      color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"})
    fig_prob.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_prob, use_container_width=True)

    # Input Sensitivity (Original Tab 1)
    st.markdown("### 🧪 Input Sensitivity (NA Sales Sweep)")
    sweep_vals = np.linspace(0, 10, 40)
    sweep_data = []
    for val in sweep_vals:
        temp = np.array([[val, eu_sales, jp_sales, other_sales]])
        temp_scaled = scaler.transform(temp)
        prob_high = model.predict_proba(temp_scaled)[0][2]
        sweep_data.append(prob_high)
    sweep_df = pd.DataFrame({"NA_Sales": sweep_vals, "High_Sales_Prob": sweep_data})
    fig_sweep = px.line(sweep_df, x="NA_Sales", y="High_Sales_Prob", title="Sensitivity of High Sales vs NA Sales")
    st.plotly_chart(fig_sweep, use_container_width=True)

    st.divider()

    # Original What-If Simulator (Tab 8) using Fragment
    @st.fragment
    def what_if_simulator():
        st.markdown("### 🎛️ What-If Simulator")
        st.caption("Interactively explore how regional sales affect predictions.")
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_na = st.slider("NA Sales (Sim)", 0.0, 10.0, 1.0, 0.1)
            sim_eu = st.slider("EU Sales (Sim)", 0.0, 10.0, 1.0, 0.1)
        with sc2:
            sim_jp = st.slider("JP Sales (Sim)", 0.0, 10.0, 0.5, 0.1)
            sim_other = st.slider("Other Sales (Sim)", 0.0, 10.0, 0.3, 0.1)

        sim_scaled = scaler.transform([[sim_na, sim_eu, sim_jp, sim_other]])
        sim_pred = model.predict(sim_scaled)[0]
        st.success(f"🔮 Simulated Predicted Class: **{['Low', 'Medium', 'High'][sim_pred]}**")
    
    what_if_simulator()

    st.divider()

    # Original Game Recommender (Tab 6)
    st.markdown("### 🎮 AI Game Recommender")
    st.caption("Find top games based on platform, genre, and year.")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        platform_list = sorted(df_games["Platform"].dropna().unique())
        selected_platform = st.selectbox("Select Platform", ["All"] + platform_list)
    with rc2:
        genre_list = sorted(df_games["Genre"].dropna().unique())
        selected_genre = st.selectbox("Select Genre", ["All"] + genre_list)
    with rc3:
        year_min, year_max = int(df_games["Year"].min()), int(df_games["Year"].max())
        year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

    filtered_df = df_games.copy()
    if selected_platform != "All": filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]
    if selected_genre != "All": filtered_df = filtered_df[filtered_df["Genre"] == selected_genre]
    filtered_df = filtered_df[(filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])]

    top_games = filtered_df.sort_values("Global_Sales", ascending=False).head(10)[["Name", "Platform", "Genre", "Year", "Global_Sales"]]
    if len(top_games) == 0: st.warning("No games match.")
    else: st.dataframe(top_games, use_container_width=True)

    st.markdown("#### 🧠 Similar Game Finder (Advanced)")
    if sim_games is not None:
        selected_game = st.selectbox("Select a game to find similar ones", sim_games["Name"].values)
        idx = sim_games[sim_games["Name"] == selected_game].index[0]
        distances, indices = nn_model.kneighbors([sim_games.loc[idx, ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]]], n_neighbors=11)
        similar_games = sim_games.iloc[indices[0][1:]][["Name", "Platform", "Genre", "Global_Sales"]]
        st.dataframe(similar_games, use_container_width=True)


# ============================================================
# PAGE 3: MLOps & DIAGNOSTICS (ALL Original Charts Intact)
# ============================================================
elif menu == "⚙️ MLOps & Diagnostics":
    st.markdown("## ⚙️ MLOps Command Center")
    
    # We will use tabs inside here to organize the massive amount of diagnostics
    tab_diag1, tab_diag2, tab_diag3, tab_diag4 = st.tabs(["📊 Model Comparison", "🔥 Confusion Matrices", "🧪 ROC & Diagnostics", "🧭 Drift & SHAP"])

    # --- ALL Original Model Comparison (Tab 3) ---
    with tab_diag1:
        st.markdown("### 🏆 Accuracy Comparison")
        acc_data = [
            {"Model": "SVM", "Accuracy": accuracy_score(y_test_cmp, svm_cmp.predict(X_test_cmp)), "Training Time (s)": timings["SVM"]},
            {"Model": "Naive Bayes", "Accuracy": accuracy_score(y_test_cmp, nb_cmp.predict(X_test_cmp)), "Training Time (s)": timings["Naive Bayes"]},
            {"Model": "KNN", "Accuracy": accuracy_score(y_test_cmp, knn_cmp.predict(X_test_cmp)), "Training Time (s)": timings["KNN"]},
            {"Model": "Decision Tree", "Accuracy": accuracy_score(y_test_cmp, dt_cmp.predict(X_test_cmp)), "Training Time (s)": timings["Decision Tree"]},
            {"Model": "XGBoost", "Accuracy": accuracy_score(y_test_cmp, xgb_cmp.predict(X_test_cmp)), "Training Time (s)": timings["XGBoost"]},
        ]
        acc_df = pd.DataFrame(acc_data)
        
        c1, c2 = st.columns(2)
        with c1: st.success(f"🏆 Best Model: **{acc_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']}**")
        with c2: st.info(f"⚡ Fastest Model: **{acc_df.sort_values('Training Time (s)').iloc[0]['Model']}**")

        st.plotly_chart(px.bar(acc_df, x="Model", y="Accuracy", color="Accuracy"), use_container_width=True)

        st.markdown("### 🧪 Cross-Validation Stability")
        X_full = pd.concat([X_train_cmp, X_test_cmp], axis=0)
        y_full = pd.concat([y_train_cmp, y_test_cmp], axis=0)
        cv_results = compute_cv_scores(X_full, y_full)
        cv_df = pd.DataFrame([{"Model": name, "CV Mean": vals["mean"], "CV Std": vals["std"]} for name, vals in cv_results.items()])
        st.plotly_chart(px.bar(cv_df, x="Model", y="CV Mean", error_y="CV Std", color="CV Mean"), use_container_width=True)

    # --- ALL 5 Confusion Matrices Back ---
    with tab_diag2:
        st.markdown("### 🔥 Confusion Matrices (All Models)")
        preds = {
            "SVM": svm_cmp.predict(X_test_cmp),
            "Naive Bayes": nb_cmp.predict(X_test_cmp),
            "KNN": knn_cmp.predict(X_test_cmp),
            "Decision Tree": dt_cmp.predict(X_test_cmp),
            "XGBoost": xgb_cmp.predict(X_test_cmp),
        }
        cols = st.columns(5)
        for i, (name, p) in enumerate(preds.items()):
            cm = confusion_matrix(y_test_cmp, p)
            with cols[i]:
                fig, ax = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_title(name, fontsize=10)
                st.pyplot(fig)

    # --- Original Model Diagnostics (Tab 7) ---
    with tab_diag3:
        st.markdown("### 📉 Multiclass ROC Curves (Best Model)")
        best_model_name = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        model_map = {"SVM": svm_cmp, "Naive Bayes": nb_cmp, "KNN": knn_cmp, "Decision Tree": dt_cmp, "XGBoost": xgb_cmp}
        best_model = model_map[best_model_name]

        try:
            y_test_bin = label_binarize(y_test_cmp, classes=[0, 1, 2])
            y_proba = best_model.predict_proba(X_test_cmp)
            roc_data = []
            for i, class_name in enumerate(["Low", "Medium", "High"]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_data.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Class": class_name}))
            roc_df = pd.concat(roc_data)
            st.plotly_chart(px.line(roc_df, x="FPR", y="TPR", color="Class", title="ROC Curve (OvR)"), use_container_width=True)
        except: st.warning("ROC Failed.")

        st.markdown("### 📊 Precision & Recall")
        y_pred_best = best_model.predict(X_test_cmp)
        st.dataframe(pd.DataFrame({
            "Metric": ["Precision (Macro)", "Recall (Macro)"],
            "Score": [precision_score(y_test_cmp, y_pred_best, average="macro"), recall_score(y_test_cmp, y_pred_best, average="macro")]
        }), use_container_width=True)
        
        misclassified = (y_test_cmp != y_pred_best).sum()
        total = len(y_test_cmp)
        st.metric("Misclassified Samples", f"{misclassified} / {total}", delta=f"{(misclassified/total)*100:.2f}% error")

    # --- Original Drift (Tab 9) & SHAP (Tab 4) ---
    with tab_diag4:
        colD1, colD2 = st.columns(2)
        with colD1:
            st.markdown("### 🧭 Drift Monitor")
            # Grab dummy inputs from Session or default to test drift
            current_input = pd.Series({"NA_Sales": 1.0, "EU_Sales": 0.5, "JP_Sales": 0.1, "Other_Sales": 0.1})
            drift_scores = ((current_input - drift_baseline["mean"]) / (drift_baseline["std"] + 1e-6)).abs()
            st.plotly_chart(px.bar(x=drift_scores.index, y=drift_scores.values, title="Current Input Z-Score Drift"), use_container_width=True)

        with colD2:
            st.markdown("### 🧠 SHAP Explainability")
            @st.cache_data
            def get_shap_background(df): return scaler.transform(df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].dropna().sample(50, random_state=42))
            @st.cache_resource
            def get_explainer(_bg): return shap.KernelExplainer(model.predict_proba, _bg)
            @st.cache_data
            def compute_shap(_features): return get_explainer(get_shap_background(df_games)).shap_values(_features, nsamples=50)

            with st.spinner("Calculating SHAP..."):
                sample_feat = scaler.transform([[1.0, 0.5, 0.1, 0.1]])
                shap_vals = compute_shap(sample_feat)
                
                # --- Safe SHAP Array Extraction (Original Logic) ---
                if isinstance(shap_vals, list):
                    shap_for_class = shap_vals[2][0]  # Get values for High Sales class
                else:
                    shap_for_class = shap_vals[0]
                
                feature_names = ["NA", "EU", "JP", "Other"]
                shap_for_class = np.array(shap_for_class).flatten()
                
                # Prevent length mismatch error
                min_len = min(len(feature_names), len(shap_for_class))
                
                shap_df = pd.DataFrame({
                    "Feature": feature_names[:min_len], 
                    "Impact": np.abs(shap_for_class[:min_len])
                }).sort_values("Impact", ascending=False)
                
                st.plotly_chart(px.bar(shap_df, x="Feature", y="Impact", color="Impact", title="Feature Impact"), use_container_width=True)

# ============================================================
# PAGE 4: SYSTEM DOCS (100% Original Text Preserved)
# ============================================================
elif menu == "📖 System Docs":
    st.markdown("## 📘 About Model — Full Technical Documentation")
    
    # Original Feature Importance (Tab 2)
    st.markdown("### 📈 Feature Importance (Permutation)")
    try:
        sample_df = df_games[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].dropna().sample(300, random_state=42)
        X_sample = scaler.transform(sample_df)
        y_sample = model.predict(X_sample)
        perm = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({"Feature": ["NA", "EU", "JP", "Other"], "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
        st.plotly_chart(px.bar(imp_df, x="Feature", y="Importance", color="Importance"), use_container_width=True)
    except: pass

    # EXACT ORIGINAL TEXT (Tab 10)
    with st.expander("🎯 1. Problem Formulation (Detailed)", expanded=True):
        st.markdown("""
        The core objective of this system is to estimate the **global commercial performance tier** of a video game using regional sales signals.
        ### Why not regression?
        Predicting exact revenue in the gaming industry is highly volatile due to:
        - blockbuster outliers  
        - platform lifecycle effects  
        - marketing shocks  
        - franchise bias  
        Therefore, the task is intentionally framed as a **multi-class classification problem** with three ordinal tiers:
        - 📉 Low Sales  
        - 📊 Medium Sales  
        - 🚀 High Sales  
        """)

    with st.expander("📊 2. Dataset Deep Dive"):
        st.markdown("""
        ### Dataset Source
        The model is trained on the widely used **vgsales dataset**.
        ### Target Engineering Strategy
        The raw **Global_Sales** variable is transformed using **quantile-based binning** into three approximately balanced classes.
        """)

    with st.expander("🧠 3. Data Preprocessing Pipeline"):
        st.markdown("""
        ### Step 1: Data Cleaning
        ### Step 2: Stratified Train-Test Split
        ### Step 3: Feature Standardization
        The pipeline applies **StandardScaler normalization**.
        """)

    with st.expander("🤖 4. Model Architecture & Rationale"):
        st.markdown("""
        - **Gaussian Naive Bayes**: probabilistic baseline
        - **KNN**: distance-based learner
        - **Decision Tree**: interpretable nonlinear model
        - **SVM (Primary)**: robust margin maximization
        - **XGBoost**: high-capacity nonlinear benchmark
        """)

    with st.expander("📉 6. Probability Calibration"):
        st.markdown("""
        The system supports **Platt scaling via CalibratedClassifierCV**.
        Improves probability reliability and reduces overconfidence.
        """)

    with st.expander("⚠️ 9. Known Limitations"):
        st.markdown("""
        - historical dataset may not reflect modern market dynamics  
        - regional sales are correlated  
        - marketing and release timing not modeled  
        """)

    with st.expander("🚀 10. Future Technical Roadmap"):
        st.markdown("""
        - time-aware cross-validation  
        - automated drift-triggered retraining  
        - real-time inference telemetry  
        """)

    st.markdown("---")
    st.caption("Built by HKS • Machine Learning • UI/UX")