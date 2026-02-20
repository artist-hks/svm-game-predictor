from numpy.random.mtrand import sample
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

/* ---- Main container ---- */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---- Header ---- */
.main-header {
    font-size: 44px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2em;
}

.sub-header {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 2em;
}

/* ---- Glass cards ---- */
.glass-card {
    padding: 22px;
    border-radius: 16px;
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* ---- Metric number ---- */
.metric-big {
    font-size: 32px;
    font-weight: 700;
}

/* ---- Section spacing ---- */
.section-gap {
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    base_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # ---------- build calibrated model safely ----------
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    # recreate SAME SVM config (important)
    svm_clone = SVC(
        probability=True,
        kernel="rbf",
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        svm_clone,
        method="sigmoid",
        cv=3
    )

    # ---------- prepare real calibration data ----------
    df = pd.read_csv("vgsales.csv").dropna(subset=[
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"
    ])

    df["Sales_Class"] = pd.qcut(
        df["Global_Sales"],
        q=3,
        labels=[0, 1, 2]
    ).astype(int)

    X_cal = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y_cal = df["Sales_Class"]

    # sample for speed
    X_sample = X_cal.sample(800, random_state=42)
    y_sample = y_cal.loc[X_sample.index]

    X_scaled = scaler.transform(X_sample)

    # fit calibrated model cleanly
    calibrated_model.fit(X_scaled, y_sample)

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
    base = df[[
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"
    ]].dropna()

    return {
        "mean": base.mean(),
        "std": base.std()
    }

drift_baseline = compute_training_baseline(df_games)

# ---------------- MODEL COMPARISON DATA ----------------
@st.cache_resource
def compute_cv_scores(X, y):
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "mean": scores.mean(),
            "std": scores.std()
        }

    return results
@st.cache_resource
def train_comparison_models(X_train, y_train):
    import time
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    timings = {}

    # ---------- SVM ----------
    start = time.perf_counter()
    svm = SVC(probability=True).fit(X_train, y_train)
    timings["SVM"] = time.perf_counter() - start

    # ---------- Naive Bayes ----------
    start = time.perf_counter()
    nb = GaussianNB().fit(X_train, y_train)
    timings["Naive Bayes"] = time.perf_counter() - start

    # ---------- KNN ----------
    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    timings["KNN"] = time.perf_counter() - start

        # ---------- Decision Tree ----------
    start = time.perf_counter()
    dt = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ).fit(X_train, y_train)
    timings["Decision Tree"] = time.perf_counter() - start

    # ---------- XGBoost (NEW WEAPON) ----------
    start = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    timings["XGBoost"] = time.perf_counter() - start

    return svm, nb, knn, dt, xgb, timings
    import time
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    timings = {}

    start = time.perf_counter()
    svm = SVC(probability=True).fit(X_train, y_train)
    timings["SVM"] = time.perf_counter() - start

    start = time.perf_counter()
    nb = GaussianNB().fit(X_train, y_train)
    timings["Naive Bayes"] = time.perf_counter() - start

    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    timings["KNN"] = time.perf_counter() - start

    return svm, nb, knn, timings
@st.cache_data
def prepare_model_data(df):
    temp = df.dropna(subset=[
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"
    ]).copy()

    # create target classes using quantiles
    temp["Sales_Class"] = pd.qcut(
        temp["Global_Sales"],
        q=3,
        labels=[0, 1, 2]
    )

    X = temp[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = temp["Sales_Class"].astype(int)

    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = prepare_model_data(df_games)

# ---------------- MEMORY SAFE SIMILARITY ----------------
@st.cache_resource
def prepare_similarity_engine(df):
    try:
        sim_df = df[[
            "Name",
            "Platform",
            "Genre",
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales",
            "Global_Sales"
        ]].dropna().reset_index(drop=True)

        features = sim_df[[
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales"
        ]]

        nn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )
        nn_model.fit(features)

        return sim_df, nn_model

    except Exception:
        return None, None


sim_games, nn_model = prepare_similarity_engine(df_games)
# ---------------- HEADER ----------------
st.toast("Model ready", icon="🤖")
st.markdown('<div class="main-header">🎮 Video Game Sales Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict • Analyze • Recommend</div>', unsafe_allow_html=True)

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
    st.subheader("Prediction Result")

    labels = {
        0: ("📉 Low Sales", "#ef4444"),
        1: ("📊 Medium Sales", "#f59e0b"),
        2: ("🚀 High Sales", "#22c55e")
    }
    st.caption("Adjust regional sales from the sidebar to explore predictions.")

    # ---------- HEAVY COMPUTATION WITH SPINNER ----------
    with st.spinner("Running ML inference..."):

        features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
        features_scaled = scaler.transform(features)

        active_model = calibrated_model if use_calibrated else model

        pred = active_model.predict(features_scaled)[0]
        proba = active_model.predict_proba(features_scaled)[0]
        confidence = np.max(proba) * 100

    st.session_state.prediction_count += 1
        # ---------- CALIBRATION HINT ----------
    entropy = -np.sum(proba * np.log(proba + 1e-9))
    st.caption(f"Prediction certainty score: {1/(1+entropy):.3f}")

    text, color = labels[pred]

    # ---------- RESULT CARD ----------
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

    # ---------- KPI METRICS ----------
    st.markdown("---")
    k1, k2, k3 = st.columns(3)

    k1.metric("Predicted Class", text.replace("📉 ", "").replace("📊 ", "").replace("🚀 ", ""))
    k2.metric("Confidence", f"{confidence:.2f}%")
    k3.metric("Total Input Sales", f"{na_sales+eu_sales+jp_sales+other_sales:.2f}")

    # ---------- PROBABILITY CHART ----------
    st.markdown("---")
    # ---------- ANIMATED SPEEDOMETER ----------
    st.markdown("### 🎯 Confidence Meter")

    gauge_speed = px.imshow([[confidence]], text_auto=False)

    gauge_fig = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number",
            "value": confidence,
            "number": {"suffix": "%"},
            "gauge": {
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e"},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#052e16"},
                ],
            },
        }]
    }

    st.plotly_chart(gauge_fig, use_container_width=True)
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
    # ---------- PROBABILITY PROGRESS BARS ----------
    st.markdown("### 📊 Class Confidence Breakdown")

    class_names = ["Low", "Medium", "High"]

    for cls, prob in zip(class_names, proba):
        st.progress(float(prob))
        st.caption(f"{cls}: {prob:.2%}")

    # ---------- INPUT SENSITIVITY ----------
    st.markdown("### 🧪 Input Sensitivity (NA Sales Sweep)")

    sweep_vals = np.linspace(0, 10, 40)

    sweep_data = []
    for val in sweep_vals:
        temp = np.array([[val, eu_sales, jp_sales, other_sales]])
        temp_scaled = scaler.transform(temp)
        prob_high = model.predict_proba(temp_scaled)[0][2]
        sweep_data.append(prob_high)

    sweep_df = pd.DataFrame({
        "NA_Sales": sweep_vals,
        "High_Sales_Prob": sweep_data
    })

    fig_sweep = px.line(
        sweep_df,
        x="NA_Sales",
        y="High_Sales_Prob",
        title="Sensitivity of High Sales Probability vs NA Sales"
    )

    st.plotly_chart(fig_sweep, use_container_width=True)
# ============================================================
# TAB 2 — FEATURE IMPORTANCE (Permutation for SVM)
# ============================================================
with tab2:
    st.subheader("Feature Importance (Permutation)")

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
            color="Importance"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.warning("Feature importance could not be computed.")

# ============================================================
# TAB 3 — MODEL COMPARISON
# ============================================================
with tab3:
    st.subheader("📊 Real Model Comparison")
    X_full = pd.concat([X_train_cmp, X_test_cmp], axis=0)
    y_full = pd.concat([y_train_cmp, y_test_cmp], axis=0)

    cv_results = compute_cv_scores(X_full, y_full)
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

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
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test_cmp, xgb_cmp.predict(X_test_cmp)),
    "Training Time (s)": timings["XGBoost"]
    },
    ]

    acc_df = pd.DataFrame(acc_data)

    # ---------- SUMMARY CARDS ----------
    best_acc_model = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    fastest_model = acc_df.sort_values("Training Time (s)").iloc[0]["Model"]

    c1, c2 = st.columns(2)

    with c1:
        st.success(f"🏆 Most Accurate Model: **{best_acc_model}**")

    with c2:
        st.info(f"⚡ Fastest Training Model: **{fastest_model}**")

    st.markdown("### 🏆 Accuracy Comparison")

    fig_acc = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        color="Accuracy"
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("### 🧪 Cross-Validation Stability")

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
        title="5-Fold Stratified CV Accuracy"
    )

    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("### ⏱️ Training Time Comparison")

    fig_time = px.bar(
        acc_df,
        x="Model",
        y="Training Time (s)",
        color="Training Time (s)"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Confusion Matrices")

    cols = st.columns(len(preds))

    for i, (name, p) in enumerate(preds.items()):
        cm = confusion_matrix(y_test_cmp, p)

        with cols[i]:
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

# ============================================================
# TAB 4 — SHAP EXPLAINABILITY
# ============================================================
with tab4:
    st.subheader("SHAP Explainability")
    st.caption("Local explanation of the current prediction")

    try:
        
        @st.cache_resource
        @st.cache_data
        def get_shap_background(df):
            sample = df[
        ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
        ].dropna().sample(50, random_state=42)

            return scaler.transform(sample)

        background = get_shap_background(df_games)
        def get_explainer():
            return shap.KernelExplainer(model.predict_proba, background)

        explainer = get_explainer()

        # ---------- compute shap ----------
        shap_values = explainer.shap_values(features_scaled, nsamples=50)

        # ---------- handle multiclass safely ----------
        # shap_values is list for multiclass
        if isinstance(shap_values, list):
            # take predicted class explanation
            class_idx = int(pred)
            shap_for_class = shap_values[class_idx][0]
        else:
            shap_for_class = shap_values[0]

        feature_names = ["NA", "EU", "JP", "Other"]

        # ---------- ensure same length ----------
        shap_for_class = np.array(shap_for_class).flatten()

        min_len = min(len(feature_names), len(shap_for_class))

        shap_df = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Impact": np.abs(shap_for_class[:min_len])
        }).sort_values("Impact", ascending=False)

        st.subheader("Feature Impact")

        fig_shap = px.bar(
            shap_df,
            x="Feature",
            y="Impact",
            color="Impact"
        )

        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.error("SHAP failed to compute.")
        st.caption(str(e))

# ============================================================
# TAB 5 — ANALYTICS DASHBOARD
# ============================================================
with tab5:
    st.subheader("📊 Video Game Sales Analytics")

    colA, colB = st.columns(2)

    # ---------- Global Sales Distribution ----------
    with colA:
        st.markdown("### 🌍 Global Sales Distribution")

        fig_hist = px.histogram(
            df_games,
            x="Global_Sales",
            nbins=50,
            title="Distribution of Global Sales"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- Top Publishers ----------
    with colB:
        st.markdown("### 🏢 Top Publishers")

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
            title="Top 10 Publishers by Global Sales"
        )
        st.plotly_chart(fig_pub, use_container_width=True)

    st.markdown("---")

    colC, colD = st.columns(2)

    # ---------- Platform Sales ----------
    with colC:
        st.markdown("### 🎮 Platform-wise Sales")

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
            title="Top Platforms by Sales"
        )
        st.plotly_chart(fig_plat, use_container_width=True)

    # ---------- Correlation Heatmap ----------
    with colD:
        st.markdown("### 🔥 Sales Correlation")

        corr = df_games[
            ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
        ].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Sales Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ============================================================
# TAB 6 — GAME RECOMMENDER SYSTEM
# ============================================================
with tab6:
    st.subheader("🎮 Video Game Recommender")

    st.caption("Find top games based on platform, genre, and year.")

    # ---------- FILTERS ----------
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
    st.markdown("### 🏆 Top Recommended Games")

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
    st.markdown("---")
    st.subheader("🧠 Similar Game Finder (Advanced)")
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

    # ---------- OPTIONAL CHART ----------
    fig_top = px.bar(
        top_games,
        x="Name",
        y="Global_Sales",
        color="Global_Sales",
        title="Top Recommended Games by Global Sales"
    )
    st.plotly_chart(fig_top, use_container_width=True)
# ============================================================
# TAB 7 — MODEL DIAGNOSTICS
# ============================================================
with tab7:
    st.subheader("🧪 Model Diagnostics")

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
    st.markdown("### 📉 Multiclass ROC Curves")

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

        st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.warning("ROC could not be computed.")

    # =====================================================
    # PRECISION / RECALL TABLE
    # =====================================================
    st.markdown("### 📊 Precision & Recall")

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
    st.markdown("### 🧮 Test Set Class Distribution")

    dist_df = y_test_cmp.value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]

    fig_dist = px.bar(
        dist_df,
        x="Class",
        y="Count",
        color="Count",
        title="Class Distribution in Test Set"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # =====================================================
    # MISCLASSIFICATION COUNT
    # =====================================================
    st.markdown("### 🚨 Misclassification Insight")

    misclassified = (y_test_cmp != y_pred_best).sum()
    total = len(y_test_cmp)

    st.metric(
        "Misclassified Samples",
        f"{misclassified} / {total}",
        delta=f"{(misclassified/total)*100:.2f}% error"
    )
# ============================================================
# TAB 8 — WHAT-IF SIMULATOR
# ============================================================
with tab8:
    st.subheader("🎛️ What-If Simulator")
    st.caption("Interactively explore how regional sales affect predictions.")

    # ---------- input sliders ----------
    c1, c2 = st.columns(2)

    with c1:
        sim_na = st.slider("NA Sales (Sim)", 0.0, 10.0, 1.0, 0.1)
        sim_eu = st.slider("EU Sales (Sim)", 0.0, 10.0, 1.0, 0.1)

    with c2:
        sim_jp = st.slider("JP Sales (Sim)", 0.0, 10.0, 0.5, 0.1)
        sim_other = st.slider("Other Sales (Sim)", 0.0, 10.0, 0.3, 0.1)

    # ---------- prediction ----------
    sim_features = np.array([[sim_na, sim_eu, sim_jp, sim_other]])
    sim_scaled = scaler.transform(sim_features)

    sim_pred = model.predict(sim_scaled)[0]
    sim_proba = model.predict_proba(sim_scaled)[0]

    class_labels = ["Low", "Medium", "High"]

    st.markdown("### 🔮 Simulated Prediction")

    st.success(f"Predicted Class: **{class_labels[sim_pred]}**")

    # =====================================================
    # DECISION SWEEP (NA axis)
    # =====================================================
    st.markdown("### 📉 Decision Sensitivity Curve")

    sweep_vals = np.linspace(0, 10, 60)
    sweep_probs = []

    for val in sweep_vals:
        temp = np.array([[val, sim_eu, sim_jp, sim_other]])
        temp_scaled = scaler.transform(temp)
        sweep_probs.append(model.predict_proba(temp_scaled)[0][2])

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

    st.plotly_chart(fig_boundary, use_container_width=True)

    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================
    st.markdown("### 📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": sim_proba
    })

    fig_prob_sim = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        color="Probability"
    )

    st.plotly_chart(fig_prob_sim, use_container_width=True)

# ============================================================
# TAB 9 — DRIFT MONITOR
# ============================================================
with tab9:
    st.subheader("🧭 Model Drift Monitor")
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
    # DRIFT BAR CHART
    # =====================================================
    st.markdown("### 📊 Feature Drift Scores")

    fig_drift = px.bar(
        drift_df,
        x="Feature",
        y="Drift Score (|z|)",
        color="Drift Score (|z|)",
        title="Current Input vs Training Distribution"
    )

    st.plotly_chart(fig_drift, use_container_width=True)

    # =====================================================
    # HEALTH STATUS
    # =====================================================
    max_drift = drift_scores.max()

    st.markdown("### 🚨 Drift Health Status")

    if max_drift < 1:
        st.success("✅ Input is within normal training range.")
    elif max_drift < 2.5:
        st.warning("⚠️ Moderate drift detected. Monitor recommended.")
    else:
        st.error("🚨 High drift! Model may be unreliable.")

    # =====================================================
    # RAW COMPARISON TABLE
    # =====================================================
    st.markdown("### 🔍 Detailed Comparison")

    compare_df = pd.DataFrame({
        "Current": current_input,
        "Training Mean": baseline_mean,
        "Training Std": baseline_std
    })

    st.dataframe(compare_df, use_container_width=True)

# ============================================================
# TAB 10 — ABOUT MODEL (FULL TECHNICAL DOCUMENTATION)
# ============================================================
with tab10:
    st.subheader("📘 About Model — Technical Deep Dive")
    st.caption("Comprehensive technical documentation of the Video Game Sales Intelligence system.")

    st.markdown("""
# 🎮 Video Game Sales Intelligence — System Documentation

This dashboard represents a **production-style end-to-end machine learning system**
designed to predict the global commercial performance tier of video games based on
regional sales signals.

Unlike typical academic demos that stop at model accuracy, this system emphasizes:

- reproducible preprocessing  
- multi-model benchmarking  
- probability calibration  
- explainability  
- drift awareness  
- interactive simulation  

The objective is to demonstrate disciplined machine learning engineering practices
in a deployable analytical interface.

---

# 🎯 1. Problem Definition

## 1.1 Business Framing

Predicting exact video game revenue is highly noisy and sensitive to outliers.
Instead of treating the task as regression, the problem is formulated as a
**multi-class classification task**.

The system predicts whether a game belongs to:

- 📉 Low Sales  
- 📊 Medium Sales  
- 🚀 High Sales  

### Why classification instead of regression?

- reduces sensitivity to extreme revenue values  
- improves interpretability for decision support  
- stabilizes model training  
- aligns better with risk-tier business thinking  
- simplifies evaluation across multiple algorithms  

This framing converts raw revenue prediction into a **structured success
categorization problem**.

---

## 1.2 Machine Learning Formulation

Given regional sales vector:

X = (NA, EU, JP, Other)

Learn mapping:

f(X) → {Low, Medium, High}

This is treated as a **supervised multi-class classification problem**
with balanced class construction.

---

# 📊 2. Dataset Description

## 2.1 Source

The system uses the widely adopted **vgsales dataset**, which aggregates
historical video game performance across platforms and regions.

The dataset includes:

- platform information  
- genre metadata  
- publisher data  
- regional sales figures  
- global sales totals  

For modeling stability, the current pipeline focuses on **numerical
regional sales features**.

---

## 2.2 Feature Set Used

Primary predictive features:

- **NA_Sales** — North America performance  
- **EU_Sales** — Europe performance  
- **JP_Sales** — Japan performance  
- **Other_Sales** — Rest-of-world performance  

These variables are continuous and moderately correlated, making them
suitable for margin-based, distance-based, and tree-based learners.

---

## 2.3 Target Engineering

The raw **Global_Sales** variable is transformed using **quantile binning**
into three approximately balanced classes.

### Motivation for quantile binning

- prevents severe class imbalance  
- stabilizes multi-class learning  
- improves cross-model comparability  
- reduces skew from blockbuster outliers  

Formally:

- bottom quantile → Low  
- middle quantile → Medium  
- top quantile → High  

This creates a **balanced ordinal classification target**.

---

# 🧠 3. Data Preprocessing Pipeline

The preprocessing stage follows a production-aware structure.

## 3.1 Data Cleaning

Steps performed:

- removal of rows missing critical sales fields  
- validation of numeric feature ranges  
- index reset after filtering  

This ensures model inputs remain consistent.

---

## 3.2 Train-Test Strategy

The dataset is split using:

- **stratified sampling**  
- fixed random seed  
- preserved class proportions  

Stratification is critical because naive splits can distort class balance,
leading to misleading accuracy.

---

## 3.3 Feature Scaling

The pipeline applies **StandardScaler normalization**.

### Why scaling matters

Scaling is essential for:

- Support Vector Machines  
- K-Nearest Neighbors  
- distance-sensitive algorithms  
- gradient-based optimizers  

Without scaling:

- SVM margins distort  
- KNN distance becomes biased  
- convergence stability drops  

The scaler is:

- fit on training data  
- persisted via joblib  
- reused at inference time  

This guarantees **train–serve consistency**.

---

# 🤖 4. Model Architecture

The system intentionally implements a **diverse model portfolio**
to compare learning biases.

## 4.1 Gaussian Naive Bayes

Role:

- probabilistic baseline  
- fast training  
- low variance  

Characteristics:

- assumes conditional independence  
- works well on simple distributions  
- provides reference floor performance  

---

## 4.2 K-Nearest Neighbors

Role:

- distance-based learner  
- non-parametric baseline  

Strengths:

- captures local structure  
- no explicit training phase  

Limitations:

- sensitive to feature scaling  
- slower inference at scale  

---

## 4.3 Decision Tree

Role:

- interpretable nonlinear model  
- rule-based structure  

Advantages:

- human-readable splits  
- handles nonlinearity  
- feature interaction capture  

Controls applied:

- max_depth constraint  
- random_state for reproducibility  

---

## 4.4 Support Vector Machine (Primary Model)

The SVM serves as the **primary deployed classifier**.

### Why SVM?

- strong performance on tabular data  
- robust decision boundaries  
- effective in medium-dimensional spaces  
- good bias–variance tradeoff  

Kernel: RBF  
Probability mode: enabled  
Input: standardized features  

---

## 4.5 XGBoost

Role:

- gradient boosting benchmark  
- high-capacity nonlinear learner  

Strengths:

- captures complex interactions  
- strong tabular performance  
- ensemble robustness  

Used primarily for **comparative benchmarking**.

---

# 🧪 5. Evaluation Framework

The system avoids single-metric evaluation.

## 5.1 Hold-Out Accuracy

Provides quick sanity check but is **not relied upon alone**.

---

## 5.2 Stratified 5-Fold Cross-Validation

Used to estimate generalization stability.

Benefits:

- reduces split variance  
- exposes overfitting  
- improves credibility  

Metrics reported:

- mean accuracy  
- standard deviation  

---

## 5.3 Confusion Matrix Analysis

Helps identify:

- class-wise weaknesses  
- misclassification patterns  
- bias toward specific tiers  

---

## 5.4 Multiclass ROC (OvR)

Implements one-vs-rest ROC curves.

Purpose:

- threshold behavior analysis  
- separability inspection  
- model discrimination power  

---

## 5.5 Macro Precision & Recall

Macro averaging ensures:

- equal weight to all classes  
- robustness under class imbalance  
- fair multi-class evaluation  

---

# 📉 6. Probability Calibration

Raw classifier probabilities are often misaligned.

The system supports **Platt scaling** via:

CalibratedClassifierCV

### Why calibration matters

Uncalibrated models may:

- be overconfident  
- mislead downstream decisions  
- distort risk interpretation  

Calibration improves:

- probability reliability  
- decision thresholding  
- trustworthiness  

Users can toggle calibrated inference from the sidebar.

---

# 🔍 7. Explainability Layer

Transparency is implemented through multiple techniques.

## 7.1 Permutation Importance

Measures performance drop when features are shuffled.

Benefits:

- model-agnostic  
- intuitive  
- robust baseline importance  

---

## 7.2 SHAP Local Explanations

Provides per-prediction attribution.

Helps answer:

> “Why did the model predict this class?”

---

## 7.3 Probability Breakdown

Displays class-wise confidence distribution.

---

## 7.4 Input Sensitivity Analysis

Performs controlled feature sweeps to visualize response curves.

This reveals:

- monotonicity  
- saturation effects  
- decision sensitivity  

---

# 🧭 8. Drift Monitoring

A lightweight drift detector compares live inputs against the
training distribution.

Method:

- compute z-score distance from training mean  
- flag moderate or high deviation  

Purpose:

- detect out-of-distribution queries  
- warn about reliability degradation  
- simulate production monitoring  

---

# ⚠️ 9. Known Limitations

No serious system hides its weaknesses.

Key constraints:

- historical dataset may not reflect modern markets  
- regional sales are correlated  
- quantile binning simplifies revenue dynamics  
- temporal leakage not fully modeled  
- marketing factors absent  
- platform lifecycle effects ignored  

Therefore, outputs should be interpreted as **indicative signals,
not financial guarantees**.

---

# 🚀 10. Future Technical Roadmap

Planned improvements:

- time-aware cross-validation  
- richer categorical encoding  
- automated drift-triggered retraining  
- expanded dataset coverage  
- reliability diagram visualization  
- real-time inference telemetry  
- model versioning dashboard  

---

# 🧩 11. Design Philosophy

This project is intentionally structured to resemble a **mini production
ML system**, not a notebook prototype.

Core principles:

- reproducibility  
- comparability  
- interpretability  
- monitoring awareness  
- user-centric interaction  

The emphasis is on building machine learning systems that are not only
accurate but also **transparent, reliable, and deployable**.
""")

    st.markdown("---")
    st.caption("Model Version: v1.0 • Production-Style ML Dashboard • Built by HKS")
# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by HKS • Machine Learning • UI/UX")