import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import shap
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Video Game Sales Predictor",
    page_icon="🎮",
    layout="wide"
)
# ---------------- PREMIUM GLOBAL CSS ----------------
st.markdown("""
<style>

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
    background: rgba(17, 25, 40, 0.75);
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
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ----- LOAD DATASET  -----
@st.cache_data
def load_dataset():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna(subset=["Global_Sales"])
    return df

df_games = load_dataset()

# ---------------- MODEL COMPARISON DATA ----------------
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

features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
features_scaled = scaler.transform(features)

pred = model.predict(features_scaled)[0]
proba = model.predict_proba(features_scaled)[0]
confidence = np.max(proba) * 100

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Prediction",
    "📈 Feature Importance",
    "📊 Model Comparison",
    "🧠 SHAP Explainability",
    "📊 Analytics Dashboard",
    "🎮 Game Recommender"
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
    with st.spinner("Analyzing game sales pattern..."):

        features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba) * 100

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
    st.subheader("📊 Real Model Comparison")

    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    # ----- Train models -----
    svm_cmp = SVC(probability=True).fit(X_train_cmp, y_train_cmp)
    nb_cmp = GaussianNB().fit(X_train_cmp, y_train_cmp)
    knn_cmp = KNeighborsClassifier(n_neighbors=7).fit(X_train_cmp, y_train_cmp)

    # ----- Predictions -----
    preds = {
        "SVM": svm_cmp.predict(X_test_cmp),
        "Naive Bayes": nb_cmp.predict(X_test_cmp),
        "KNN": knn_cmp.predict(X_test_cmp),
    }

    # ----- Accuracy table -----
    acc_data = []
    for name, p in preds.items():
        acc_data.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test_cmp, p)
        })

    acc_df = pd.DataFrame(acc_data)

    st.markdown("### 🏆 Accuracy Comparison")

    fig_acc = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        color="Accuracy"
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Confusion Matrices")

    cols = st.columns(3)

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
        # ---------- safe background ----------
        background = np.zeros((20, 4))

        @st.cache_resource
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
        st.warning("No games found for selected filters.")
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

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by HKS • Machine Learning • UI/UX")