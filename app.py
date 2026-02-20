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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


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

# ----- LOAD DATASET  -----
@st.cache_data
def load_dataset():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna(subset=["Global_Sales"])
    return df

df_games = load_dataset()

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
st.caption("Built by HKS • ML + UI/UX • Advanced Edition")