import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NB Master: Interactive ML", page_icon="🧪")

# --- CUSTOM CSS (Matching Screenshot 2026-05-04 111832.png) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Hero Banner Styling */
    .hero-banner {
        background-color: #3b66ad;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #4a7bc7;
    }
    .hero-title {
        color: white !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        color: #e0e0e0 !important;
        font-size: 18px;
        font-weight: 400;
    }

    /* Welcome Container Styling */
    .welcome-container {
        background-color: #262730;
        padding: 35px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 1px solid #31333f;
    }
    .welcome-header {
        color: #4facfe;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .welcome-text {
        color: #b0b2b8 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    .welcome-journey {
        color: #4facfe;
        font-weight: 700;
    }

    /* Card Styling */
    .main-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        text-align: left;
        border: 1px solid #e0e0e0;
        min-height: 250px;
        transition: transform 0.3s;
        margin-bottom: 15px;
    }
    .main-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 10px 20px rgba(255,255,255,0.1); 
    }
    .card-title { color: #1e1e1e !important; font-size: 20px; font-weight: 700; margin-top: 15px; }
    .card-text { color: #555555 !important; font-size: 14px; margin-bottom: 15px; }
    
    /* Navigation Buttons */
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    
    /* Educational Layouts */
    .edu-box { background-color: #1a1c24; padding: 20px; border-left: 5px solid #4facfe; border-radius: 5px; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
state_vars = {
    'page': "Dashboard",
    'raw_data': None,
    'proc_data': None,
    'model': None,
    'X_meta': None,
    'target_col': None,
    'features': [],
    'test_data': None,
    'variant': "GaussianNB",
    'split': 0.2,
    'cv_k': 5
}

for key, value in state_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

def navigate(p): st.session_state.page = p

# --- 1. DASHBOARD MODULE ---
def show_dashboard():
    # Hero Banner
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Learning System for Naive Bayes Classification</div>
            <div class="hero-subtitle">A Comprehensive Pipeline for Probabilistic Classification</div>
        </div>
    """, unsafe_allow_html=True)

    # Welcome Section
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-header">Welcome to the Machine Learning Laboratory</div>
            <p class="welcome-text">
                This workspace provides an end-to-end interface to explore the <b>Naive Bayes algorithm</b>. 
                Based on Bayes' Theorem, this classifier is famous for its "Naive" assumption of feature independence, 
                allowing it to perform remarkably well on high-dimensional datasets like text and medical diagnostics.
            </p>
            <p class="welcome-text">
                <span class="welcome-journey">Your Journey:</span> Use the modules below to upload raw data, perform preprocessing, 
                analyze statistical priors, train a model, and ultimately peek into the "brain" of the AI through a live 
                mathematical trace of its predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 3x3 Grid
    modules = [
        ("📁", "Dataset Upload", "Upload your CSV dataset to begin the learning journey.", "Dataset"),
        ("🛠️", "Preprocessing", "Clean and prepare your features for probabilistic calculation.", "Preprocessing"),
        ("📊", "Exploratory Analysis", "Explore data patterns, class priors, and relationships.", "EDA"),
        ("📖", "NB Learning", "Understand the math: Prior, Likelihood, and Independence.", "Learning"),
        ("⚙️", "Training Config", "Configure Train-Test splits and K-Fold cross-validation.", "Config"),
        ("🎯", "Model Training", "Execute the fit algorithm and view learned parameters.", "Training"),
        ("🔮", "Prediction Engine", "Input new samples to see a live mathematical inference trace.", "Prediction"),
        ("📈", "Evaluation Metrics", "Analyze Accuracy, F1-Score, and the Confusion Matrix.", "Evaluation"),
        ("📚", "Docs & Theory", "Access the mathematical intuition and code snippets.", "Docs")
    ]
    
    rows = [st.columns(3) for _ in range(3)]
    idx = 0
    for r in range(3):
        for c in range(3):
            icon, title, desc, page_id = modules[idx]
            with rows[r][c]:
                st.markdown(f"""<div class="main-card">
                    <div style="font-size: 35px;">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-text">{desc}</div>
                </div>""", unsafe_allow_html=True)
                st.button(f"Open {title}", key=f"btn_{page_id}", on_click=navigate, args=(page_id,))
            idx += 1

# --- 2. DATASET INPUT ---
def show_dataset():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    st.header("📁 Dataset Input Module")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        st.session_state.raw_data = df
        st.success("Dataset Uploaded!")
        st.write("**Preview (Head):**")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# --- 3. PREPROCESSING ---
def show_preprocessing():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    if st.session_state.raw_data is None:
        st.warning("Please upload a dataset first.")
        return
    
    st.header("🛠️ Preprocessing Interface")
    df = st.session_state.raw_data.copy()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.target_col = st.selectbox("Target Label", df.columns, index=len(df.columns)-1)
        st.session_state.features = st.multiselect("Features", [c for c in df.columns if c != st.session_state.target_col], default=[c for c in df.columns if c != st.session_state.target_col])
        impute = st.checkbox("Impute Missing Values")
        encode = st.checkbox("Label Encoding")
        scale = st.checkbox("Standard Scaling")
        
        if st.button("Run Pipeline"):
            if impute:
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mean()) if is_numeric_dtype(df[col]) else df[col].fillna(df[col].mode()[0])
            if encode:
                le = LabelEncoder()
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            if scale and st.session_state.features:
                df[st.session_state.features] = StandardScaler().fit_transform(df[st.session_state.features])
            
            st.session_state.proc_data = df
            st.success("Processing Complete!")

    with c2:
        st.subheader("Data Comparison")
        if st.session_state.proc_data is not None:
            st.dataframe(st.session_state.proc_data.head(10))

# --- 4. EDA ---
def show_eda():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    if st.session_state.proc_data is None: return
    st.header("📊 Exploratory Analysis")
    df = st.session_state.proc_data
    fig, ax = plt.subplots(); sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax); st.pyplot(fig)

# --- 5. LEARNING (MATH) ---
def show_learning():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    st.header("📖 Naive Bayes Learning Module")
    st.latex(r"P(C|x_1, ..., x_n) \propto P(C) \prod_{i=1}^{n} P(x_i|C)")
    st.markdown("""
    **Step-by-Step Logic:**
    1. **Prior:** $P(Class)$ based on training distribution.
    2. **Likelihood:** $P(Feature|Class)$ using Gaussian or Multinomial PDF.
    3. **Evidence:** Normalizing constant (often ignored in classification).
    """)

# --- 6. CONFIG ---
def show_config():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    st.header("⚙️ Training Configuration")
    st.session_state.variant = st.selectbox("NB Variant", ["GaussianNB", "MultinomialNB"])
    st.session_state.split = st.slider("Test Ratio", 0.1, 0.5, 0.2)
    st.session_state.cv_k = st.number_input("K-Fold", 2, 10, 5)

# --- 7. TRAINING ---
def show_training():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    if st.session_state.proc_data is None: return
    df = st.session_state.proc_data
    X = df[st.session_state.features]; y = df[st.session_state.target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.split)
    
    model = GaussianNB() if st.session_state.variant == "GaussianNB" else MultinomialNB()
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.X_meta = X
    st.session_state.test_data = (X_test, y_test)
    st.success("Model Trained Successfully!")

# --- 8. PREDICTION ---
def show_prediction():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    if st.session_state.model is None: return
    X = st.session_state.X_meta
    inputs = [st.number_input(f"{f}", value=float(X[f].mean())) for f in st.session_state.features]
    if st.button("Predict"):
        prob = st.session_state.model.predict_proba([inputs])
        st.write(f"**Result:** {st.session_state.model.predict([inputs])[0]}")
        st.write("**Posterior Probabilities:**", prob)

# --- 9. EVALUATION ---
def show_evaluation():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    if st.session_state.test_data is None: return
    X_test, y_test = st.session_state.test_data
    y_pred = st.session_state.model.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    fig, ax = plt.subplots(); sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=ax); st.pyplot(fig)

# --- 10. DOCS ---
def show_docs():
    st.button("⬅️ Back to Dashboard", on_click=navigate, args=("Dashboard",))
    st.header("📚 Docs & Theory")
    st.info("The 'Naive' part assumes that features are independent. While rarely true in real life, it works exceptionally well for high-dimensional data.")

# --- ROUTER ---
pages = {
    "Dashboard": show_dashboard, "Dataset": show_dataset, "Preprocessing": show_preprocessing,
    "EDA": show_eda, "Learning": show_learning, "Config": show_config,
    "Training": show_training, "Prediction": show_prediction, "Evaluation": show_evaluation, "Docs": show_docs
}
pages[st.session_state.page]()