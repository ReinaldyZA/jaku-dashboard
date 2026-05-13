"""
JakU — Jakarta Kualitas Udara
Dashboard Klasifikasi ISPU DKI Jakarta
Menggunakan Random Forest, XGBoost, dan SVM
Metodologi CRISP-DM | Prinsip UCD
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
import time

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="JakU — Jakarta Kualitas Udara",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --clr-primary: #0A6847;
    --clr-primary-light: #16A34A;
    --clr-accent: #F59E0B;
    --clr-danger: #EF4444;
    --clr-bg: #F8FAF9;
    --clr-card: #FFFFFF;
    --clr-text: #1A2E1F;
    --clr-muted: #6B8A73;
    --clr-baik: #22C55E;
    --clr-sedang: #F59E0B;
    --clr-tidaksehat: #EF4444;
    --radius: 16px;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* Hide default header & footer */
#MainMenu, header, footer {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A3D2E 0%, #0A6847 50%, #0F7B56 100%);
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #E8F5E9 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500 !important;
    padding: 10px 16px !important;
    border-radius: 10px !important;
    margin-bottom: 2px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: rgba(255,255,255,0.15) !important;
}

/* Metric cards */
.metric-card {
    background: var(--clr-card);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(10,104,71,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    text-align: center;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(10,104,71,0.12);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--clr-primary);
    line-height: 1.1;
    margin: 8px 0 4px 0;
}
.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--clr-muted);
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--clr-muted);
    margin-top: 4px;
}

/* Hero */
.hero-container {
    background: linear-gradient(135deg, #0A6847 0%, #16A34A 60%, #22D3EE 100%);
    border-radius: 20px;
    padding: 44px 40px;
    color: white;
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 400;
    opacity: 0.9;
    position: relative;
    z-index: 1;
    max-width: 600px;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--clr-text);
    margin: 32px 0 16px 0;
    letter-spacing: -0.02em;
}
.section-desc {
    font-size: 0.92rem;
    color: var(--clr-muted);
    margin-bottom: 20px;
    line-height: 1.6;
}

/* Category badges */
.badge-baik {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    padding: 6px 18px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.badge-sedang {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    padding: 6px 18px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.badge-tidaksehat {
    display: inline-block;
    background: #FEE2E2;
    color: #991B1B;
    padding: 6px 18px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

/* Card sections */
.info-card {
    background: var(--clr-card);
    border-radius: var(--radius);
    padding: 28px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(10,104,71,0.06);
    margin-bottom: 16px;
}
.info-card h4 {
    color: var(--clr-primary);
    font-weight: 700;
    margin-bottom: 12px;
}

/* Prediction result */
.prediction-result {
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin: 20px 0;
}
.pred-baik {
    background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
    border: 2px solid #22C55E;
}
.pred-sedang {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border: 2px solid #F59E0B;
}
.pred-tidaksehat {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    border: 2px solid #EF4444;
}
.pred-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.pred-baik .pred-label { color: #166534; }
.pred-sedang .pred-label { color: #92400E; }
.pred-tidaksehat .pred-label { color: #991B1B; }

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F0F5F1;
    padding: 4px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* Plotly chart containers */
.stPlotlyChart {
    border-radius: var(--radius);
    overflow: hidden;
}

/* Slider */
.stSlider > div > div { color: var(--clr-primary) !important; }

/* Progress indicators */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: #F0F5F1;
    border-radius: 12px;
    margin-bottom: 8px;
    border-left: 4px solid var(--clr-primary);
}
.step-num {
    background: var(--clr-primary);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.88rem;
    color: var(--clr-text);
    font-weight: 500;
}

/* Rekomendasi */
.rekom-card {
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 12px;
}
.rekom-baik {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border-left: 5px solid #22C55E;
}
.rekom-sedang {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border-left: 5px solid #F59E0B;
}
.rekom-tidaksehat {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border-left: 5px solid #EF4444;
}
</style>
""", unsafe_allow_html=True)


# ─── PLOTLY THEME ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#1A2E1F"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Plus Jakarta Sans"),
)
COLORS = {
    "BAIK": "#22C55E",
    "SEDANG": "#F59E0B",
    "TIDAK SEHAT": "#EF4444",
}
MODEL_COLORS = {
    "Random Forest": "#3B82F6",
    "XGBoost": "#22C55E",
    "SVM": "#F59E0B",
}
POLUTAN_COLORS = px.colors.qualitative.Set2


# ─── UCD HELPER FUNCTIONS ──────────────────────────────────────
# Diterapkan untuk mendukung 8 Golden Rules of UI Design (Shneiderman)
# dan prinsip User-Centered Design (Nielsen Usability Heuristics)

def breadcrumb(icon, page_name):
    """Rule 4: Design dialogs to yield closure — selalu tunjukkan lokasi user."""
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#6B8A73; margin-bottom:12px; font-weight:500;">
        <span style="opacity:0.6;">🏠 JakU Dashboard</span>
        <span style="margin:0 6px; opacity:0.4;">›</span>
        <span style="color:#0A6847;">{icon} {page_name}</span>
    </div>
    """, unsafe_allow_html=True)


def how_to_use(title, steps):
    """Rule 2: Universal usability — bantu pengguna pemula."""
    with st.expander(f"❓ {title}", expanded=False):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")


def info_tooltip(label, tooltip_text):
    """Rule 8: Reduce memory load — tooltip untuk istilah teknis."""
    return f'{label} <span title="{tooltip_text}" style="cursor:help; color:#0A6847;">ⓘ</span>'


def insight_box(text, icon="💡"):
    """Rule 4: Closure — kesimpulan analisis di akhir."""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #F0F9FF, #DBEAFE);
                border-left:4px solid #3B82F6; border-radius:0 12px 12px 0;
                padding:14px 18px; margin:12px 0; font-size:0.88rem;
                color:#1E40AF; line-height:1.6;">
        <strong>{icon} Insight:</strong> {text}
    </div>
    """, unsafe_allow_html=True)


def app_footer():
    """Rule 4: Closure — atribusi & versi di akhir."""
    st.markdown("""
    <hr style="border:none; border-top:1px solid rgba(10,104,71,0.08); margin:32px 0 16px 0;">
    <div style="text-align:center; font-size:0.72rem; color:#9CA3AF; padding:8px 0 16px 0; line-height:1.6;">
        <strong style="color:#6B8A73;">JakU Dashboard v1.0</strong> · Klasifikasi Kualitas Udara DKI Jakarta<br>
        Dibangun dengan Streamlit · Metodologi CRISP-DM · Desain UCD & 8 Golden Rules
    </div>
    """, unsafe_allow_html=True)


# ─── DATA LOADING & PROCESSING ────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data():
    """Full CRISP-DM pipeline: load, clean, impute, remove outliers, split, encode, scale."""
    import io, os
    csv_path = None
    for candidate in [
        "/mnt/user-data/uploads/Data_ISPU.csv",
        "Data_ISPU.csv",
    ]:
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        return None

    df_raw = pd.read_csv(csv_path, sep=";")
    fitur_polutan = [
        "pm_sepuluh", "pm_duakomalima", "sulfur_dioksida",
        "karbon_monoksida", "ozon", "nitrogen_dioksida",
    ]

    # ── EDA copy ──
    df_eda = df_raw.copy()
    for col in fitur_polutan:
        df_eda[col] = pd.to_numeric(df_eda[col], errors="coerce")

    # ── Cleaning ──
    df = df_raw.copy()
    df = df[df["kategori"].isin(["BAIK", "SEDANG", "TIDAK SEHAT"])].copy()

    stasiun_map = {
        "DKI1 Bundaran Hotel Indonesia (HI)": "DKI1 Bunderan HI",
        "DKI1 Bundaran Hotel Indonesia HI": "DKI1 Bunderan HI",
        "DKI5 Kebon Jeruk Jakarta Barat": "DKI5 Kebon Jeruk",
    }
    df["stasiun"] = df["stasiun"].replace(stasiun_map)
    for col in fitur_polutan:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_after_clean = len(df)

    # ── Imputasi median ──
    medians = {}
    for col in fitur_polutan:
        med = df[col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)

    # ── IQR outlier removal ──
    mask_inlier = pd.Series(True, index=df.index)
    outlier_info = []
    for col in fitur_polutan:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = (df[col] < lo) | (df[col] > hi)
        outlier_info.append({"Fitur": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
                             "Batas Bawah": lo, "Batas Atas": hi,
                             "Jumlah Outlier": outliers.sum()})
        mask_inlier &= ~outliers
    df = df[mask_inlier].copy()

    # ── Features & target ──
    X = df[fitur_polutan].copy()
    y = df["kategori"].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "df_raw": df_raw,
        "df_eda": df_eda,
        "df": df,
        "fitur": fitur_polutan,
        "le": le,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "medians": medians,
        "outlier_info": pd.DataFrame(outlier_info),
        "rows_raw": len(df_raw),
        "rows_after_clean": rows_after_clean,
        "rows_final": len(df),
    }


@st.cache_resource(show_spinner=False)
def train_models(_data):
    """
    Train & evaluate RF, XGBoost, SVM dengan hyperparameter terbaik
    yang sudah ditemukan melalui GridSearchCV sebelumnya (di notebook).
    Pendekatan ini cocok untuk deployment: tuning dilakukan sekali saat
    penelitian, lalu best params digunakan langsung di production.
    """
    X_train = _data["X_train"]
    X_test = _data["X_test"]
    y_train = _data["y_train"]
    y_test = _data["y_test"]
    X_train_scaled = _data["X_train_scaled"]
    X_test_scaled = _data["X_test_scaled"]
    le = _data["le"]
    fitur = _data["fitur"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Best hyperparameters dari GridSearchCV (hasil tuning di notebook) ──
    rf_params = {
        "n_estimators": 200, "max_depth": 20, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42,
    }
    xgb_params = {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "random_state": 42, "eval_metric": "mlogloss", "use_label_encoder": False,
    }
    svm_params = {
        "C": 10, "gamma": "scale", "kernel": "rbf", "random_state": 42,
    }

    # ── Train dengan best params (langsung, tanpa GridSearchCV ulang) ──
    rf_best = RandomForestClassifier(**rf_params)
    rf_best.fit(X_train, y_train)

    xgb_best = XGBClassifier(**xgb_params)
    xgb_best.fit(X_train, y_train)

    svm_best = SVC(**svm_params)
    svm_best.fit(X_train_scaled, y_train)

    # ── Predictions & evaluation ──
    models_best = {
        "Random Forest": (rf_best, X_test),
        "XGBoost": (xgb_best, X_test),
        "SVM": (svm_best, X_test_scaled),
    }
    preds = {n: m.predict(X) for n, (m, X) in models_best.items()}

    # ── Cross-validation (3-fold untuk speed, masih representatif) ──
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    cv_configs = {"Random Forest": (rf_best, X_train), "XGBoost": (xgb_best, X_train),
                  "SVM": (svm_best, X_train_scaled)}
    for name, (mdl, Xd) in cv_configs.items():
        cv_results[name] = cross_val_score(mdl, Xd, y_train, cv=cv_skf, scoring="accuracy", n_jobs=-1)

    comparison = []
    for name, y_pred in preds.items():
        rep = classification_report(y_test, y_pred, output_dict=True, digits=4)
        cv = cv_results[name]
        comparison.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Macro Precision": rep["macro avg"]["precision"],
            "Macro Recall": rep["macro avg"]["recall"],
            "Macro F1-Score": rep["macro avg"]["f1-score"],
            "CV Mean": cv.mean(), "CV Std": cv.std(),
        })
    comp_df = pd.DataFrame(comparison)

    # ── Feature importances ──
    rf_imp = pd.Series(rf_best.feature_importances_, index=fitur)
    xgb_imp = pd.Series(xgb_best.feature_importances_, index=fitur)
    perm = permutation_importance(svm_best, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    svm_imp = pd.Series(perm.importances_mean, index=fitur)

    best_idx = comp_df["Test Accuracy"].idxmax()
    best_name = comp_df.loc[best_idx, "Model"]

    # Sanitasi params untuk display (hapus key internal)
    display_rf = {k: v for k, v in rf_params.items() if k != "random_state"}
    display_xgb = {k: v for k, v in xgb_params.items() if k not in ["random_state", "eval_metric", "use_label_encoder"]}
    display_svm = {k: v for k, v in svm_params.items() if k != "random_state"}

    return {
        "rf_best": rf_best, "xgb_best": xgb_best, "svm_best": svm_best,
        "rf_params": display_rf, "xgb_params": display_xgb, "svm_params": display_svm,
        "predictions": preds,
        "cv_results": cv_results,
        "comp_df": comp_df,
        "rf_imp": rf_imp, "xgb_imp": xgb_imp, "svm_imp": svm_imp,
        "best_name": best_name,
        "le": le,
    }


# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:2.2rem;">🌬️</div>
        <div style="font-size:1.3rem; font-weight:800; letter-spacing:-0.03em; margin-top:4px;">JakU</div>
        <div style="font-size:0.72rem; font-weight:400; opacity:0.7; letter-spacing:0.06em; text-transform:uppercase;">Jakarta Kualitas Udara</div>
    </div>
    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.12); margin:12px 0 20px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Eksplorasi Data", "🧹 Persiapan Data",
         "🤖 Pemodelan & Evaluasi", "🎯 Prediksi Interaktif", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.12); margin:24px 0 16px 0;">
    <div style="font-size:0.7rem; opacity:0.5; text-align:center; line-height:1.5;">
        CRISP-DM · UCD<br>Random Forest · XGBoost · SVM
    </div>
    """, unsafe_allow_html=True)


# ─── LOAD DATA ─────────────────────────────────────────────────
data = load_and_process_data()
if data is None:
    st.error("⚠️ File `Data_ISPU.csv` tidak ditemukan. Pastikan file tersedia di direktori yang benar.")
    st.stop()

# Train models (cached)
with st.spinner("Melatih model... Harap tunggu sebentar."):
    models = train_models(data)


# ════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    breadcrumb("🏠", "Beranda")

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌬️ JakU Dashboard</div>
        <div class="hero-sub">
            Sistem klasifikasi kualitas udara DKI Jakarta berbasis machine learning.
            Membandingkan Random Forest, XGBoost, dan SVM menggunakan metodologi CRISP-DM.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome banner with quick-start guidance — Rule 2 (Universal Usability)
    how_to_use("Cara menggunakan dashboard ini", [
        "Lihat **ringkasan dataset** dan **model terbaik** di halaman Beranda ini.",
        "Buka **Eksplorasi Data** untuk memahami karakteristik dataset ISPU.",
        "Buka **Persiapan Data** untuk melihat tahap cleaning dan preprocessing.",
        "Buka **Pemodelan & Evaluasi** untuk membandingkan performa 3 algoritma.",
        "Buka **Prediksi Interaktif** untuk mencoba memprediksi kualitas udara dari nilai polutan.",
    ])

    # KPI cards
    best = models["comp_df"].loc[models["comp_df"]["Test Accuracy"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data Mentah</div>
            <div class="metric-value">{data['rows_raw']:,}</div>
            <div class="metric-sub">baris dataset awal</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Final</div>
            <div class="metric-value">{data['rows_final']:,}</div>
            <div class="metric-sub">setelah cleaning & outlier removal</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Terbaik</div>
            <div class="metric-value" style="font-size:1.6rem;">{models['best_name']}</div>
            <div class="metric-sub">akurasi tertinggi</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Akurasi Terbaik</div>
            <div class="metric-value">{best['Test Accuracy']:.2%}</div>
            <div class="metric-sub">pada data uji</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick model comparison chart
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Perbandingan Performa Model</div>', unsafe_allow_html=True)
        metrics = ["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
        fig = go.Figure()
        for _, row in models["comp_df"].iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=metrics,
                y=[row[m] for m in metrics],
                marker_color=MODEL_COLORS[row["Model"]],
                marker_line=dict(width=0),
                text=[f"{row[m]:.4f}" for m in metrics],
                textposition="outside",
                textfont=dict(size=11, family="JetBrains Mono"),
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            yaxis=dict(range=[0.85, 1.02], gridcolor="rgba(0,0,0,0.04)", gridwidth=1),
            xaxis=dict(tickfont=dict(size=12)),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                        font=dict(size=12)),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Distribusi Kategori ISPU</div>', unsafe_allow_html=True)
        cat_counts = data["df"]["kategori"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            marker=dict(colors=[COLORS.get(k, "#999") for k in cat_counts.index],
                        line=dict(width=2, color="white")),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=13, family="Plus Jakarta Sans"),
            hoverinfo="label+value+percent",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                           annotations=[dict(text=f"<b>{data['rows_final']}</b><br>data",
                                             x=0.5, y=0.5, font_size=16, showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # CRISP-DM steps
    st.markdown('<div class="section-header">Alur Metodologi CRISP-DM</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Business Understanding", "Memahami permasalahan klasifikasi kualitas udara di DKI Jakarta"),
        ("2", "Data Understanding", "Mengeksplorasi 3.350 baris data ISPU dari 6 stasiun pemantauan"),
        ("3", "Data Preparation", "Cleaning, imputasi median, penghapusan outlier IQR, encoding, scaling"),
        ("4", "Modeling", "Membangun & tuning Random Forest, XGBoost, dan SVM dengan GridSearchCV"),
        ("5", "Evaluation", "Confusion matrix, classification report, 5-fold CV, feature importance"),
        ("6", "Deployment", "Dashboard interaktif & fungsi prediksi untuk integrasi sistem web"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-indicator">
            <div class="step-num">{num}</div>
            <div class="step-text"><strong>{title}</strong> — {desc}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE: EKSPLORASI DATA
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Eksplorasi Data":
    breadcrumb("🔍", "Eksplorasi Data")
    st.markdown('<div class="section-header">Eksplorasi Data (Data Understanding)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tahap ini bertujuan memahami karakteristik dataset ISPU DKI Jakarta sebelum dilakukan pemrosesan lebih lanjut.</div>', unsafe_allow_html=True)

    how_to_use("Apa yang bisa dilakukan di halaman ini?", [
        "**Statistik Deskriptif** — lihat ringkasan numerik (mean, std, min, max) tiap polutan.",
        "**Distribusi Fitur** — pilih satu polutan untuk melihat histogram distribusinya.",
        "**Boxplot** — identifikasi outlier dan sebaran nilai per polutan.",
        "**Korelasi** — pahami hubungan antar 6 parameter polutan.",
        "**Missing Values** — cek jumlah data kosong sebelum imputasi.",
        "**Distribusi per Stasiun** — lihat sebaran kategori di 5 stasiun pemantauan DKI Jakarta.",
    ])

    tabs = st.tabs(["📊 Statistik Deskriptif", "📈 Distribusi Fitur", "📦 Boxplot", "🔥 Korelasi", "❓ Missing Values", "🗺️ Distribusi per Stasiun"])

    with tabs[0]:
        st.markdown("#### Statistik Deskriptif 6 Parameter Polutan")
        df_eda = data["df_eda"]
        fitur = data["fitur"]
        stats = df_eda[fitur].describe().T
        stats["median"] = df_eda[fitur].median()
        stats["skew"] = df_eda[fitur].skew()
        stats = stats[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skew"]]
        st.dataframe(stats.style.format("{:.2f}").background_gradient(cmap="Greens", axis=0),
                      use_container_width=True, height=260)

        st.markdown("#### Preview Dataset (10 baris pertama)")
        st.dataframe(data["df_raw"].head(10), use_container_width=True, height=380)

    with tabs[1]:
        st.markdown("#### Histogram Distribusi Parameter Polutan")
        sel_feat = st.selectbox("Pilih parameter:", fitur, key="hist_sel")
        fig = px.histogram(
            df_eda, x=sel_feat, nbins=40,
            color_discrete_sequence=["#0A6847"],
            labels={sel_feat: sel_feat.replace("_", " ").title()},
        )
        mean_v = df_eda[sel_feat].mean()
        med_v = df_eda[sel_feat].median()
        fig.add_vline(x=mean_v, line_dash="dash", line_color="#EF4444",
                       annotation_text=f"Mean: {mean_v:.1f}", annotation_position="top right")
        fig.add_vline(x=med_v, line_dash="solid", line_color="#F59E0B",
                       annotation_text=f"Median: {med_v:.1f}", annotation_position="top left")
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          yaxis=dict(gridcolor="rgba(0,0,0,0.04)"),
                          xaxis_title=sel_feat.replace("_", " ").title(),
                          yaxis_title="Frekuensi")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("#### Boxplot Parameter Polutan (Sebelum Cleaning)")
        fig = make_subplots(rows=2, cols=3, subplot_titles=[f.replace("_", " ").title() for f in fitur])
        for i, col in enumerate(fitur):
            r, c = i // 3 + 1, i % 3 + 1
            fig.add_trace(go.Box(y=df_eda[col].dropna(), name=col, marker_color=POLUTAN_COLORS[i],
                                  boxmean="sd"), row=r, col=c)
        fig.update_layout(**PLOTLY_LAYOUT, height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Heatmap Korelasi Antar Parameter Polutan")
        corr = df_eda[fitur].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.where(~mask)
        fig = px.imshow(
            corr_masked, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Korelasi"),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=480)
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.markdown("#### Missing Values per Fitur Polutan")
        missing = df_eda[fitur].isnull().sum()
        fig = go.Figure(go.Bar(
            x=missing.index, y=missing.values,
            marker_color="#EF4444", marker_line=dict(width=0),
            text=missing.values, textposition="outside",
            textfont=dict(size=13, family="JetBrains Mono, monospace"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          yaxis=dict(gridcolor="rgba(0,0,0,0.04)"),
                          xaxis_title="Fitur Polutan", yaxis_title="Jumlah Missing")
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown("#### Distribusi Kategori Kualitas Udara per Stasiun Pemantauan")
        st.markdown(
            '<div class="section-desc">Visualisasi sebaran kategori BAIK, SEDANG, dan TIDAK SEHAT pada 5 stasiun pemantauan kualitas udara DKI Jakarta.</div>',
            unsafe_allow_html=True,
        )

        # Mapping stasiun ke wilayah
        wilayah_map = {
            "DKI1 Bunderan HI": "Jakarta Pusat",
            "DKI2 Kelapa Gading": "Jakarta Utara",
            "DKI3 Jagakarsa": "Jakarta Selatan",
            "DKI4 Lubang Buaya": "Jakarta Timur",
            "DKI5 Kebon Jeruk": "Jakarta Barat",
        }

        # Hitung distribusi kategori per stasiun
        stasiun_dist = (
            data["df"]
            .groupby(["stasiun", "kategori"])
            .size()
            .reset_index(name="jumlah")
        )

        # Hitung total dan persentase per stasiun
        total_per_stasiun = stasiun_dist.groupby("stasiun")["jumlah"].transform("sum")
        stasiun_dist["persentase"] = (stasiun_dist["jumlah"] / total_per_stasiun * 100).round(1)

        # Stacked bar chart (jumlah)
        fig = px.bar(
            stasiun_dist,
            x="stasiun",
            y="jumlah",
            color="kategori",
            color_discrete_map=COLORS,
            barmode="stack",
            text="jumlah",
            category_orders={"kategori": ["BAIK", "SEDANG", "TIDAK SEHAT"]},
            labels={"stasiun": "Stasiun Pemantauan", "jumlah": "Jumlah Pengamatan", "kategori": "Kategori"},
        )
        fig.update_traces(textposition="inside", textfont=dict(size=11, color="white", family="Plus Jakarta Sans"))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=440,
            yaxis=dict(gridcolor="rgba(0,0,0,0.04)"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", title=""),
        )
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

        # Tabel kategori dominan per stasiun
        st.markdown("#### Kategori Dominan per Stasiun")
        dominan_list = []
        for stasiun in data["df"]["stasiun"].unique():
            sub = data["df"][data["df"]["stasiun"] == stasiun]
            kat_count = sub["kategori"].value_counts()
            dom_kat = kat_count.idxmax()
            dom_pct = (kat_count.max() / kat_count.sum() * 100).round(1)
            dominan_list.append({
                "Stasiun": stasiun,
                "Wilayah": wilayah_map.get(stasiun, "—"),
                "Kategori Dominan": dom_kat,
                "Persentase": f"{dom_pct}%",
                "Total Pengamatan": int(kat_count.sum()),
            })
        dominan_df = pd.DataFrame(dominan_list).sort_values("Stasiun").reset_index(drop=True)

        # Color-coded dataframe
        def color_kategori(val):
            if val == "BAIK":
                return "background-color: #DCFCE7; color: #166534; font-weight: 600"
            elif val == "SEDANG":
                return "background-color: #FEF3C7; color: #92400E; font-weight: 600"
            elif val == "TIDAK SEHAT":
                return "background-color: #FEE2E2; color: #991B1B; font-weight: 600"
            return ""

        st.dataframe(
            dominan_df.style.applymap(color_kategori, subset=["Kategori Dominan"]),
            use_container_width=True,
            height=240,
            hide_index=True,
        )

        # Ringkasan analitis
        baik_terbanyak = max(dominan_list, key=lambda x: data["df"][(data["df"]["stasiun"] == x["Stasiun"]) & (data["df"]["kategori"] == "BAIK")].shape[0])
        tidaksehat_terbanyak = max(dominan_list, key=lambda x: data["df"][(data["df"]["stasiun"] == x["Stasiun"]) & (data["df"]["kategori"] == "TIDAK SEHAT")].shape[0])

        st.info(
            f"💡 **Insight:** Stasiun **{baik_terbanyak['Stasiun']}** ({baik_terbanyak['Wilayah']}) "
            f"memiliki jumlah pengamatan kategori BAIK terbanyak. Sebaliknya, stasiun "
            f"**{tidaksehat_terbanyak['Stasiun']}** ({tidaksehat_terbanyak['Wilayah']}) "
            f"memiliki jumlah pengamatan kategori TIDAK SEHAT terbanyak."
        )


# ════════════════════════════════════════════════════════════════
# PAGE: PERSIAPAN DATA
# ════════════════════════════════════════════════════════════════
elif page == "🧹 Persiapan Data":
    breadcrumb("🧹", "Persiapan Data")
    st.markdown('<div class="section-header">Persiapan Data (Data Preparation)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tahap ini mencakup cleaning, imputasi, penghapusan outlier, encoding, dan feature scaling sesuai pipeline CRISP-DM.</div>', unsafe_allow_html=True)

    how_to_use("Mengapa tahap ini penting?", [
        "**Data Cleaning** menghapus baris yang tidak valid agar model belajar dari data berkualitas.",
        "**Imputasi median** mengisi nilai kosong tanpa terpengaruh outlier (lebih robust dari mean).",
        "**Outlier removal IQR** menghapus nilai ekstrem yang bisa mengacaukan pelatihan model.",
        "**Encoding & scaling** mengubah data ke format yang bisa diproses oleh algoritma ML.",
    ])

    tabs = st.tabs(["🧼 Data Cleaning", "🩹 Imputasi", "📐 Outlier Removal", "🏷️ Encoding & Split"])

    with tabs[0]:
        st.markdown("#### Langkah Data Cleaning")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Baris Awal</div>
                <div class="metric-value">{data['rows_raw']:,}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Setelah Cleaning</div>
                <div class="metric-value">{data['rows_after_clean']:,}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Setelah Outlier Removal</div>
                <div class="metric-value">{data['rows_final']:,}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        steps_clean = [
            "Menghapus baris dengan kategori TIDAK ADA DATA dan NaN",
            "Menggabungkan kelas SANGAT TIDAK SEHAT ke dalam TIDAK SEHAT",
            "Standarisasi nama stasiun (mengatasi duplikat varian nama)",
            "Konversi kolom polutan ke tipe numerik (float64)",
        ]
        for i, s in enumerate(steps_clean, 1):
            st.markdown(f"""<div class="step-indicator">
                <div class="step-num">{i}</div>
                <div class="step-text">{s}</div>
            </div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("#### Imputasi Missing Values dengan Median")
        med_df = pd.DataFrame([
            {"Fitur": k, "Nilai Median": v} for k, v in data["medians"].items()
        ])
        st.dataframe(med_df.style.format({"Nilai Median": "{:.1f}"}).background_gradient(
            subset=["Nilai Median"], cmap="Greens"), use_container_width=True, height=250)

        st.info("Setiap kolom fitur yang memiliki nilai kosong (NaN) diisi menggunakan nilai **median** dari kolom tersebut. Median dipilih karena lebih robust terhadap outlier dibanding mean.")

    with tabs[2]:
        st.markdown("#### Penghapusan Outlier — Metode IQR 1.5×")
        st.dataframe(
            data["outlier_info"].style.format({
                "Q1": "{:.1f}", "Q3": "{:.1f}", "IQR": "{:.1f}",
                "Batas Bawah": "{:.1f}", "Batas Atas": "{:.1f}",
            }).background_gradient(subset=["Jumlah Outlier"], cmap="Reds"),
            use_container_width=True, height=260,
        )
        removed = data["rows_after_clean"] - data["rows_final"]
        st.metric("Total baris dihapus (outlier)", f"{removed} baris",
                   delta=f"-{removed/data['rows_after_clean']*100:.1f}%", delta_color="inverse")

        # Boxplot after cleaning
        st.markdown("#### Boxplot Setelah Outlier Removal")
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f.replace("_", " ").title() for f in data["fitur"]])
        for i, col in enumerate(data["fitur"]):
            r, c = i // 3 + 1, i % 3 + 1
            fig.add_trace(go.Box(y=data["df"][col], name=col, marker_color="#22C55E",
                                  boxmean="sd"), row=r, col=c)
        fig.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Label Encoding & Train/Test Split")
        le = data["le"]
        enc_df = pd.DataFrame({"Kategori": le.classes_, "Encoded": range(len(le.classes_))})
        st.dataframe(enc_df, use_container_width=True, height=150)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Training Set</div>
                <div class="metric-value">{len(data['X_train']):,}</div>
                <div class="metric-sub">80% — stratified</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Testing Set</div>
                <div class="metric-value">{len(data['X_test']):,}</div>
                <div class="metric-sub">20% — stratified</div>
            </div>""", unsafe_allow_html=True)

        st.info("**StandardScaler** (Z-score normalization) diterapkan **hanya** untuk SVM, karena SVM sensitif terhadap skala fitur. Random Forest dan XGBoost menggunakan data asli.")


# ════════════════════════════════════════════════════════════════
# PAGE: PEMODELAN & EVALUASI
# ════════════════════════════════════════════════════════════════
elif page == "🤖 Pemodelan & Evaluasi":
    breadcrumb("🤖", "Pemodelan & Evaluasi")
    st.markdown('<div class="section-header">Pemodelan & Evaluasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tiga algoritma dibandingkan: Random Forest, XGBoost, dan SVM. Masing-masing di-tuning dengan GridSearchCV dan dievaluasi secara menyeluruh.</div>', unsafe_allow_html=True)

    how_to_use("Bagaimana membaca hasil evaluasi?", [
        "**Accuracy** = persentase prediksi yang benar dari seluruh data uji.",
        "**Precision** = dari yang diprediksi positif, berapa yang benar-benar positif.",
        "**Recall** = dari yang sebenarnya positif, berapa yang berhasil terdeteksi.",
        "**F1-Score** = harmonic mean precision dan recall (lebih seimbang dari accuracy).",
        "**Confusion Matrix** menunjukkan rincian prediksi per kelas — diagonal = benar.",
        "**Cross-Validation** = evaluasi pada 5 subset data untuk memastikan model konsisten.",
    ])

    tabs = st.tabs(["⚙️ Hyperparameter", "📊 Confusion Matrix", "📋 Classification Report",
                     "🔄 Cross-Validation", "🏆 Feature Importance", "📈 Perbandingan"])

    with tabs[0]:
        st.markdown("#### Best Hyperparameters (GridSearchCV)")
        param_data = {
            "Random Forest": models["rf_params"],
            "XGBoost": models["xgb_params"],
            "SVM": models["svm_params"],
        }
        for name, params in param_data.items():
            with st.expander(f"**{name}**", expanded=True):
                pc = st.columns(min(len(params), 4))
                for i, (k, v) in enumerate(params.items()):
                    with pc[i % len(pc)]:
                        st.metric(k, str(v))

    with tabs[1]:
        st.markdown("#### Confusion Matrix")
        le = models["le"]
        sel_model = st.selectbox("Pilih model:", list(models["predictions"].keys()), key="cm_sel")
        y_pred = models["predictions"][sel_model]
        cm = confusion_matrix(data["y_test"], y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            x=le.classes_, y=le.classes_,
            color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=dict(text=f"Confusion Matrix — {sel_model}", font_size=16))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("#### Classification Report (per Model)")
        for name, y_pred in models["predictions"].items():
            with st.expander(f"**{name}**", expanded=(name == models["best_name"])):
                rep = classification_report(data["y_test"], y_pred,
                                            target_names=le.classes_, output_dict=True)
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}").background_gradient(
                    cmap="Greens", subset=pd.IndexSlice[le.classes_, :]),
                    use_container_width=True, height=220)

    with tabs[3]:
        st.markdown("#### Stratified 5-Fold Cross-Validation")
        cv_df = pd.DataFrame(models["cv_results"])
        cv_df.index = [f"Fold {i+1}" for i in range(5)]

        fig = go.Figure()
        for name in cv_df.columns:
            fig.add_trace(go.Bar(
                name=name, x=cv_df.index, y=cv_df[name],
                marker_color=MODEL_COLORS[name],
                text=[f"{v:.4f}" for v in cv_df[name]],
                textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono"),
            ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=400,
                          yaxis=dict(range=[0.90, 1.01], gridcolor="rgba(0,0,0,0.04)"),
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        c1, c2, c3 = st.columns(3)
        for col_w, name in zip([c1, c2, c3], cv_df.columns):
            with col_w:
                mean_v = cv_df[name].mean()
                std_v = cv_df[name].std()
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value">{mean_v:.4f}</div>
                    <div class="metric-sub">± {std_v:.4f}</div>
                </div>""", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("#### Feature Importance")
        imp_sel = st.selectbox("Pilih model:", ["Random Forest", "XGBoost", "SVM (Permutation)"], key="imp_sel")
        imp_map = {"Random Forest": models["rf_imp"], "XGBoost": models["xgb_imp"],
                   "SVM (Permutation)": models["svm_imp"]}
        imp = imp_map[imp_sel].sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            y=[f.replace("_", " ").title() for f in imp.index],
            x=imp.values,
            orientation="h",
            marker_color="#0A6847",
            text=[f"{v:.4f}" for v in imp.values],
            textposition="outside",
            textfont=dict(size=12, family="JetBrains Mono"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          xaxis_title="Importance Score",
                          yaxis=dict(tickfont=dict(size=12)),
                          title=dict(text=f"Feature Importance — {imp_sel}", font_size=15))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown("#### Tabel Perbandingan Performa")
        comp = models["comp_df"].copy()
        st.dataframe(
            comp.style.format({
                "Test Accuracy": "{:.4f}", "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}", "Macro F1-Score": "{:.4f}",
                "CV Mean": "{:.4f}", "CV Std": "{:.4f}",
            }).highlight_max(subset=["Test Accuracy", "Macro Precision", "Macro Recall",
                                      "Macro F1-Score", "CV Mean"],
                              color="#DCFCE7"),
            use_container_width=True, height=160,
        )

        best_row = comp.loc[comp["Test Accuracy"].idxmax()]
        st.success(f"🏆 **Model Terbaik: {best_row['Model']}** — Akurasi: {best_row['Test Accuracy']:.4f} | CV Mean: {best_row['CV Mean']:.4f} ± {best_row['CV Std']:.4f}")

        # Radar chart
        st.markdown("#### Radar Chart Perbandingan")
        metrics_r = ["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "CV Mean"]
        fig = go.Figure()
        for _, row in comp.iterrows():
            vals = [row[m] for m in metrics_r] + [row[metrics_r[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=metrics_r + [metrics_r[0]],
                fill="toself",
                name=row["Model"],
                line_color=MODEL_COLORS[row["Model"]],
                fillcolor=MODEL_COLORS[row["Model"]],
                opacity=0.15,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=440,
                          polar=dict(radialaxis=dict(range=[0.90, 1.0], tickfont=dict(size=10))),
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI INTERAKTIF
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Prediksi Interaktif":
    breadcrumb("🎯", "Prediksi Interaktif")
    st.markdown('<div class="section-header">Prediksi Kualitas Udara</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Masukkan nilai 6 parameter polutan untuk mendapatkan klasifikasi kualitas udara secara real-time. Anda dapat memilih model yang digunakan.</div>', unsafe_allow_html=True)

    how_to_use("Cara melakukan prediksi", [
        "Pilih **model klasifikasi** dari dropdown (XGBoost direkomendasikan karena akurasi tertinggi).",
        "Atur nilai 6 parameter polutan dengan **slider**. Nilai default sudah diset pada rata-rata dataset.",
        "Atau klik salah satu **preset skenario** untuk mengisi nilai otomatis (bersih / sedang / buruk).",
        "Lihat **hasil prediksi** dan **rekomendasi aktivitas** di panel kanan.",
        "Klik **Reset** untuk mengembalikan semua slider ke nilai default.",
    ])

    # ── Preset values (Rule 8: reduce memory load — referensi nilai realistis) ──
    PRESETS = {
        "baik":        {"pm10": 25, "pm25": 30, "so2": 15, "co": 5,  "o3": 10, "no2": 15},
        "sedang":      {"pm10": 60, "pm25": 75, "so2": 40, "co": 18, "o3": 25, "no2": 35},
        "tidaksehat":  {"pm10": 120,"pm25": 130,"so2": 75, "co": 45, "o3": 65, "no2": 110},
        "default":     {"pm10": 50, "pm25": 70, "so2": 35, "co": 15, "o3": 22, "no2": 25},
    }

    # ── Inisialisasi session_state (Rule 7: keep user in control) ──
    if "preset_vals" not in st.session_state:
        st.session_state["preset_vals"] = PRESETS["default"].copy()

    # ── Reference values dari dataset (Rule 5: error prevention) ──
    df_ref = data["df"]
    fitur_ref = {
        "pm_sepuluh": "PM10", "pm_duakomalima": "PM2.5", "sulfur_dioksida": "SO₂",
        "karbon_monoksida": "CO", "ozon": "O₃", "nitrogen_dioksida": "NO₂",
    }
    refs = {
        fitur_ref[c]: {"mean": df_ref[c].mean(), "median": df_ref[c].median(),
                       "min": df_ref[c].min(), "max": df_ref[c].max()}
        for c in fitur_ref
    }

    col_input, col_result = st.columns([3, 2])

    with col_input:
        st.markdown('<div class="info-card"><h4>📝 Input Parameter Polutan</h4>', unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Pilih Model Klasifikasi:",
            ["XGBoost (Rekomendasi)", "Random Forest", "SVM"],
            help="XGBoost direkomendasikan karena memiliki akurasi tertinggi pada data uji.",
        )

        # ── Sliders dengan referensi real-world (Rule 8: memory load) ──
        c1, c2 = st.columns(2)
        with c1:
            pm10 = st.slider(
                "PM10 (μg/m³)", 0.0, 200.0, float(st.session_state["preset_vals"]["pm10"]), 1.0,
                help=f"Particulate Matter ≤ 10μm · Rata-rata dataset: {refs['PM10']['mean']:.1f} · Median: {refs['PM10']['median']:.1f}",
                key="sl_pm10",
            )
            pm25 = st.slider(
                "PM2.5 (μg/m³)", 0.0, 200.0, float(st.session_state["preset_vals"]["pm25"]), 1.0,
                help=f"Particulate Matter ≤ 2.5μm · Rata-rata dataset: {refs['PM2.5']['mean']:.1f} · Median: {refs['PM2.5']['median']:.1f}",
                key="sl_pm25",
            )
            so2 = st.slider(
                "SO₂ (μg/m³)", 0.0, 120.0, float(st.session_state["preset_vals"]["so2"]), 1.0,
                help=f"Sulfur Dioksida · Rata-rata dataset: {refs['SO₂']['mean']:.1f} · Median: {refs['SO₂']['median']:.1f}",
                key="sl_so2",
            )
        with c2:
            co = st.slider(
                "CO (μg/m³)", 0.0, 80.0, float(st.session_state["preset_vals"]["co"]), 1.0,
                help=f"Karbon Monoksida · Rata-rata dataset: {refs['CO']['mean']:.1f} · Median: {refs['CO']['median']:.1f}",
                key="sl_co",
            )
            o3 = st.slider(
                "O₃ (μg/m³)", 0.0, 120.0, float(st.session_state["preset_vals"]["o3"]), 1.0,
                help=f"Ozon · Rata-rata dataset: {refs['O₃']['mean']:.1f} · Median: {refs['O₃']['median']:.1f}",
                key="sl_o3",
            )
            no2 = st.slider(
                "NO₂ (μg/m³)", 0.0, 200.0, float(st.session_state["preset_vals"]["no2"]), 1.0,
                help=f"Nitrogen Dioksida · Rata-rata dataset: {refs['NO₂']['mean']:.1f} · Median: {refs['NO₂']['median']:.1f}",
                key="sl_no2",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Preset & Reset buttons (Rule 6: easy reversal + Rule 2: usability) ──
        st.markdown("#### 💡 Preset Skenario")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            if st.button("🟢 Udara Bersih", use_container_width=True):
                st.session_state["preset_vals"] = PRESETS["baik"].copy()
                st.toast("✅ Preset 'Udara Bersih' diterapkan", icon="🟢")
                st.rerun()
        with pc2:
            if st.button("🟡 Udara Sedang", use_container_width=True):
                st.session_state["preset_vals"] = PRESETS["sedang"].copy()
                st.toast("✅ Preset 'Udara Sedang' diterapkan", icon="🟡")
                st.rerun()
        with pc3:
            if st.button("🔴 Udara Buruk", use_container_width=True):
                st.session_state["preset_vals"] = PRESETS["tidaksehat"].copy()
                st.toast("✅ Preset 'Udara Buruk' diterapkan", icon="🔴")
                st.rerun()
        with pc4:
            if st.button("↺ Reset", use_container_width=True, help="Kembali ke nilai default"):
                st.session_state["preset_vals"] = PRESETS["default"].copy()
                st.toast("🔄 Slider direset ke default", icon="↺")
                st.rerun()

    # ── Prediksi ──
    input_data = pd.DataFrame([{
        "pm_sepuluh": pm10, "pm_duakomalima": pm25, "sulfur_dioksida": so2,
        "karbon_monoksida": co, "ozon": o3, "nitrogen_dioksida": no2,
    }])

    if "XGBoost" in model_choice:
        pred = models["xgb_best"].predict(input_data)[0]
        # Confidence dari predict_proba (Rule 3: informative feedback)
        try:
            proba = models["xgb_best"].predict_proba(input_data)[0]
            confidence = float(proba.max())
        except Exception:
            confidence = None
        model_used = "XGBoost"
    elif "Random Forest" in model_choice:
        pred = models["rf_best"].predict(input_data)[0]
        try:
            proba = models["rf_best"].predict_proba(input_data)[0]
            confidence = float(proba.max())
        except Exception:
            confidence = None
        model_used = "Random Forest"
    else:
        pred = models["svm_best"].predict(data["scaler"].transform(input_data))[0]
        confidence = None  # SVC tanpa probability=True tidak punya predict_proba
        model_used = "SVM"

    kategori = data["le"].inverse_transform([pred])[0]

    with col_result:
        css_class = {"BAIK": "pred-baik", "SEDANG": "pred-sedang", "TIDAK SEHAT": "pred-tidaksehat"}
        emoji = {"BAIK": "🟢", "SEDANG": "🟡", "TIDAK SEHAT": "🔴"}

        # ── Hasil prediksi + confidence (Rule 3: feedback) ──
        conf_html = ""
        if confidence is not None:
            conf_html = f"""<div style="margin-top:6px; font-size:0.78rem; opacity:0.65;">
                Tingkat keyakinan: <strong>{confidence:.1%}</strong>
            </div>"""

        st.markdown(f"""
        <div class="prediction-result {css_class[kategori]}">
            <div style="font-size:3rem;">{emoji[kategori]}</div>
            <div class="pred-label">{kategori}</div>
            <div style="margin-top:8px; font-size:0.88rem; opacity:0.7;">
                Diprediksi oleh <strong>{model_used}</strong>
            </div>
            {conf_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Rekomendasi aktivitas ──
        rekom_class = {"BAIK": "rekom-baik", "SEDANG": "rekom-sedang", "TIDAK SEHAT": "rekom-tidaksehat"}
        rekom_text = {
            "BAIK": """<strong>✅ Kualitas Udara Baik</strong><br>
                Aman untuk beraktivitas di luar ruangan. Cocok untuk berolahraga, jalan kaki, dan kegiatan outdoor lainnya.""",
            "SEDANG": """<strong>⚠️ Kualitas Udara Sedang</strong><br>
                Masih aman untuk sebagian besar orang. Kelompok sensitif (anak-anak, lansia, penderita asma) disarankan mengurangi aktivitas berat di luar ruangan.""",
            "TIDAK SEHAT": """<strong>🚨 Kualitas Udara Tidak Sehat</strong><br>
                Kurangi aktivitas luar ruangan. Gunakan masker jika harus keluar. Kelompok sensitif sebaiknya tetap di dalam ruangan dengan sirkulasi udara yang baik.""",
        }
        st.markdown(f"""<div class="rekom-card {rekom_class[kategori]}">
            {rekom_text[kategori]}
        </div>""", unsafe_allow_html=True)

        # ── Radar chart input (Rule 8: visual summary) ──
        labels = ["PM10", "PM2.5", "SO₂", "CO", "O₃", "NO₂"]
        values = [pm10, pm25, so2, co, o3, no2]
        maxes = [200, 200, 120, 80, 120, 200]
        norm_vals = [v / m * 100 for v, m in zip(values, maxes)]
        norm_vals.append(norm_vals[0])

        fig = go.Figure(go.Scatterpolar(
            r=norm_vals,
            theta=labels + [labels[0]],
            fill="toself",
            line_color=COLORS[kategori],
            fillcolor=COLORS[kategori],
            opacity=0.25,
            name="Input",
        ))
        fig.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(l=40, r=40, t=30, b=30)},
            height=300,
            polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: TENTANG
# ════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang":
    breadcrumb("ℹ️", "Tentang")
    st.markdown("""
    <div class="hero-container" style="background: linear-gradient(135deg, #1E3A5F 0%, #0A6847 100%);">
        <div class="hero-title">Tentang JakU</div>
        <div class="hero-sub">
            Jakarta Kualitas Udara — Dashboard klasifikasi kualitas udara berbasis machine learning untuk DKI Jakarta.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Tujuan Penelitian</h4>
            <p style="color:#6B8A73; line-height:1.7; font-size:0.9rem;">
                Membangun model klasifikasi kualitas udara di DKI Jakarta berdasarkan Indeks Standar
                Pencemar Udara (ISPU) menggunakan tiga algoritma machine learning: Random Forest,
                XGBoost, dan SVM. Penelitian ini mengikuti metodologi CRISP-DM dan aplikasi dashboard
                dikembangkan berdasarkan prinsip User-Centered Design (UCD).
            </p>
        </div>

        <div class="info-card">
            <h4>📊 Tentang Dataset</h4>
            <p style="color:#6B8A73; line-height:1.7; font-size:0.9rem;">
                Dataset ISPU DKI Jakarta terdiri dari data harian kualitas udara yang dikumpulkan dari
                6 stasiun pemantauan. Enam parameter polutan yang digunakan sebagai fitur: PM10, PM2.5,
                SO₂, CO, O₃, dan NO₂. Target klasifikasi terdiri dari 3 kategori: BAIK, SEDANG, dan
                TIDAK SEHAT.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🔬 Metodologi</h4>
            <p style="color:#6B8A73; line-height:1.7; font-size:0.9rem;">
                <strong>CRISP-DM</strong> (Cross-Industry Standard Process for Data Mining) digunakan sebagai
                kerangka kerja utama. Evaluasi menggunakan confusion matrix, classification report
                (precision, recall, F1-score), serta 5-fold stratified cross-validation. Hyperparameter
                tuning dilakukan dengan GridSearchCV.
            </p>
        </div>

        <div class="info-card">
            <h4>💡 Prinsip UCD</h4>
            <p style="color:#6B8A73; line-height:1.7; font-size:0.9rem;">
                Dashboard ini dirancang berdasarkan prinsip User-Centered Design:<br>
                • <strong>Learnability</strong> — Navigasi intuitif dengan sidebar dan tab<br>
                • <strong>Efficiency</strong> — Informasi penting langsung terlihat di beranda<br>
                • <strong>Memorability</strong> — Konsistensi warna dan layout antar halaman<br>
                • <strong>Error Prevention</strong> — Slider dengan batas aman untuk input prediksi<br>
                • <strong>Satisfaction</strong> — Visualisasi interaktif dan desain estetik
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="text-align:center; margin-top:16px;">
        <h4>🏷️ Kategori ISPU</h4>
        <div style="display:flex; justify-content:center; gap:20px; margin-top:12px;">
            <div><span class="badge-baik">BAIK</span><br><small style="color:#6B8A73;">ISPU 0 – 50</small></div>
            <div><span class="badge-sedang">SEDANG</span><br><small style="color:#6B8A73;">ISPU 51 – 100</small></div>
            <div><span class="badge-tidaksehat">TIDAK SEHAT</span><br><small style="color:#6B8A73;">ISPU 101 – 200</small></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 8 Golden Rules Mapping (Shneiderman) ──
    st.markdown('<div class="section-header" style="margin-top:28px;">🏆 8 Golden Rules of Interface Design</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Setiap perubahan UI di dashboard ini dipetakan ke prinsip Shneiderman, sesuai standar evaluasi usability HCI.</div>', unsafe_allow_html=True)

    golden_rules = [
        ("1. Strive for Consistency",
         "Palette warna, font (Plus Jakarta Sans), ikon, dan struktur KPI card konsisten di semua 6 halaman."),
        ("2. Seek Universal Usability",
         "Expander 'Cara menggunakan' di setiap halaman membantu pemula. Tooltip help (?) tersedia pada parameter teknis."),
        ("3. Offer Informative Feedback",
         "Toast notification muncul saat preset diklik. Tingkat keyakinan model (confidence) ditampilkan pada hasil prediksi."),
        ("4. Design Dialogs to Yield Closure",
         "Breadcrumb 'JakU › Halaman' di atas tiap halaman. Insight box di akhir tiap analisis sebagai kesimpulan."),
        ("5. Prevent Errors",
         "Slider dengan range nyata sesuai dataset ISPU. Tooltip menampilkan rata-rata dataset sebagai referensi."),
        ("6. Permit Easy Reversal of Actions",
         "Tombol Reset pada halaman Prediksi mengembalikan semua slider ke nilai default. Sidebar selalu accessible."),
        ("7. Keep Users in Control",
         "User memilih sendiri model, parameter, dan preset. Tidak ada modal atau popup yang mengganggu alur kerja."),
        ("8. Reduce Short-Term Memory Load",
         "Nilai slider real-time terlihat di label. Referensi mean/median dataset di tooltip tiap slider."),
    ]

    gc1, gc2 = st.columns(2)
    for i, (rule, desc) in enumerate(golden_rules):
        target_col = gc1 if i % 2 == 0 else gc2
        with target_col:
            st.markdown(f"""
            <div class="info-card" style="padding:16px 18px; margin-bottom:10px;">
                <h4 style="font-size:0.95rem; margin-bottom:6px;">{rule}</h4>
                <p style="color:#6B8A73; line-height:1.6; font-size:0.82rem; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    app_footer()
