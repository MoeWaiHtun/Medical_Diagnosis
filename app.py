import base64
import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
import speech_recognition as sr

import nltk
from nltk.stem import PorterStemmer

try:
    stemmer = PorterStemmer()
except Exception:
    stemmer = None

from utils.data_loader import load_all_datasets
from utils.text_processing import clean_text, predict_next_words

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="PulseMind AI - ဆေးဘက်ဆိုင်ရာ အကူအညီပေးစနစ်", 
    page_icon="🏥", 
    layout="wide"
)

# Initialize Session States
if 'lang' not in st.session_state:
    st.session_state.lang = "my"

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "dark"

if 'onboarding_step' not in st.session_state:
    st.session_state.onboarding_step = 0

if "kb_search_query" not in st.session_state:
    st.session_state.kb_search_query = ""

if "selected_symptoms_list" not in st.session_state:
    st.session_state.selected_symptoms_list = []

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# ==============================================================================
# PROCESS, MORPHOLOGY & N-GRAM FUNCTIONS
# ==============================================================================
MY_STOPWORDS = [
    "နေတယ်", "နေတာ", "တယ်", "သည်", "တာ", "ခြင်း", "လွန်းလို့", 
    "အရမ်း", "ရမ်း", "ရတာ", "ဖြစ်တယ်", "ဖြစ်နေတာ", "ဖြစ်လို့", "တွေ", "မအီမသာ",
    "နေပါတယ်", "ပါတယ်", "များ", "ကြီး"
]

def clean_myanmar_morphology(text):
    for word in MY_STOPWORDS:
        text = text.replace(word, "")
    return text.strip()

def process_symptom_text(text, lang="my"):
    cleaned = clean_text(text, lang=lang) if callable(clean_text) else text.lower().strip()
    return cleaned

def stem_word(word):
    word = word.lower().strip()
    if stemmer:
        return stemmer.stem(word)
    for suffix in ['ing', 'ed', 'es', 's']:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def generate_ngrams(words_list, n):
    ngrams = []
    for i in range(len(words_list) - n + 1):
        ngrams.append(" ".join(words_list[i:i+n]))
    return ngrams

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            return text
    except sr.UnknownValueError:
        return "⚠️ Could not understand the audio. Please try again."
    except Exception as e:
        return f"⚠️ Speech Recognition Error: {str(e)}"

# ==============================================================================
# DYNAMIC MODEL & DATASET LOADERS
# ==============================================================================
@st.cache_resource
def load_artifacts_by_lang(lang_code):
    model_dir = f"models/{lang_code}" if os.path.exists(f"models/{lang_code}") else "models"
    try:
        all_models = joblib.load(os.path.join(model_dir, "all_models.pkl"))
        tfidf = joblib.load(os.path.join(model_dir, "tfidf.pkl"))
        le = joblib.load(os.path.join(model_dir, "le.pkl"))
        results = joblib.load(os.path.join(model_dir, "model_results.pkl"))
        X_train = joblib.load(os.path.join(model_dir, "X_train_tfidf.pkl"))
        y_train = joblib.load(os.path.join(model_dir, "y_train.pkl"))
        classes = joblib.load(os.path.join(model_dir, "classes.pkl"))
        
        df_dummy, desc_dict, prec_dict, severity_dict = load_all_datasets(lang=lang_code)
        
        return all_models, tfidf, le, results, desc_dict, prec_dict, severity_dict, X_train, y_train, classes, df_dummy
    except Exception as e:
        st.error(f"❌ Model or Dataset artifacts for [{lang_code.upper()}] not found!\nError: {e}")
        st.stop()

def get_artifacts_for_text(input_text, default_lang):
    if re.search(r'[\u1000-\u109F]', input_text):
        target_lang = "my"
    elif input_text.strip():
        target_lang = "en"
    else:
        target_lang = default_lang
        
    return load_artifacts_by_lang(target_lang), target_lang

all_models, tfidf, le, results, desc_dict, prec_dict, severity_dict, X_train, y_train, classes, df_dummy = load_artifacts_by_lang(st.session_state.lang)

# ==============================================================================
# UI TRANSLATION DICTIONARY
# ==============================================================================
translations = {
    "my": {
        "title": "PulseMind AI",
        "subtitle": "ဆေးဘက်ဆိုင်ရာ AI အကူအညီပေးစနစ်",
        "nav_header": "အဓိက ကဏ္ဍများ",
        "nav_pred": "🩺 ရောဂါလက္ခဏာ ခန့်မှန်းစနစ်",
        "nav_models": "📊 မော်ဒယ်များ နှိုင်းယှဉ်ချက်",
        "nav_pca": "📈 ဒေတာ အချက်အလက်များ",
        "nav_kb": "📚 ဆေးပညာ ဗဟုသုတဘဏ်",
        "info_box": "ℹ️ ဤစနစ်သည် Machine Learning အယ်လ်ဂိုရီသမ်များနှင့် TF-IDF N-gram နည်းပညာကို အသုံးပြု၍ ရောဂါ ခန့်မှန်းပေးပါသည်။",
        "pred_title": "🩺 ရောဂါလက္ခဏာ အခြေပြု ခန့်မှန်းစနစ်",
        "voice_title": "🎙️ Voice Input System (English Only)",
        "select_sym": "ခံစားနေရသော ရောဂါလက္ခဏာများကို ရွေးချယ်ပါ:",
        "input_sym": "သို့မဟုတ် ဖြစ်ပွားနေပုံကို စာဖြင့် ရေးသားဖော်ပြပါ:",
        "placeholder_sym": "ဥပမာ- severe headache သို့မဟုတ် fever",
        "matched_sym": "🔍 Matched Symptoms (N-gram & Morphology):",
        "select_model": "အသုံးပြုမည့် AI အယ်လ်ဂိုရီသမ်ကို ရွေးချယ်ပါ:",
        "btn_predict": "🚀 ရောဂါခန့်မှန်းချက် ထုတ်ပြန်မည်",
        "warning_sym": "⚠️ ကျေးဇူးပြု၍ မှန်ကန်သော ရောဂါလက္ခဏာများ ရွေးချယ်ပါ သို့မဟုတ် ရေးသားပါ။",
        "no_symptom_found": "⚠️ ထည့်သွင်းထားသော စာသားတွင် တိကျသော ရောဂါလက္ခဏာ မပါဝင်ပါ။ ကျေးဇူးပြု၍ ရောဂါလက္ခဏာကို ပိုမိုတိကျစွာ ရေးသားပါ။",
        "res_title": "🎯 ခန့်မှန်းရရှိသော ရောဂါရလဒ်",
        "confidence": "ခန့်မှန်းရရှိမှု သေချာမှုနှုန်း:",
        "desc": "📄 ရောဂါအကြောင်းအရာ:",
        "precautions": "🛡️ ကြိုတင်ကာကွယ်ရန်/ဆောင်ရွက်ရန်:",
        "top3_title": "📊 အခြားဖြစ်နိုင်ခြေရှိသော ရောဂါများ (Top 3)",
        "quick_info": "⚡ အချက်အလက်",
        "total_severity": "စုစုပေါင်း ပြင်းထန်မှုရမှတ်",
        "sev_high": "🟥 စိုးရိမ်ရသည့် အခြေအနေ",
        "sev_mod": "🟧 အသင့်အတင့် အခြေအနေ",
        "sev_low": "🟩 ပုံမှန်/စိုးရိမ်စရာမရှိ",
        "search_kb": "🔍 ရောဂါအမည်ဖြင့် ရှာဖွေရန်:",
        "skip": "ကျော်သွားရန်",
        "next": "ရှေ့သို့ ➔",
        "start": "စတင်မည် 🚀",
        "welcome_title": "PulseMind AI မှ ကြိုဆိုပါသည်",
        "welcome_desc": "တိကျသော ရောဂါလက္ခဏာ ခွဲခြားစစ်ဆေးမှုများနှင့် ကျန်းမာရေးဆိုင်ရာ အထောက်အကူများ ရရှိနိုင်ပါသည်။",
        "welcome_btn": "စတင်အသုံးပြုရန် ➔",
        "step1_title": "AI ဖြင့် ရောဂါခန့်မှန်းခြင်း",
        "step1_desc": "ရောဂါလက္ခဏာများကို စာဖြင့်ရေးသား၍ဖြစ်စေ ရွေးချယ်၍ဖြစ်စေ လွယ်ကူစွာ စစ်ဆေးနိုင်ပါသည်။",
        "step2_title": "လွယ်ကူစုံလင်သော ပြသမှု",
        "step2_desc": "ဖြစ်နိုင်ခြေ ရာခိုင်နှုန်းများ၊ စိုးရိမ်ရမှု အဆင့်များနှင့် ဆေးပညာ ဗဟုသုတများကို တပြိုင်နက် ကြည့်ရှုနိုင်ပါသည်။",
        "step3_title": "စိတ်ချရသော လမ်းညွှန်ချက်များ",
        "step3_desc": "Machine Learning မော်ဒယ်များ၏ တွက်ချက်မှုဖြင့် ဆောင်ရန်/ရှောင်ရန် အကြံပြုချက်များကို ရရှိမည်ဖြစ်ပါသည်။"
    },
    "en": {
        "title": "PulseMind AI",
        "subtitle": "AI Clinical Decision Support",
        "nav_header": "Main Menu",
        "nav_pred": "🩺 Symptom Predictor",
        "nav_models": "📊 Model Comparison",
        "nav_pca": "📈 Data Visualization",
        "nav_kb": "📚 Knowledge Base",
        "info_box": "ℹ️ This system uses Machine Learning algorithms & TF-IDF N-gram vectorization to predict conditions.",
        "pred_title": "🩺 Symptom-Based Diagnostic System",
        "voice_title": "🎙️ Voice Input System (English Only)",
        "select_sym": "Select presenting symptoms:",
        "input_sym": "Or type symptom description:",
        "placeholder_sym": "e.g., severe headache or fever",
        "matched_sym": "🔍 Matched Symptoms (N-gram & Morphology):",
        "select_model": "Select AI Algorithm:",
        "btn_predict": "🚀 Run Diagnostic Analysis",
        "warning_sym": "⚠️ Please select or type valid medical symptoms.",
        "no_symptom_found": "⚠️ No recognized symptoms found in text. Please describe symptoms more accurately.",
        "res_title": "🎯 Diagnostic Prediction Result",
        "confidence": "Model Confidence:",
        "desc": "📄 Overview:",
        "precautions": "🛡️ Care Guidelines & Precautions:",
        "top3_title": "📊 Differential Diagnosis (Top 3)",
        "quick_info": "⚡ Quick Metrics",
        "total_severity": "Total Severity Score",
        "sev_high": "🟥 High Clinical Severity",
        "sev_mod": "🟧 Moderate Severity",
        "sev_low": "🟩 Low / Normal Severity",
        "search_kb": "🔍 Search by Condition Name:",
        "skip": "Skip",
        "next": "Next ➔",
        "start": "Get Started 🚀",
        "welcome_title": "Welcome to PulseMind AI",
        "welcome_desc": "Your intelligent AI assistant for accurate symptom analysis and medical decision support.",
        "welcome_btn": "Start Experience ➔",
        "step1_title": "AI-Powered Diagnosis",
        "step1_desc": "Input your symptoms in natural text or choose from list to receive fast diagnostic analysis.",
        "step2_title": "Easy & Interactive",
        "step2_desc": "Visualize probability spreads, severity ratings, and medical knowledge instantly.",
        "step3_title": "Reliable Care Guidelines",
        "step3_desc": "Get tailored recommendations and precautions backed by machine learning models."
    }
}

t = translations.get(st.session_state.lang, translations["my"])

def toggle_language():
    if st.session_state.lang == "my":
        st.session_state.lang = "en"
    else:
        st.session_state.lang = "my"
    st.session_state.selected_symptoms_list = []
    if 'active_tab' in st.session_state:
        del st.session_state.active_tab

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = t['nav_pred']

# ==============================================================================
# UI STYLES & DYNAMIC THEME SELECTORS
# ==============================================================================
if st.session_state.theme_mode == "dark":
    bg_color = "#0e1117"
    card_bg = "#1e2130"
    text_color = "#f0f2f6"
    text_muted = "#a8b2d1"
    border_color = "#3e4559"
    input_bg = "#262b3a"
    dropdown_bg = "#1e2130"
    dropdown_text = "#ffffff"
    dropdown_hover = "#3566d6"
    plotly_template = "plotly_dark"
    chart_font_color = "#f0f2f6"
    chart_grid_color = "#2d3748"
    nav_btn_bg = "#262b3a"
    nav_btn_text = "#ffffff"
else:
    bg_color = "#f8f9fa"
    card_bg = "#ffffff"
    text_color = "#1f2937"
    text_muted = "#4b5563"
    border_color = "#cbd5e1"
    input_bg = "#ffffff"
    dropdown_bg = "#ffffff"
    dropdown_text = "#111827"
    dropdown_hover = "#2563eb"
    plotly_template = "plotly_white"
    chart_font_color = "#1f2937"
    chart_grid_color = "#e2e8f0"
    nav_btn_bg = "#e2e8f0"
    nav_btn_text = "#0f172a"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Padauk:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Padauk', 'Plus Jakarta Sans', sans-serif !important;
    }}

    .stApp {{ 
        background-color: {bg_color} !important; 
        color: {text_color} !important; 
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color};
    }}

    p, label, span, div, h1, h2, h3, h4, h5, h6, li, strong {{
        color: {text_color} !important;
        line-height: 1.5;
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        gap: 0.4rem !important;
    }}

    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button {{
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: none !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }}

    [data-testid="stSidebar"] button[kind="secondary"] {{
        background-color: {nav_btn_bg} !important;
        color: {nav_btn_text} !important;
        border: 1px solid {border_color} !important;
    }}

    [data-testid="stSidebar"] button[kind="secondary"] * {{
        color: {nav_btn_text} !important;
    }}

    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, #4c8bf5 0%, #3566d6 100%) !important;
        color: #ffffff !important;
        border: none !important;
    }}

    [data-testid="stSidebar"] button[kind="primary"] * {{
        color: #ffffff !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-radius: 10px !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        transition: all 0.2s ease-in-out !important;
    }}

    div[data-baseweb="select"] > div:hover {{
        border-color: #4c8bf5 !important;
        box-shadow: 0 0 0 1px #4c8bf5 !important;
    }}

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [role="button"],
    div[data-baseweb="select"] div {{
        color: {text_color} !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {dropdown_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2) !important;
    }}

    li[role="option"],
    div[role="option"] {{
        background-color: {dropdown_bg} !important;
        color: {dropdown_text} !important;
        border-radius: 6px !important;
        margin: 2px 6px !important;
        padding: 8px 12px !important;
        transition: background-color 0.15s ease !important;
    }}

    li[role="option"]:hover,
    div[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {{
        background-color: {dropdown_hover} !important;
        color: #ffffff !important;
    }}

    span[data-baseweb="tag"] {{
        background-color: #3566d6 !important;
        border-radius: 6px !important;
    }}
    
    span[data-baseweb="tag"] span {{
        color: #ffffff !important;
    }}

    input[type="text"], textarea {{
        background-color: {input_bg} !important;
        border-radius: 10px !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        font-size: 1rem !important;
    }}

    .onboarding-card {{
        max-width: 440px;
        margin: 2rem auto;
        background: {card_bg} !important;
        border-radius: 24px;
        padding: 2.5rem 2rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
        border: 1px solid {border_color} !important;
        text-align: center;
    }}
    .onboarding-icon {{ font-size: 3.8rem; margin-bottom: 1rem; }}
    .onboarding-title {{ font-size: 1.6rem; font-weight: 800; color: {text_color} !important; margin-bottom: 0.6rem; }}
    .onboarding-desc {{ font-size: 0.95rem; color: {text_muted} !important; line-height: 1.5; margin-bottom: 1.5rem; }}

    .stMainBlockContainer div.stButton>button {{ 
        background: linear-gradient(135deg, #4c8bf5 0%, #3566d6 100%) !important;
        color: #ffffff !important; 
        border-radius: 12px !important; 
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        width: 100% !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: #4c8bf5 !important;
        font-size: 1.8rem !important;
    }}

    .risk-box {{
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        font-size: 1rem;
    }}
    .high-risk {{ background-color: #fee2e2; color: #991b1b !important; border: 1px solid #f87171; }}
    .mod-risk {{ background-color: #fef3c7; color: #92400e !important; border: 1px solid #fbbf24; }}
    .low-risk {{ background-color: #d1fae5; color: #065f46 !important; border: 1px solid #34d399; }}

    .result-info {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }}

    .main-title {{
        text-align: center;
        font-weight: 800;
        margin-bottom: 20px;
        font-size: 1.8rem;
        background: linear-gradient(90deg, #4c8bf5, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ONBOARDING FLOW LOGIC
# ==============================================================================
if st.session_state.onboarding_step < 4:
    ob_col1, ob_col2 = st.columns([3, 2])
    with ob_col2:
        lang_btn_label = "🇬🇧 English" if st.session_state.lang == "my" else "🇲🇲 မြန်မာ"
        theme_btn_label = "☀️ Light" if st.session_state.theme_mode == "dark" else "🌙 Dark"
        
        c_lang, c_theme = st.columns(2)
        with c_lang:
            st.button(lang_btn_label, key="ob_lang_toggle", on_click=toggle_language, use_container_width=True)
        with c_theme:
            if st.button(theme_btn_label, key="ob_theme_toggle", use_container_width=True):
                st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                st.rerun()

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        step = st.session_state.onboarding_step
        
        if step == 0:
            st.markdown(f"""
            <div class="onboarding-card">
                <div class="onboarding-icon">🩺</div>
                <div class="onboarding-title">{t['welcome_title']}</div>
                <div class="onboarding-desc">{t['welcome_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(t["welcome_btn"], use_container_width=True):
                st.session_state.onboarding_step = 1
                st.rerun()

        elif step == 1:
            st.markdown(f"""
            <div class="onboarding-card">
                <div class="onboarding-icon">🧠</div>
                <div class="onboarding-title">{t['step1_title']}</div>
                <div class="onboarding-desc">{t['step1_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(t["skip"], use_container_width=True):
                    st.session_state.onboarding_step = 4
                    st.rerun()
            with btn_col2:
                if st.button(t["next"], use_container_width=True):
                    st.session_state.onboarding_step = 2
                    st.rerun()

        elif step == 2:
            st.markdown(f"""
            <div class="onboarding-card">
                <div class="onboarding-icon">⚡</div>
                <div class="onboarding-title">{t['step2_title']}</div>
                <div class="onboarding-desc">{t['step2_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(t["skip"], use_container_width=True):
                    st.session_state.onboarding_step = 4
                    st.rerun()
            with btn_col2:
                if st.button(t["next"], use_container_width=True):
                    st.session_state.onboarding_step = 3
                    st.rerun()

        elif step == 3:
            st.markdown(f"""
            <div class="onboarding-card">
                <div class="onboarding-icon">🛡️</div>
                <div class="onboarding-title">{t['step3_title']}</div>
                <div class="onboarding-desc">{t['step3_desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(t["start"], use_container_width=True):
                st.session_state.onboarding_step = 4
                st.rerun()

# ==============================================================================
# MAIN APPLICATION FLOW
# ==============================================================================
else:
    with st.sidebar:
        try:
            with open("image1.jpg", "rb") as f:
                img = base64.b64encode(f.read()).decode()

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/jpeg;base64,{img}"
                         style="
                            width: 90px;
                            height: 90px;
                            border-radius: 50%;
                            object-fit: cover;
                            display: block;
                            margin: auto;
                            border: 3px solid #4c8bf5;
                         ">
                </div>
                """,
                unsafe_allow_html=True
            )
        except FileNotFoundError:
            st.markdown(
                """
                <div style="text-align: center; font-size: 4.5rem; margin-bottom: 0.5rem;">
                    🩺
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(f"<h2 style='text-align: center;'>🩺 {t['title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: {text_muted}; font-size: 0.9em;'>{t['subtitle']}</p>", unsafe_allow_html=True)
        
        lang_btn_label = "🇬🇧 English" if st.session_state.lang == "my" else "🇲🇲 မြန်မာ"
        theme_btn_label = "☀️ Light" if st.session_state.theme_mode == "dark" else "🌙 Dark"

        col_side_lang, col_side_theme = st.columns(2)
        with col_side_lang:
            st.button(lang_btn_label, key="sidebar_lang_toggle", on_click=toggle_language, use_container_width=True)
        with col_side_theme:
            if st.button(theme_btn_label, key="sidebar_theme_toggle", use_container_width=True):
                st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                st.rerun()

        st.markdown("---")
        st.markdown(f"**{t['nav_header']}**")

        nav_items = [
            t['nav_pred'],
            t['nav_models'],
            t['nav_pca'],
            t['nav_kb']
        ]

        for item in nav_items:
            btn_type = "primary" if st.session_state.active_tab == item else "secondary"
            if st.button(item, key=f"nav_btn_{item}", type=btn_type, use_container_width=True):
                st.session_state.active_tab = item
                st.rerun()

        tab_choice = st.session_state.active_tab

        st.markdown("---")
        st.info(t['info_box'])

    # TAB 1: PREDICTOR
    if tab_choice == t['nav_pred']:
        st.markdown(f"<h1 class='main-title'>{t['pred_title']}</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2.2, 1])
        
        with col1:
            audio_val = st.audio_input(t["voice_title"])

            if audio_val:
                with st.spinner("🎙️ Transcribing Audio to English Text..."):
                    result_text = transcribe_audio(audio_val)
                    if not result_text.startswith("⚠️"):
                        st.session_state.transcribed_text = result_text
                        st.success(f"🗣️ Recognized Text: **{result_text}**")
                    else:
                        st.warning(result_text)

            symptom_cols = [c for c in df_dummy.columns if c.startswith('Symptom') or c.startswith('ရောဂါလက္ခဏာ')]
            all_syms = set()
            for col in symptom_cols:
                vals = df_dummy[col].dropna().astype(str).str.strip().unique()
                for v in vals:
                    if v != 'nan' and v != '':
                        all_syms.add(v)
            all_syms = set(all_syms)

            for item in st.session_state.selected_symptoms_list:
                all_syms.add(item)
            
            all_syms_list = sorted(list(all_syms))

            def on_multiselect_change():
                st.session_state.selected_symptoms_list = st.session_state.ms_widget_key

            selected_symptoms = st.multiselect(
                t["select_sym"], 
                options=all_syms_list, 
                default=st.session_state.selected_symptoms_list,
                key="ms_widget_key",
                on_change=on_multiselect_change,
                placeholder="Search symptoms..."
            )
            
            user_text = st.text_input(
                t["input_sym"], 
                value=st.session_state.transcribed_text,
                placeholder=t["placeholder_sym"]
            )
            
            matched = []
            if user_text.strip():
                is_myanmar_input = bool(re.search(r'[\u1000-\u109F]', user_text))
                detected_lang = "my" if is_myanmar_input else "en"
                
                target_artifacts, active_lang = get_artifacts_for_text(user_text, detected_lang)
                target_df_dummy = target_artifacts[10]
                
                target_symptom_cols = [c for c in target_df_dummy.columns if c.startswith('Symptom') or c.startswith('ရောဂါလက္ခဏာ')]
                searchable_syms = set()
                for col in target_symptom_cols:
                    vals = target_df_dummy[col].dropna().astype(str).str.strip().unique()
                    for v in vals:
                        if v != 'nan' and v != '':
                            searchable_syms.add(v)
                
                searchable_syms_list = sorted(list(searchable_syms))

                raw_chunks = re.split(r'[,;&]|\band\b|\bwith\b|\bplus\b|\bနှင့်\b|\bနှင့်အတူ\b|\bပြီး\b|\bပြီးတော့\b', user_text, flags=re.IGNORECASE)
                
                for chunk in raw_chunks:
                    chunk_cleaned = process_symptom_text(chunk, lang=detected_lang)
                    if not chunk_cleaned:
                        continue
                    
                    if detected_lang == "my":
                        chunk_cleaned = clean_myanmar_morphology(chunk_cleaned)

                    words = chunk_cleaned.split()
                    
                    input_ngrams = []
                    for n in range(1, 4):
                        input_ngrams.extend(generate_ngrams(words, n))
                    
                    for s in searchable_syms_list:
                        s_clean = process_symptom_text(s.lower().replace('_', ' '), lang=detected_lang)
                        if detected_lang == "my":
                            s_clean = clean_myanmar_morphology(s_clean)

                        match_found = False
                        
                        if detected_lang == "my":
                            if s_clean in chunk_cleaned or chunk_cleaned in s_clean:
                                match_found = True
                            else:
                                s_words = [w for w in s_clean.split() if len(w) > 1]
                                chunk_words = [w for w in chunk_cleaned.split() if len(w) > 1]
                                
                                overlap_count = sum(1 for sw in s_words if any(sw in cw or cw in sw for cw in chunk_words))
                                if overlap_count >= len(s_words) and len(s_words) > 0:
                                    match_found = True
                        
                        else:
                            # 💡 FIX FOR GENERAL KEYWORDS LIKE "fever":
                            # If chunk contains a general word (e.g. fever) that matches a sub-symptom (high_fever/mild_fever)
                            if chunk_cleaned in s_clean or s_clean in input_ngrams or s_clean == chunk_cleaned:
                                match_found = True
                            else:
                                s_words = s_clean.split()
                                s_stemmed_words = [stem_word(sw) for sw in s_words]
                                stemmed_words = [stem_word(w) for w in words]
                                
                                matched_count = 0
                                for st_w in stemmed_words:
                                    if len(st_w) >= 3:
                                        if any(st_w == target_st_w for target_st_w in s_stemmed_words):
                                            matched_count += 1
                                
                                if matched_count > 0 and matched_count == len(s_stemmed_words):
                                    match_found = True
                        
                        if match_found:
                            matched.append(s)
                
                matched = list(dict.fromkeys(matched))
                
                if matched:
                    matched_str = ", ".join(matched)
                    st.success(f"{t['matched_sym']} {matched_str}")
                    
                    new_added = False
                    for m in matched:
                        if m not in st.session_state.selected_symptoms_list:
                            st.session_state.selected_symptoms_list.append(m)
                            new_added = True
                    if new_added:
                        st.rerun()

            model_choice = st.selectbox(
                t["select_model"], 
                list(all_models.keys())
            )
            
            predict_btn = st.button(t["btn_predict"], use_container_width=True)
            
            if predict_btn:
                combined_symptoms = list(st.session_state.selected_symptoms_list)
                if matched:
                    for m in matched:
                        if m not in combined_symptoms:
                            combined_symptoms.append(m)
                            
                # Check for Invalid Input / No Symptoms Selected & Found
                if not combined_symptoms and not user_text.strip():
                    st.warning(t["warning_sym"])
                else:
                    symptom_text = ' '.join(combined_symptoms) if combined_symptoms else user_text
                    
                    (active_models, active_tfidf, active_le, _, active_desc, active_prec, active_sev, _, _, _, _), active_lang = get_artifacts_for_text(symptom_text, st.session_state.lang)
                    
                    clean_input = process_symptom_text(symptom_text, lang=active_lang)
                    X_input = active_tfidf.transform([clean_input])
                    
                    # 💡 AVERAGE WEIGHT LOGIC FOR GENERAL KEYWORDS (e.g., "fever")
                    # If multiple sub-symptoms (like high_fever, mild_fever) are matched from a single word
                    if matched and len(matched) > 1:
                        feature_names = list(active_tfidf.get_feature_names_out())
                        X_dense = X_input.toarray()
                        matched_indices = []
                        for m in matched:
                            m_clean = m.lower().replace('_', ' ')
                            for idx, f_name in enumerate(feature_names):
                                if f_name in m_clean or m_clean in f_name:
                                    matched_indices.append(idx)
                        
                        if matched_indices:
                            avg_weight = 1.0 / len(matched_indices)
                            for idx in matched_indices:
                                X_dense[0, idx] = avg_weight
                            X_input = X_dense

                    # TF-IDF Non-zero Check
                    if not combined_symptoms and (hasattr(X_input, "nnz") and X_input.nnz == 0):
                        st.error(t["no_symptom_found"])
                    else:
                        with st.spinner("⏳ Analyzing symptoms..."):
                            current_model = active_models.get(model_choice, list(active_models.values())[0])
                            
                            try:
                                pred_encoded = current_model.predict(X_input)[0]
                                probs = current_model.predict_proba(X_input)[0]
                            except Exception:
                                X_dense = X_input.toarray() if hasattr(X_input, "toarray") else X_input
                                pred_encoded = current_model.predict(X_dense)[0]
                                probs = current_model.predict_proba(X_dense)[0]

                            pred_disease = active_le.inverse_transform([pred_encoded])[0]
                            confidence = max(probs) * 100
                            
                            top3_idx = np.argsort(probs)[-3:][::-1]
                            top3 = [(active_le.inverse_transform([idx])[0], probs[idx]*100) for idx in top3_idx]
                        
                        st.markdown("---")
                        st.markdown(f"<h3>{t['res_title']}</h3>", unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class='result-info'>
                            <h2 style='color: #4c8bf5; margin-top:0;'>{pred_disease}</h2>
                            <p style='color: {text_muted}; font-size: 1.1em;'>{t['confidence']} <strong>{confidence:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(float(confidence)/100)
                        
                        st.markdown(f"**{t['desc']}** {active_desc.get(pred_disease, 'N/A')}")
                        
                        st.markdown(f"**{t['precautions']}**")
                        precautions = active_prec.get(pred_disease, ["Consult a healthcare professional."])
                        for prec in precautions:
                            if pd.notna(prec) and str(prec).strip() != '':
                                st.markdown(f"- {prec}")
                        
                        st.markdown("---")
                        st.markdown(f"<h4>{t['top3_title']}</h4>", unsafe_allow_html=True)
                        
                        fig = px.bar(
                            x=[p for _, p in top3],
                            y=[name for name, _ in top3],
                            orientation='h',
                            labels={'x': 'Probability (%)', 'y': 'Condition'},
                            color=[p for _, p in top3],
                            color_continuous_scale='Blues',
                            template=plotly_template
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color=chart_font_color),
                            height=280,
                            margin=dict(l=10, r=10, t=30, b=10),
                            coloraxis_showscale=False
                        )
                        fig.update_xaxes(tickfont=dict(color=chart_font_color), title_font=dict(color=chart_font_color), gridcolor=chart_grid_color)
                        fig.update_yaxes(tickfont=dict(color=chart_font_color), title_font=dict(color=chart_font_color), gridcolor=chart_grid_color)
                        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>{t['quick_info']}</h3>", unsafe_allow_html=True)
            
            active_severity_dict = severity_dict
            if user_text.strip():
                is_myanmar_input = bool(re.search(r'[\u1000-\u109F]', user_text))
                input_lang = "my" if is_myanmar_input else "en"
                dynamic_arts, _ = get_artifacts_for_text(user_text, input_lang)
                active_severity_dict = dynamic_arts[6]

            total_severity = sum([active_severity_dict.get(s, 1) for s in st.session_state.selected_symptoms_list])
            st.metric(t["total_severity"], f"{total_severity} / 30")
            
            if st.session_state.selected_symptoms_list:
                if total_severity > 15:
                    st.markdown(f"<div class='risk-box high-risk'>{t['sev_high']}</div>", unsafe_allow_html=True)
                elif total_severity > 7:
                    st.markdown(f"<div class='risk-box mod-risk'>{t['sev_mod']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-box low-risk'>{t['sev_low']}</div>", unsafe_allow_html=True)
            else:
                st.info("Select symptoms to evaluate severity.")

    # TAB 2: MODEL COMPARISON
    elif tab_choice == t['nav_models']:
        st.markdown("<h1 class='main-title'>📊 Algorithm Performance Matrix</h1>", unsafe_allow_html=True)
        
        results_df = pd.DataFrame(results)
        
        st.markdown("### Accuracy Score Comparison")
        fig = px.bar(
            results_df, 
            x='Model', 
            y='Accuracy', 
            color='Accuracy', 
            color_continuous_scale='Viridis', 
            text=results_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%"),
            labels={'Model': 'Algorithm', 'Accuracy': 'Accuracy Score'},
            template=plotly_template
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=chart_font_color),
            coloraxis_showscale=False
        )
        fig.update_xaxes(tickfont=dict(color=chart_font_color), title_font=dict(color=chart_font_color), gridcolor=chart_grid_color)
        fig.update_yaxes(tickfont=dict(color=chart_font_color), title_font=dict(color=chart_grid_color), gridcolor=chart_grid_color)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Metrics Table")
        st.dataframe(results_df.style.format({'Accuracy': '{:.2%}'}), use_container_width=True)

    # TAB 3: DATA VISUALIZATION (PCA)
    elif tab_choice == t['nav_pca']:
        st.markdown("<h1 class='main-title'>📈 High-Dimensional PCA Clustering</h1>", unsafe_allow_html=True)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train.toarray())
        y_labels = le.inverse_transform(y_train)
        
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Disease': y_labels
        })
        
        fig = px.scatter(
            df_pca, 
            x='PC1', 
            y='PC2', 
            color='Disease', 
            hover_data={'Disease': True},
            template=plotly_template
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=chart_font_color, size=13),
            legend=dict(
                font=dict(color=chart_font_color),
                title=dict(font=dict(color=chart_font_color))
            )
        )
        fig.update_xaxes(
            tickfont=dict(color=chart_font_color), 
            title_font=dict(color=chart_font_color), 
            gridcolor=chart_grid_color
        )
        fig.update_yaxes(
            tickfont=dict(color=chart_font_color), 
            title_font=dict(color=chart_grid_color), 
            gridcolor=chart_grid_color
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: MEDICAL KNOWLEDGE BASE
    elif tab_choice == t['nav_kb']:
        st.markdown("<h1 class='main-title'>📚 Diagnostic Condition Directory</h1>", unsafe_allow_html=True)
        
        diseases = list(desc_dict.keys())
        
        if "kb_search_input" not in st.session_state:
            st.session_state.kb_search_input = st.session_state.kb_search_query

        def select_suggestion(suggested_text):
            st.session_state.kb_search_input = suggested_text
            st.session_state.kb_search_query = suggested_text

        search_q = st.text_input(
            t["search_kb"], 
            key="kb_search_input"
        )
        st.session_state.kb_search_query = search_q
        
        filtered_diseases = [d for d in diseases if search_q.lower() in d.lower()] if search_q.strip() else diseases
        
        for d in filtered_diseases:
            with st.expander(f"🩺 {d}"):
                st.markdown(f"**📖 Overview:** {desc_dict.get(d, 'N/A')}")
                st.markdown("**🛡️ Precautions:**")
                precautions = prec_dict.get(d, [])
                if precautions:
                    for prec in precautions:
                        if pd.notna(prec) and str(prec).strip() != '':
                            st.markdown(f"- {prec}")
