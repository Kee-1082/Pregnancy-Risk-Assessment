import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Maternal Health Risk Assessment",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS - PASTEL YELLOW THEME
# ========================================
pastel_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', 'Segoe UI', sans-serif;
    }
    
    /* Main app background - PASTEL YELLOW */
    .stApp {
        background-color: #FFF9E6 !important;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        padding: 30px;
        border-radius: 15px;
        color: #000000;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
        margin-bottom: 30px;
        text-align: center;
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
    }
    
    .dashboard-header p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        color: #000000;
        opacity: 0.85;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 10px 20px;
        color: #000000;
        font-weight: 500;
        border: 2px solid #FFA500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        border-color: #FF8C00;
    }
    
    /* All text BLACK */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }
    
    /* Section headers */
    h3 {
        margin-top: 20px !important;
        margin-bottom: 15px !important;
        font-weight: 600 !important;
    }
    
    h4 {
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    
    /* Input field labels - BLACK TEXT */
    .stNumberInput label, .stCheckbox label, .stSelectbox label, 
    .stTextInput label, .stTextArea label {
        color: #000000 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Input fields - TRANSPARENT WHITE BACKGROUND */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.7) !important;
        color: #000000 !important;
        border: 2px solid #FFA500 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #FF8C00 !important;
        box-shadow: 0 0 0 2px rgba(255, 165, 0, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Checkbox styling - TRANSPARENT WHITE */
    .stCheckbox {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #FFD700;
        margin: 5px 0;
    }
    
    .stCheckbox:hover {
        border-color: #FFA500;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    .stCheckbox label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    .stCheckbox label span {
        color: #000000 !important;
    }
    
    /* Expander styling - TRANSPARENT WHITE */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        color: #000000 !important;
        border-radius: 10px;
        font-weight: 600;
        padding: 15px;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.7);
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 20px;
        margin-top: 5px;
    }
    
    /* Risk cards */
    .risk-card-high {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4757 100%);
        padding: 25px;
        border-radius: 15px;
        color: #000000;
        box-shadow: 0 8px 25px rgba(255, 71, 87, 0.3);
        text-align: center;
        margin: 20px 0;
    }
    
    .risk-card-high h2, .risk-card-high h3, .risk-card-high p {
        color: #000000 !important;
        font-weight: 700;
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #7BED8D 0%, #5FD068 100%);
        padding: 25px;
        border-radius: 15px;
        color: #000000;
        box-shadow: 0 8px 25px rgba(95, 208, 104, 0.3);
        text-align: center;
        margin: 20px 0;
    }
    
    .risk-card-low h2, .risk-card-low h3, .risk-card-low p {
        color: #000000 !important;
        font-weight: 700;
    }
    
    /* Info box for risk factors */
    .risk-factors-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-left: 5px solid #FF6B6B;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .risk-factors-box h4 {
        color: #FF6B6B !important;
        margin-top: 0;
    }
    
    .risk-factors-box ul {
        color: #000000;
        margin: 10px 0;
    }
    
    .risk-factors-box li {
        color: #000000 !important;
        margin: 5px 0;
        font-weight: 500;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FFA500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .stMetric label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Button styling - ANALYZE BUTTON */
    .stButton > button {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%) !important;
        color: #000000 !important;
        border: 2px solid #FF8C00 !important;
        border-radius: 10px !important;
        padding: 15px 40px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3) !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 140, 0, 0.5) !important;
        background: linear-gradient(90deg, #FFA500 0%, #FF8C00 100%) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #7BED8D 0%, #5FD068 100%) !important;
        color: #000000 !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4) !important;
    }
    
    /* Sidebar styling - LIGHT ORANGE */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE4B5 0%, #FFDAB9 100%) !important;
        border-right: 3px solid #FFA500;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    /* Radio button text */
    .stRadio label span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        border-left: 4px solid #FFA500;
        color: #000000 !important;
    }
    
    .stAlert > div {
        color: #000000 !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #FFD700;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 2px solid #FFA500 !important;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        border-top: 2px solid #FFD700;
        margin: 30px 0;
    }
    
    /* Make sure all markdown text is black */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #000000 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
        opacity: 0.7;
    }
</style>
"""

st.markdown(pastel_css, unsafe_allow_html=True)

# ========================================
# LOAD MODEL ARTIFACTS - WITH ERROR HANDLING
# ========================================
@st.cache_resource
def load_model_artifacts():
    """Load ML model artifacts with error handling"""
    required_files = {
        'model': 'pregnancy_model.pkl',
        'scaler': 'scaler.pkl',
        'label_encoder': 'label_encoder.pkl',
        'features': 'feature_columns.pkl'
    }
    
    try:
        artifacts = {}
        for key, filename in required_files.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Missing required file: {filename}")
            artifacts[key] = joblib.load(filename)
        
        return artifacts['model'], artifacts['scaler'], artifacts['label_encoder'], artifacts['features']
    
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("📝 Please run `python model.py` first to generate model artifacts")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.info("The model files may be corrupted. Try regenerating them.")
        st.stop()

# ========================================
# AI AGENT WITH GEMINI - FIXED
# ========================================
@st.cache_resource
def get_ai_client():
    """Initialize Gemini AI client"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash instead of gemini-pro
        return genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
        )
    except Exception as e:
        st.error(f"Failed to initialize AI client: {e}")
        return None

# ========================================
# NEWS API INTEGRATION
# ========================================
@st.cache_data(ttl=3600)
def fetch_maternal_health_news():
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "maternal health pregnancy care",
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
            "pageSize": 5
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")
    
    return []

# ========================================
# MAIN FUNCTION
# ========================================
def main():
    model, scaler, label_encoder, feature_cols = load_model_artifacts()
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🤰 Maternal Health Risk Assessment</h1>
        <p>Simple and effective pregnancy risk evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Section",
            ["Risk Assessment", "Maternal News", "AI Health Assistant", "Model Comparison"],
            label_visibility="collapsed"
        )
    
    # Page routing
    if page == "Risk Assessment":
        risk_assessment_page(model, scaler, label_encoder, feature_cols)
    elif page == "Maternal News":
        maternal_news_page()
    elif page == "AI Health Assistant":
        ai_assistant_page()
    elif page == "Model Comparison":
        model_comparison_page()

# ========================================
# RISK ASSESSMENT PAGE
# ========================================
def risk_assessment_page(model, scaler, label_encoder, feature_cols):
    
    # Create tabs for organized input
    tab1, tab2 = st.tabs(["📋 Basic Information", "🏥 Clinical Assessment"])
    
    # Initialize session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    with tab1:
        basic_info_form()
    
    with tab2:
        clinical_assessment_form()
    
    # Analyze Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_col, clear_col = st.columns([2, 1])
        with analyze_col:
            if st.button("🔍 ANALYZE RISK", key="analyze_btn", use_container_width=True):
                if st.session_state.form_data:
                    st.session_state.show_results = True
                    st.rerun()
                else:
                    st.warning("⚠️ Please fill in the basic information first!")
        
        with clear_col:
            if st.button("🔄 Clear", key="clear_btn", use_container_width=True):
                st.session_state.form_data = {}
                st.session_state.show_results = False
                st.rerun()

    # Show results only after button click
    if st.session_state.show_results and st.session_state.form_data:
        st.markdown("---")
        perform_risk_analysis(model, scaler, label_encoder, feature_cols)

# ========================================
# BASIC INFORMATION FORM
# ========================================
def basic_info_form():
    st.markdown("### 📊 Patient Demographics & Vitals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 15, 60, 25, key="age_input")
        
        st.markdown("**Vital Signs**")
        blood_pressure = st.number_input("Systolic BP (mmHg)", 70, 200, 120, key="bp_input")
        diastolic = st.number_input("Diastolic BP (mmHg)", 40, 150, 80, key="dias_input")
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75, key="hr_input")
        
        st.session_state.form_data.update({
            'age': age,
            'blood_pressure': blood_pressure,
            'diastolic': diastolic,
            'heart_rate': heart_rate
        })
    
    with col2:
        st.markdown("**Metabolic Indicators**")
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", 4.0, 20.0, 7.0, 0.1, key="bs_input")
        body_temp = st.number_input("Body Temperature (°F)", 95.0, 104.0, 98.6, 0.1, key="temp_input")
        bmi = st.number_input("BMI", 15.0, 45.0, 23.0, 0.1, key="bmi_input")
        
        st.markdown("**Optional Lab Value**")
        hemoglobin = st.number_input("Hemoglobin (g/dL) - Optional", 0.0, 20.0, 12.0, 0.1, 
                                      help="Normal range: 11-14 g/dL. Leave at 0 if unknown.", key="hb_input")
        
        st.session_state.form_data.update({
            'blood_sugar': blood_sugar,
            'body_temp': body_temp,
            'bmi': bmi,
            'hemoglobin': hemoglobin if hemoglobin > 0 else np.nan
        })

# ========================================
# CLINICAL ASSESSMENT FORM
# ========================================
def clinical_assessment_form():
    st.markdown("### 🏥 Medical History & Risk Factors")
    
    # Medical History
    with st.expander("🩺 Medical Conditions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            asthma = st.checkbox("Asthma", key="asthma_check")
            preexisting_diabetes = st.checkbox("Preexisting Diabetes", key="prediab_check")
            gestational_diabetes = st.checkbox("Gestational Diabetes", key="gestdiab_check")
            advanced_maternal_age = st.checkbox("Advanced Maternal Age (>35 years)", key="age35_check")
            
            st.session_state.form_data.update({
                'asthma': int(asthma),
                'preexisting_diabetes': int(preexisting_diabetes),
                'gestational_diabetes': int(gestational_diabetes),
                'advanced_maternal_age': int(advanced_maternal_age)
            })
        
        with col2:
            chronic_hypertension = st.checkbox("Chronic Hypertension", key="chtn_check")
            pregnancy_induced_hypertension = st.checkbox("Pregnancy-Induced Hypertension", key="pih_check")
            genetic_disorder = st.checkbox("Genetic Disorder", key="genetic_check")
            mental_health = st.checkbox("Mental Health Issues", key="mental_check")
            
            st.session_state.form_data.update({
                'chronic_hypertension': int(chronic_hypertension),
                'pregnancy_induced_hypertension': int(pregnancy_induced_hypertension),
                'genetic_disorder': int(genetic_disorder),
                'mental_health': int(mental_health)
            })
    
    # Obstetric History
    with st.expander("🤰 Obstetric History", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            previous_complications = st.checkbox("Previous Pregnancy Complications", key="prevcomp_check")
            previous_preterm_birth = st.checkbox("Previous Preterm Birth (<5 lbs)", key="preterm_check")
            
            st.session_state.form_data.update({
                'previous_complications': int(previous_complications),
                'previous_preterm_birth': int(previous_preterm_birth)
            })
        
        with col2:
            current_placental_problems = st.checkbox("Current Placental Problems", key="placenta_check")
            medications_affecting_fetus = st.checkbox("Medications That May Affect Fetus", key="meds_check")
            multiple_gestation = st.checkbox("Multiple Gestation (Twins/Triplets)", key="twins_check")
            
            st.session_state.form_data.update({
                'current_placental_problems': int(current_placental_problems),
                'medications_affecting_fetus': int(medications_affecting_fetus),
                'multiple_gestation': int(multiple_gestation)
            })
    
    # Substance Use
    with st.expander("🚬 Substance Use", expanded=True):
        st.warning("⚠️ Honest disclosure helps provide better care. All information is confidential.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alcohol_use = st.checkbox("Alcohol Use", key="alcohol_check")
            tobacco_use = st.checkbox("Tobacco/Smoking", key="tobacco_check")
            
            st.session_state.form_data.update({
                'alcohol_use': int(alcohol_use),
                'tobacco_use': int(tobacco_use)
            })
        
        with col2:
            narcotics_use = st.checkbox("Narcotics/Opioid Use", key="narcotics_check")
            sedatives_use = st.checkbox("Sedatives/Tranquilizers", key="sedatives_check")
            
            st.session_state.form_data.update({
                'narcotics_use': int(narcotics_use),
                'sedatives_use': int(sedatives_use)
            })
    
    # Nutrition
    with st.expander("🍎 Nutrition & Weight", expanded=True):
        pregnancy_weight_issue = st.checkbox("Pregnancy Weight Issue (Under/Overweight or Inadequate Weight Gain)", key="weight_check")
        obstetric_diet_condition = st.checkbox("Medical Condition Requiring Diet Modification", key="diet_check")
        cervical_length_issue = st.checkbox("Short Cervical Length (<25mm)", key="cervix_check")
        
        st.session_state.form_data.update({
            'pregnancy_weight_issue': int(pregnancy_weight_issue),
            'obstetric_diet_condition': int(obstetric_diet_condition),
            'cervical_length': 20 if cervical_length_issue else 35
        })
    
    # Blood Work
    with st.expander("🩸 Blood Work", expanded=True):
        anemia = st.checkbox("Anemia (Low Hemoglobin)", key="anemia_check")
        
        st.session_state.form_data.update({
            'anemia': int(anemia)
        })

# ========================================
# IDENTIFY RISK FACTORS
# ========================================
def identify_risk_factors(form_data):
    """Identify which factors are contributing to high risk"""
    risk_factors = []
    
    # Vital signs
    if form_data.get('blood_pressure', 0) >= 140:
        risk_factors.append("⚠️ Elevated Blood Pressure (≥140 mmHg)")
    
    if form_data.get('diastolic', 0) >= 90:
        risk_factors.append("⚠️ High Diastolic Blood Pressure (≥90 mmHg)")
    
    if form_data.get('blood_sugar', 0) >= 10:
        risk_factors.append("⚠️ Elevated Blood Sugar (≥10 mg/dL)")
    
    if form_data.get('bmi', 0) >= 30:
        risk_factors.append("⚠️ High BMI (≥30 - Obesity)")
    elif form_data.get('bmi', 0) < 18.5:
        risk_factors.append("⚠️ Low BMI (<18.5 - Underweight)")
    
    # Medical conditions
    if form_data.get('preexisting_diabetes'):
        risk_factors.append("🔴 Preexisting Diabetes")
    
    if form_data.get('gestational_diabetes'):
        risk_factors.append("🔴 Gestational Diabetes")
    
    if form_data.get('chronic_hypertension'):
        risk_factors.append("🔴 Chronic Hypertension")
    
    if form_data.get('pregnancy_induced_hypertension'):
        risk_factors.append("🔴 Pregnancy-Induced Hypertension")
    
    if form_data.get('asthma'):
        risk_factors.append("🔴 Asthma")
    
    if form_data.get('genetic_disorder'):
        risk_factors.append("🔴 Genetic Disorder")
    
    if form_data.get('mental_health'):
        risk_factors.append("🔴 Mental Health Issues")
    
    # Obstetric history
    if form_data.get('previous_complications'):
        risk_factors.append("📋 Previous Pregnancy Complications")
    
    if form_data.get('previous_preterm_birth'):
        risk_factors.append("📋 Previous Preterm Birth")
    
    if form_data.get('current_placental_problems'):
        risk_factors.append("📋 Current Placental Problems")
    
    if form_data.get('multiple_gestation'):
        risk_factors.append("📋 Multiple Gestation (Twins/Triplets)")
    
    # Substance use
    if form_data.get('tobacco_use'):
        risk_factors.append("🚬 Tobacco Use")
    
    if form_data.get('alcohol_use'):
        risk_factors.append("🍺 Alcohol Use")
    
    if form_data.get('narcotics_use'):
        risk_factors.append("💊 Narcotics/Opioid Use")
    
    # Other factors
    if form_data.get('anemia'):
        risk_factors.append("🩸 Anemia")
    
    if form_data.get('pregnancy_weight_issue'):
        risk_factors.append("⚖️ Pregnancy Weight Issue")
    
    if form_data.get('advanced_maternal_age'):
        risk_factors.append("👵 Advanced Maternal Age (>35 years)")
    
    return risk_factors

# ========================================
# PERFORM RISK ANALYSIS
# ========================================
def perform_risk_analysis(model, scaler, label_encoder, feature_cols):
    st.markdown("### 📊 Risk Analysis Results")
    
    # Prepare input data
    input_df = pd.DataFrame([st.session_state.form_data])
    
    # Fill missing features with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder to match model's expected order
    X = input_df[feature_cols]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    risk_level = label_encoder.inverse_transform([prediction])[0]
    risk_prob = probabilities[prediction]

    # Display Risk Result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if risk_level.lower() == 'high':
            result_html = f"""
            <div class="risk-card-high">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <h3>{risk_prob:.1%} Confidence</h3>
                <p>Immediate medical attention recommended</p>
            </div>
            """
        else:
            result_html = f"""
            <div class="risk-card-low">
                <h2>✅ LOW RISK</h2>
                <h3>{risk_prob:.1%} Confidence</h3>
                <p>Continue routine prenatal care</p>
            </div>
            """
        
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Risk breakdown 
        st.markdown("")
        st.write("")
        
        risk_classes = label_encoder.classes_
        fig_probs = go.Figure(data=[
            go.Bar(
                x=risk_classes,
                y=probabilities,
                marker=dict(
                    color=['#FF6B6B' if prob > 0.5 else '#7BED8D' for prob in probabilities],
                    line=dict(color='#000000', width=2)
                ),
                text=[f'{prob:.1%}' for prob in probabilities],
                textposition='outside',  # Changed from 'auto'
                textfont=dict(color='#000000', size=16, family='Poppins', weight='bold'),
                constraintext='none'
            )
        ])
        fig_probs.update_layout(
            title={
                'text': "Risk Distribution",
                'font': {'size': 18, 'color': '#000000', 'family': 'Poppins'}
            },
            xaxis_title="Risk Category",
            yaxis_title="Probability",
            xaxis=dict(
                title_font=dict(size=14, color='#000000', weight='bold'),
                tickfont=dict(size=14, color='#000000', weight='bold')
            ),
            yaxis=dict(
                title_font=dict(size=14, color='#000000'),
                tickfont=dict(size=12, color='#000000'),
                range=[0, max(probabilities) * 1.15]  # Add space for labels
            ),
            height=350,
            showlegend=False,
            paper_bgcolor="rgba(255,255,255,0.7)",
            plot_bgcolor="rgba(255,255,255,0.7)"
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    
    with col2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={'text': "Risk Score", 'font': {'size': 20, 'color': '#000000'}},
            number={'font': {'size': 40, 'color': '#000000'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#000000"},
                'bar': {'color': "#FF6B6B"},
                'steps': [
                    {'range': [0, 33], 'color': "#7BED8D"},
                    {'range': [33, 66], 'color': "#FFD700"},
                    {'range': [66, 100], 'color': "#FF6B6B"}
                ],
                'threshold': {
                    'line': {'color': "#000000", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(255,255,255,0.7)",
            font={'color': '#000000', 'family': 'Poppins'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk factor count
        st.write("")
        
        medical_count = sum([
            st.session_state.form_data.get('asthma', 0),
            st.session_state.form_data.get('preexisting_diabetes', 0),
            st.session_state.form_data.get('gestational_diabetes', 0),
            st.session_state.form_data.get('chronic_hypertension', 0),
            st.session_state.form_data.get('pregnancy_induced_hypertension', 0)
        ])
        
        obstetric_count = sum([
            st.session_state.form_data.get('previous_complications', 0),
            st.session_state.form_data.get('previous_preterm_birth', 0),
            st.session_state.form_data.get('current_placental_problems', 0)
        ])
        
        substance_count = sum([
            st.session_state.form_data.get('alcohol_use', 0),
            st.session_state.form_data.get('tobacco_use', 0),
            st.session_state.form_data.get('narcotics_use', 0)
        ])
        
        st.metric("Medical Conditions", medical_count)
        st.metric("Obstetric History", obstetric_count)
        st.metric("Substance Use", substance_count)
    
    # RISK FACTORS INFO BOX
    if risk_level.lower() == 'high':
        st.markdown("---")
        risk_factors = identify_risk_factors(st.session_state.form_data)
        
        if risk_factors:
            st.markdown(f"""
            <div class="risk-factors-box">
                <h4>⚠️ Identified Risk Factors Causing High Risk:</h4>
                <p>The following factors are contributing to the high-risk assessment:</p>
                <ul>
                    {''.join([f'<li>{factor}</li>' for factor in risk_factors])}
                </ul>
                <p><strong>Total Risk Factors Identified: {len(risk_factors)}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 🎯 Personalized Recommendations")
    
    recommendations = generate_recommendations(st.session_state.form_data, risk_level)
    
    for rec in recommendations:
        st.markdown(f"{rec}")
    
    # Export option
    st.markdown("---")
    if st.button("📄 Export Assessment Report", key="export_btn"):
        report_data = {
            'Assessment_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Risk_Level': risk_level,
            'Confidence': f"{risk_prob:.2%}",
            **st.session_state.form_data
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv,
            file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_btn"
        )

# ========================================
# GENERATE RECOMMENDATIONS
# ========================================
def generate_recommendations(form_data, risk_level):
    recommendations = []
    
    if risk_level.lower() == 'high':
        recommendations.append("**🏥 Urgent:** Schedule immediate consultation with high-risk pregnancy specialist")
        recommendations.append("**📅 Monitoring:** Weekly prenatal visits recommended")
    
    if form_data.get('preexisting_diabetes') or form_data.get('gestational_diabetes'):
        recommendations.append("**💉 Diabetes:** Daily glucose monitoring and endocrinology consult")
    
    if form_data.get('pregnancy_induced_hypertension') or form_data.get('blood_pressure', 0) >= 140:
        recommendations.append("**💊 Blood Pressure:** Monitor BP twice daily, medication may be needed")
    
    if form_data.get('asthma'):
        recommendations.append("**🫁 Asthma:** Review asthma action plan with healthcare provider")
    
    if any([form_data.get('alcohol_use'), form_data.get('tobacco_use'), form_data.get('narcotics_use')]):
        recommendations.append("**🚭 Substance Use:** Cessation counseling and support services recommended")
    
    if form_data.get('mental_health'):
        recommendations.append("**🧠 Mental Health:** Counseling or psychiatric support recommended")
    
    if form_data.get('pregnancy_weight_issue'):
        recommendations.append("**🍎 Nutrition:** Consult with dietitian for meal planning")
    
    if form_data.get('anemia'):
        recommendations.append("**💊 Anemia:** Iron supplementation and dietary counseling")
    
    if not recommendations:
        recommendations.append("**✅ Continue** routine prenatal care with standard monitoring")
    
    return recommendations

# ========================================
# MATERNAL NEWS PAGE
# ========================================
def maternal_news_page():
    st.markdown("### 📰 Latest Maternal Health News")
    
    news_articles = fetch_maternal_health_news()
    
    if not news_articles:
        st.info("📡 Could not fetch news. Please check your NewsAPI key.")
        st.markdown("""
        **Pregnancy Care Topics:**
        - Regular prenatal check-ups
        - Nutrition during pregnancy
        - Safe exercises
        - Mental health support
        - Post-pregnancy recovery
        """)
    else:
        for i, article in enumerate(news_articles, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"#### {article.get('title', 'No title')}")
                    st.write(article.get('description', 'No description')[:200] + "...")
                    st.caption(f"Source: {article.get('source', {}).get('name', 'Unknown')} | {article.get('publishedAt', 'N/A')[:10]}")
                with col2:
                    if st.button("Read More", key=f"news_{i}"):
                        st.markdown(f"[Open Article]({article.get('url', '#')})")
            st.markdown("---")

# ========================================
# AI ASSISTANT PAGE - FIXED
# ========================================
def ai_assistant_page():
    st.markdown("### 🤖 AI Health Assistant")
    
    client = get_ai_client()
    
    if not client:
        st.error("❌ AI Assistant not configured. Please follow these steps:")
        st.markdown("""
        **Setup Instructions:**
        1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a `.env` file in your project folder: `c:\\Users\\keert\\Documents\\ML\\.env`
        3. Add this line: `GEMINI_API_KEY=your_key_here`
        4. Restart the Streamlit app
        """)
        st.info("💡 You can still use the quick questions below without an API key!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat" not in st.session_state and client:
        st.session_state.chat = client.start_chat(history=[])
    
    st.markdown("#### ❓ Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is high blood pressure in pregnancy?", key="faq1"):
            answer = "**High Blood Pressure in Pregnancy:**\n\nBlood pressure ≥140/90 mmHg is considered elevated. Normal is <120/80 mmHg. Hypertension during pregnancy can lead to complications like preeclampsia. Monitor BP regularly and contact your doctor if readings are consistently high."
            st.info(answer)
    
        if st.button("How often are prenatal visits?", key="faq2"):
            answer = "**Prenatal Visit Schedule:**\n\n• Weeks 1-28: Monthly visits\n• Weeks 28-36: Bi-weekly visits\n• Week 36+: Weekly visits\n\nHigh-risk pregnancies may require more frequent monitoring."
            st.info(answer)
    
    with col2:
        if st.button("Is exercise safe during pregnancy?", key="faq3"):
            answer = "**Exercise During Pregnancy:**\n\nYes! 150 minutes of moderate activity per week is generally safe. Recommended: walking, swimming, prenatal yoga. Avoid: contact sports, lying flat after first trimester, activities with fall risk."
            st.info(answer)
        
        if st.button("When to contact doctor immediately?", key="faq4"):
            answer = "**Emergency Warning Signs:**\n\n🚨 Severe headache\n🚨 Vision changes/blurred vision\n🚨 Chest pain\n🚨 Severe abdominal pain\n🚨 Vaginal bleeding\n🚨 Decreased fetal movement\n🚨 Severe swelling of hands/face"
            st.info(answer)
    
    if not client:
        return
    
    st.markdown("---")
    st.markdown("#### 💬 Ask Your Questions")
    
    # Display recent messages
    for message in st.session_state.messages[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Ask about pregnancy and maternal health...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            with st.spinner("🤔 Thinking..."):
                # Create context-aware prompt
                system_context = """You are a knowledgeable and empathetic maternal health assistant. 
                Provide accurate, evidence-based information about pregnancy and maternal health. 
                Always recommend consulting healthcare professionals for specific medical advice. 
                Be supportive, clear, and avoid medical jargon when possible. 
                If unsure, say so and recommend professional consultation."""
                
                # Combine context with user input
                full_prompt = f"{system_context}\n\nUser question: {user_input}"
                
                response = st.session_state.chat.send_message(full_prompt)
                assistant_message = response.text
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            
            with st.chat_message("assistant"):
                st.markdown(assistant_message)
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error communicating with AI: {str(e)}")
            st.info("Please check your API key and internet connection.")
            
            # Add fallback response
            fallback = "I'm having trouble connecting right now. Please consult with your healthcare provider for personalized medical advice."
            st.session_state.messages.append({"role": "assistant", "content": fallback})

# ========================================
# MODEL COMPARISON PAGE
# ========================================
def model_comparison_page():
    st.markdown("### 🤖 ML Algorithm Comparison")
    
    try:
        comparison_df = pd.read_csv("model_comparison_results.csv")
        
        st.dataframe(
            comparison_df.sort_values('Accuracy', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                comparison_df.sort_values('Accuracy', ascending=True),
                x='Accuracy',
                y='Algorithm',
                orientation='h',
                title='Algorithm Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
                text='Accuracy'  # Add text labels
            )
            fig_bar.update_traces(
                texttemplate='%{text:.4f}',
                textposition='outside',
                textfont=dict(size=14, color='#000000', family='Poppins', weight='bold')
            )
            fig_bar.update_layout(
                height=400,
                paper_bgcolor="rgba(255,255,255,0.7)",
                plot_bgcolor="rgba(255,255,255,0.9)",
                font=dict(color='#000000', family='Poppins', size=12),
                xaxis=dict(
                    title_font=dict(size=14, color='#000000', weight='bold'),
                    tickfont=dict(size=12, color='#000000', weight='bold'),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color='#000000', weight='bold'),
                    tickfont=dict(size=12, color='#000000', weight='bold')
                ),
                margin=dict(r=80)  # Add right margin for labels
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                comparison_df,
                values='Accuracy',
                names='Algorithm',
                title='Accuracy Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont=dict(size=14, color='#000000', family='Poppins', weight='bold'),
                marker=dict(line=dict(color='#000000', width=2))
            )
            fig_pie.update_layout(
                height=400,
                paper_bgcolor="rgba(255,255,255,0.7)",
                font=dict(color='#000000', family='Poppins', size=12),
                legend=dict(
                    font=dict(size=12, color='#000000', weight='bold'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#000000',
                    borderwidth=2
                ),
                title_font=dict(size=16, color='#000000', weight='bold')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    except FileNotFoundError:
        st.warning("⚠️ Model comparison results not found. Run model.py first.")

# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    main()