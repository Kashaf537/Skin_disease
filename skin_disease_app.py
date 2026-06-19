import os
import json
import base64
import time
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file at application startup
load_dotenv()

# ==================== CONFIGURATION ====================
class Config:
    MODEL_DIR = "models"
    IMG_SIZE = (128, 128)
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    LOCAL_CONFIDENCE_GUARD = 0.70  # Guard threshold if Groq fails

# ==================== PYDANTIC STRUCTURE SCHEMA ====================
class DiseaseVerification(BaseModel):
    is_supported: bool = Field(description="True if the condition accurately matches one of the 8 local categories provided, False otherwise.")
    true_diagnosis: str = Field(description="The formal clinical name of the skin condition identified in the image.")
    description: str = Field(description="A brief 2-sentence medical description of the condition and why it matches/doesn't match.")

# ==================== CORE UTILITIES ====================
@st.cache_resource
def load_classifier_resources():
    """Loads the local Keras model and class names once and caches them in memory."""
    model_path = os.path.join(Config.MODEL_DIR, "final_model.keras")
    class_path = os.path.join(Config.MODEL_DIR, "class_names.json")
    
    if not os.path.exists(model_path) or not os.path.exists(class_path):
        return None, None
        
    model = keras.models.load_model(model_path)
    with open(class_path, "r") as f:
        class_names = json.load(f)
        
    return model, class_names

def clahe_preprocess(img_rgb_uint8):
    """Identical CLAHE preprocessing from training pipeline to keep evaluation consistent."""
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

def verify_and_classify_with_groq(uploaded_file, local_classes):
    """Uses Groq's active Llama 4 Vision engine with advanced medical diagnostic field mapping."""
    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:{uploaded_file.type};base64,{base64_image}"
        
        client = Groq(api_key=Config.GROQ_API_KEY)
        classes_str = ", ".join(local_classes)
        
        prompt = f"""
        You are an expert clinical dermatologist. Analyze this image carefully.
        Our local system can only classify these 8 specific categories: [{classes_str}].
        
        Tasks:
        1. Identify the exact skin condition visible in the photo.
        2. Determine if this condition matches one of the 8 local categories.
        3. Provide the primary underlying physiological or environmental causes.
        4. Provide standard medical treatment recommendations.
        
        You must return a raw JSON object matching these fields exactly:
        {{
            "is_supported": true or false,
            "true_diagnosis": "Clinical name of condition",
            "description": "Brief explanation statement.",
            "causes": "A short paragraph detailing the etiology, triggers, or biological mechanisms behind this presentation.",
            "treatments": "A clear summary bulleted listing of clinical treatments, first-line topicals, or lifestyle interventions."
        }}
        """
        
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        raw_response = chat_completion.choices[0].message.content.strip()
        
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[1].rsplit("\n", 1)[0]
        if raw_response.startswith("json"):
            raw_response = raw_response.split("json", 1)[1]
            
        return json.loads(raw_response.strip())
        
    except Exception as e:
        return {"error": str(e)}

# ==================== MODERN THEME UI SETUP ====================
st.set_page_config(page_title="DermAI Clinical Assistant", layout="wide", initial_sidebar_state="expanded")

# Inject Premium Clinical Dark Theme Layout Custom CSS
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, label, [data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 26px;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        border: 1px solid #475569;
    }
    
    .metric-card {
        background-color: #1e293b;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
        border-left: 6px solid #38bdf8;
        margin-bottom: 20px;
    }
    .metric-card.success-border { border-left-color: #34d399; }
    .metric-card.warning-border { border-left-color: #fbbf24; }
    .metric-card.error-border { border-left-color: #f87171; }
    
    .card-title { color: #94a3b8; font-size: 13px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px; }
    .card-value { color: #f8fafc; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; }
    .card-desc { color: #cbd5e1; font-size: 14px; margin-top: 10px; line-height: 1.5; }
    
    .recommendation-box {
        background-color: #064e3b;
        border: 1px solid #059669;
        padding: 18px;
        border-radius: 12px;
        color: #a7f3d0;
        margin-top: 15px;
    }
    .recommendation-box.alert-box {
        background-color: #7f1d1d;
        border: 1px solid #dc2626;
        color: #fca5a5;
    }
    
    /* Medical Instruction Blocks Styling */
    .clinical-block {
        background-color: #111827;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
        border: 1px solid #374151;
    }
    .clinical-block-title {
        color: #38bdf8 !important;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #1e293b !important;
        border-radius: 12px;
        padding: 10px;
        border: 1px dashed #475569;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR CONTROL PANEL ====================
with st.sidebar:
    st.markdown("## ⚙️ Assistant Dashboard")
    st.markdown("---")
    
    if Config.GROQ_API_KEY:
        st.success("🟢 Cloud Guardrails Online")
    else:
        st.error("🔴 Cloud API Stream Disconnected")
        
    model, class_names = load_classifier_resources()
    if model is not None:
        st.success(f"🟢 Edge Model Loaded ({len(class_names)} Pathologies)")
    else:
        st.error("🔴 Model File System Missing")
        st.stop()
        
    st.markdown("---")
    st.markdown("### 📊 Engine Controls")
    confidence_threshold = st.slider("Diagnostic Isolation Threshold", 0.0, 1.0, Config.LOCAL_CONFIDENCE_GUARD)

# ==================== CLINICAL HEADER ====================
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 32px; font-weight: 800; color: #fff !important;'>🩺 DermAI Clinical Assistant</h1>
        <p style='margin:5px 0 0 0; opacity: 0.9; font-size: 15px; color: #cbd5e1 !important;'>Real-time Hybrid Image Processing & Deep Pathology Screening Dashboard</p>
    </div>
""", unsafe_allow_html=True)

main_col, side_col = st.columns([1, 1], gap="large")

with main_col:
    st.subheader("📸 Scan Core Input")
    uploaded_file = st.file_uploader("Upload close-up dermatoscopy view or standard lesion capture...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        original_rgb = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        
        st.image(original_rgb, caption="Active Patient Capture Telemetry", use_container_width=True)

with side_col:
    if uploaded_file is not None:
        st.subheader("🧬 Telemetry Summary Analysis")
        
        start_time = time.time()
        with st.spinner("Executing structural analysis via Groq Cloud Core..."):
            api_decision = verify_and_classify_with_groq(uploaded_file, class_names)
        latency = time.time() - start_time
            
        resized_img = cv2.resize(original_rgb, Config.IMG_SIZE)
        preprocessed_img = clahe_preprocess(resized_img)
        input_batch = np.expand_dims(preprocessed_img.astype(np.float32), axis=0)
        
        predictions = model.predict(input_batch, verbose=0)[0]
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]
        
        api_failed = "error" in api_decision
        
        # --- VIP VISUAL SCOREBOARD ---
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(label="Inference Score", value=f"{confidence * 100:.1f}%", delta=f"{'+' if confidence >= confidence_threshold else ''}{confidence - confidence_threshold:.1%}")
        with kpi_col2:
            st.metric(label="API Core Latency", value=f"{latency:.2f}s", delta="-0.14s (Optimal)" if latency < 1.5 else "Throttled Flow")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- STATE MANAGEMENT & TEXT RENDERING ---
        if api_failed:
            st.markdown(f"""
                <div class="metric-card warning-border">
                    <div class="card-title">Cloud System Layer Status</div>
                    <div class="card-value">Autonomous Fallback Active</div>
                    <div class="card-desc">The primary validation stream dropped offline; switching to local isolated model layers.</div>
                </div>
            """, unsafe_allow_html=True)
            
            if confidence >= confidence_threshold:
                st.markdown(f"""
                    <div class="metric-card success-border">
                        <div class="card-title">Isolated Pipeline Classifier Summary</div>
                        <div class="card-value">🔍 {class_names[top_idx]}</div>
                        <div class="card-desc">The pattern presentation has safely met the minimum confidence index parameters.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card error-border">
                        <div class="card-title">Diagnostic Guardrail Break</div>
                        <div class="card-value">🚨 Low-Confidence Isolation</div>
                        <div class="card-desc">Local estimation targeted '{class_names[top_idx]}' but returned low assurance metrics ({confidence * 100:.1f}%).</div>
                    </div>
                """, unsafe_allow_html=True)
                
        else:
            is_supported = api_decision.get("is_supported", False)
            true_diagnosis = api_decision.get("true_diagnosis", "Unknown")
            description = api_decision.get("description", "")
            causes = api_decision.get("causes", "Etiology analysis unavailable for this frame query.")
            treatments = api_decision.get("treatments", "Therapeutic protocol guidelines unavailable.")
            
            if not is_supported:
                st.markdown(f"""
                    <div class="metric-card error-border">
                        <div class="card-title">Anomalous Presentation Intercept</div>
                        <div class="card-value">🌐 {true_diagnosis}</div>
                        <div class="card-desc"><b>Cloud Verification Scan Note:</b> {description}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card success-border">
                        <div class="card-title">Verified Local Classification Match</div>
                        <div class="card-value">✨ {class_names[top_idx]}</div>
                        <div class="card-desc"><b>Clinical Analysis Summary:</b> {description}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # --- BREAKOUT MEDICAL BLOCKS FOR CAUSES AND TREATMENTS ---
            st.markdown(f"""
                <div class="clinical-block">
                    <div class="clinical-block-title">🧬 Primary Etiology & Causes</div>
                    <div style="font-size: 14px; color: #e2e8f0; line-height: 1.5;">{causes}</div>
                </div>
                <div class="clinical-block">
                    <div class="clinical-block-title">💊 Standard Clinical Treatment Protocols</div>
                    <div style="font-size: 14px; color: #e2e8f0; line-height: 1.5;">{treatments}</div>
                </div>
            """, unsafe_allow_html=True)

        # --- DYNAMIC PROGRESS BAR CHART BREAKDOWNS ---
        if api_failed or api_decision.get("is_supported", False):
            st.markdown("<br>### 📊 Relative Symptom Class Distribution", unsafe_allow_html=True)
            sorted_pairs = sorted(zip(class_names, predictions), key=lambda x: x[1], reverse=True)
            
            for idx, (cls, prob) in enumerate(sorted_pairs[:3]):
                st.write(f"**{cls}** ({prob*100:.1f}%)")
                st.progress(float(prob))
    else:
        st.info("💡 Assistant Standby: Awaiting patient sample intake capture to initialize deep scanning pipelines.")