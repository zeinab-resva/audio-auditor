import streamlit as st
import librosa
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI Audio Quality Auditor", page_icon="🎧", layout="centered")

# 2. Advanced Custom Styling (CSS for Background + Floating Particles Animation + Glowing UI)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #3b0764, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc !important;
        overflow-x: hidden;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .audio-wave-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 4px;
        height: 40px;
        margin-bottom: 20px;
    }
    .wave-bar {
        width: 4px;
        height: 10px;
        background: linear-gradient(to top, #6366f1, #a855f7);
        animation: wave 1.2s ease-in-out infinite;
        border-radius: 2px;
    }
    .wave-bar:nth-child(2) { animation-delay: 0.2s; height: 25px; }
    .wave-bar:nth-child(3) { animation-delay: 0.4s; height: 35px; }
    .wave-bar:nth-child(4) { animation-delay: 0.6s; height: 15px; }
    .wave-bar:nth-child(5) { animation-delay: 0.8s; height: 30px; }
    @keyframes wave {
        0%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(2.5); }
    }
    h1, h3, p, span, label {
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .subheader-text {
        color: #38bdf8 !important;
        font-weight: 500;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    section[data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed #a855f7 !important;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.2);
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.5);
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.8);
    }
    div[data-testid="stDataFrame"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border-radius: 12px;
        border: 1px solid #475569;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="audio-wave-container">
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎧 Automated Audio Noise Auditor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-text'>High-precision, standalone call filtration system</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("👇 Choose or drop a call file here (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("🔍 Start Audio Audit", type="primary"):
        with st.spinner("⏳ Running ultra-sensitive acoustic audit filters..."):
            try:
                # Load audio
                y, sr = librosa.load(uploaded_file, sr=16000)
                hop_length = 16000
                
                # Extract Audio Features per second
                rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=16000, hop_length=hop_length)[0]
                
                violations = []
                
                # Dynamic Threshold Calculation with a more aggressive approach
                mean_energy = np.mean(rms)
                mean_centroid = np.mean(centroid)
                
                # Sensitive baseline criteria
                ENERGY_LIMIT = mean_energy * 1.1
                CENTROID_LIMIT = mean_centroid * 1.05
                
                for i in range(len(rms)):
                    energy = rms[i]
                    spectral_val = centroid[i]
                    
                    current_second = i
                    minutes = current_second // 60
                    seconds = current_second % 60
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    # ENHANCED LOGIC: Detects high static/distortion OR sudden background noise spikes
                    if (energy > ENERGY_LIMIT and spectral_val > CENTROID_LIMIT) or (spectral_val > mean_centroid * 1.3):
                        violations.append({
                            "Timestamp ⏱️": timestamp,
                            "Audio Status": "Detected Background Noise / Static Distortion",
                            "Severity Level 📊": "High 🚨" if spectral_val > (mean_centroid * 1.4) else "Medium ⚠️"
                        })
                
                st.markdown("### 💡 Audit Summary & Results")
                if len(violations) > 0:
                    df = pd.DataFrame(violations)
                    df = df.drop_duplicates(subset=['Timestamp ⏱️'])
                    st.error(f"🚨 Detected {len(df)} seconds containing background noise or distortions.")
                    st.markdown("### 📊 Violation Log Table")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("✅ Audit Passed: Call complies with strict quality standards. No micro-noise detected.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

st.markdown("---")
st.caption("Smart Audit Tool - Ultra Sensitive Edition.")
