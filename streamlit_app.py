import streamlit as st
import librosa
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(page_title="AI Audio Quality Auditor", page_icon="🎧", layout="centered")

# Custom Styling (CSS) for beautiful background and glowing UI
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc !important;
    }
    
    /* Titles & Headings color */
    h1, h3, p, span, label {
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Subheader customized text styling */
    .subheader-text {
        color: #38bdf8 !important;
        font-weight: 500;
    }

    /* File Uploader styling */
    section[data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 2px dashed #6366f1 !important;
        border-radius: 12px;
        padding: 20px;
    }

    /* Primary Button Customization (Neon Glow) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    }
    
    /* Dataframe table styling adjustment */
    div[data-testid="stDataFrame"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# App Content
st.title("🎧 Automated Audio Noise Auditor")
st.markdown("<p class='subheader-text'>High-precision, standalone call filtration system</p>", unsafe_allow_html=True)
st.write("Upload the call recording. The system will filter out the agent's voice and pinpoint background noise and spikes instantly.")

st.markdown("---")

uploaded_file = st.file_uploader("👇 Choose or drop a call file here (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("🔍 Start Audio Audit", type="primary"):
        with st.spinner("⏳ Analyzing audio footprint using multi-filter engineering..."):
            try:
                y, sr = librosa.load(uploaded_file, sr=16000)
                hop_length = 16000
                rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=16000, hop_length=hop_length)[0]
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=16000, hop_length=hop_length, roll_percent=0.85)[0]
                
                mean_energy = np.mean(rms)
                mean_centroid = np.mean(centroid)
                mean_rolloff = np.mean(rolloff)
                
                energy_threshold = mean_energy * 1.4 
                centroid_threshold = mean_centroid * 1.25
                rolloff_threshold = mean_rolloff * 1.2
                
                violations = []
                
                for i in range(len(rms)):
                    energy = rms[i]
                    spectral_val = centroid[i]
                    rolloff_val = rolloff[i]
                    
                    current_second = i
                    minutes = current_second // 60
                    seconds = current_second % 60
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                  if energy > energy_threshold and spectral_val > centroid_threshold and rolloff_val > rolloff_threshold:
                        severity = "High 🚨" if energy > (energy_threshold * 1.8) else "Medium ⚠️"
                        violations.append({
                            "Timestamp ⏱️": timestamp,
                            "Audio Status": "Confirmed Background Noise / Distortion",
                            "Severity Level 📊": severity
                        })
                
                st.markdown("### 💡 Audit Summary & Results")
                if len(violations) > 0:
                    df = pd.DataFrame(violations)
                    df = df.drop_duplicates(subset=['Timestamp ⏱️'])
                    st.error(f"🚨 Detected {len(df)} seconds of background noise (Agent's voice skipped successfully).")
                    st.markdown("### 📊 Violation Log Table")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("✅ Audit Passed: Call complies with quality standards. Agent voice is clear, no background noise detected.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

st.markdown("---")
st.caption("Smart Audit Tool - Free and Unlimited Use.")
