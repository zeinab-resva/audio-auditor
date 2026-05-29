import streamlit as st
import librosa
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI Audio Quality Auditor", page_icon="🎧", layout="centered")

# 2. Advanced Custom Styling (CSS for Background + Glowing UI)
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
    .call-box {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
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
st.markdown("<p class='subheader-text'>High-precision, standalone bulk filtration system</p>", unsafe_allow_html=True)

st.markdown("---")


def classify_noise_type(centroid: float, zcr: float, rolloff: float, flatness: float) -> str:
    """
    Classify detected background noise using spectral shape features.
    """
    if flatness > 0.25:
        return "📡 Electronic interference / line static"
    elif centroid < 400:
        return "💥 Low-freq impact / door slam / physical thud"
    elif centroid < 1200 and zcr < 0.09:
        return "🗣️ Background chatter / overlapping speech"
    elif centroid < 2500 and zcr > 0.12:
        return "🎵 Background music"
    elif centroid > 3500:
        return "📡 High-freq hiss / electronic noise"
    elif zcr > 0.20:
        return "🏠 Ambient office / environment noise"
    else:
        return "📢 General background disturbance"


# ENABLED MULTIPLE FILES UPLOAD (accept_multiple_files=True)
uploaded_files = st.file_uploader(
    "👇 Choose or drop multiple call files here (MP3 / WAV)", 
    type=["mp3", "wav"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📂 Total calls loaded: {len(uploaded_files)} files.")
    
    if st.button("🔍 Start Bulk Audio Audit", type="primary"):
        # Progress bar for visual monitoring
        progress_bar = st.progress(0)
        
        st.markdown("### 💡 Bulk Audit Summary & Results")
        
        for index, file in enumerate(uploaded_files):
            # Update progress bar status
            progress_bar.progress((index + 1) / len(uploaded_files))
            
            # Create a nice container block for each call result
            with st.container():
                st.markdown(f"<div class='call-box'>", unsafe_allow_html=True)
                st.markdown(f"📁 **File Name:** `{file.name}`")
                
                try:
                    # Load audio
                    y, sr = librosa.load(file, sr=16000)
                    hop_length = 16000
                    
                    # Extract RMS Energy per second
                    rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                    
                    violations = []
                    issue_timestamps = []
                    
                    # Calculate statistical baseline to ignore constant background hiss
                    mean_energy = np.mean(rms)
                    std_energy = np.std(rms)
                    
                    for i in range(len(rms)):
                        energy = rms[i]
                        
                        current_second = i
                        minutes = current_second // 60
                        seconds = current_second % 60
                        timestamp = f"{minutes:02d}:{seconds:02d}"
                        
                        # LOGIC: Catch explicit spikes while ignoring line hiss
                        if energy > (mean_energy + 1.2 * std_energy) and energy > 0.02:
                            issue_timestamps.append(timestamp)
                            
                            # Extract segment for classification
                            y_sec = y[i * sr : (i + 1) * sr]
                            if len(y_sec) > 0:
                                centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_sec, sr=sr)))
                                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_sec)))
                                rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_sec, sr=sr)))
                                flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_sec)))
                                noise_type = classify_noise_type(centroid, zcr, rolloff, flatness)
                            else:
                                noise_type = "📢 General background disturbance"

                            violations.append(noise_type)
                    
                    if len(violations) > 0:
                        unique_timestamps = sorted(list(set(issue_timestamps)))
                        unique_noises = sorted(list(set(violations)))
                        timestamps_str = ", ".join(unique_timestamps)
                        
                        st.error(f"❌ **Problem Detected at ({timestamps_str}) -> Identified Issues:**")
                        for noise in unique_noises:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **{noise}**")
                    else:
                        st.success("✅ **Result:** Quality Audit Passed. Call environment complies with quiet-workspace standards.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis of this file: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        st.balloons() # Dynamic celebration effect when all 100 calls are done!

st.markdown("---")
st.caption("Smart Audit Tool - Bulk Precision Edition.")
