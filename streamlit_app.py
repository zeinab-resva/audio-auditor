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

def classify_noise_type(centroid: float, zcr: float, flatness: float) -> str:
    """
    Classify verified background noises, ignoring speech and breathing artifacts.
    """
    if centroid < 500:
        return "💥 Low-frequency Impact / Loud physical thud / Banging"
    elif centroid < 1400 and zcr < 0.08:
        return "🗣️ Background chatter / Environmental overlapping speech"
    else:
        return "📢 Explicit background disturbance / Loud ambient noise"

# Bulk upload configuration
uploaded_files = st.file_uploader(
    "👇 Choose or drop multiple call files here (MP3 / WAV)", 
    type=["mp3", "wav"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📂 Total calls loaded: {len(uploaded_files)} files.")
    
    if st.button("🔍 Start Bulk Audio Audit", type="primary"):
        progress_bar = st.progress(0)
        st.markdown("### 💡 Bulk Audit Summary & Results")
        
        for index, file in enumerate(uploaded_files):
            progress_bar.progress((index + 1) / len(uploaded_files))
            
            with st.container():
                st.markdown(f"<div class='call-box'>", unsafe_allow_html=True)
                st.markdown(f"📁 **File Name:** `{file.name}`")
                
                try:
                    # Load audio with standardized 16kHz sampling
                    y, sr = librosa.load(file, sr=16000)
                    hop_length = 16000
                    
                    # Extract robust features per second
                    rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                    
                    violations = []
                    issue_timestamps = []
                    
                    # Calculate stats to understand the baseline line hiss/static
                    mean_energy = np.mean(rms)
                    std_energy = np.std(rms)
                    
                    for i in range(len(rms)):
                        energy = rms[i]
                        
                        current_second = i
                        minutes = current_second // 60
                        seconds = current_second % 60
                        timestamp = f"{minutes:02d}:{seconds:02d}"
                        
                        # 1. Adjusted baseline check to ensure mild line static during agent silence isn't flagged
                        if energy > (mean_energy + 1.8 * std_energy) and energy > 0.045:
                            
                            # Extract the exact 1-second audio segment for deep-validation
                            y_sec = y[i * sr : (i + 1) * sr]
                            if len(y_sec) > 0:
                                # Compute features for the target second
                                centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_sec, sr=sr)))
                                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_sec)))
                                flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_sec)))
                                
                                # 2. FILTER OUT HEAVY BREATHING / MICROPHONE PROXIMITY (Enhanced thresholds)
                                if zcr > 0.14 or (zcr > 0.10 and flatness > 0.06):
                                    continue
                                    
                                # 3. FILTER OUT AGENT'S LOUD/ENERGETIC VOICE AND HARMONIC SPEECH HISS
                                # Pure environmental noise doesn't follow the precise harmonic structure of voice
                                if 550 < centroid < 2400 and 0.02 < zcr < 0.13:
                                    # If it has the speech footprint and isn't a massive explosion of sound, skip it
                                    if flatness < 0.05 or energy < (mean_energy + 3.2 * std_energy):
                                        continue 
                                
                                # 4. VERIFIED EXPLICIT ENVIRONMENTAL NOISE (Banging or Background Whistling/Chatter)
                                noise_type = classify_noise_type(centroid, zcr, flatness)
                                issue_timestamps.append(timestamp)
                                violations.append(noise_type)
                    
                    # Display the audio player so user can review the clip
                    st.audio(file, format='audio/wav')
                    
                    # NEW CRITERIA: Fail the call only if genuine environmental issues persist for 2 seconds or more
                    if len(violations) >= 2:
                        unique_timestamps = sorted(list(set(issue_timestamps)))
                        unique_noises = sorted(list(set(violations)))
                        timestamps_str = ", ".join(unique_timestamps)
                        
                        st.error(f"❌ **Problem Detected at ({timestamps_str}) -> Identified Issues:**")
                        for noise in unique_noises:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **{noise}**")
                    else:
                        st.success("✅ **Result:** Quality Audit Passed. (Any minor micro-noise under 2 seconds was safely bypassed).")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis of this file: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        st.balloons()

st.markdown("---")
st.caption("Smart Audit Tool - Bulk Precision Edition.")
