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
 +       color: #f8fafc !important;
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


def classify_noise_type(
    centroid: float, zcr: float, rolloff: float, flatness: float
) -> str:
    """
    Classify detected background noise using spectral shape features.
    Spectral flatness near 1.0 → white-noise / static; near 0 → tonal / structured.
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


uploaded_file = st.file_uploader("👇 Choose or drop a call file here (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("🔍 Start Audio Audit", type="primary"):
        with st.spinner("⏳ Analyzing call for explicit background noise..."):
            try:
                # Load audio
                y, sr = librosa.load(uploaded_file, sr=16000)
                hop_length = 16000
                
                # Extract RMS Energy per second
                rms = librosa.feature.rms(y=y, frame_length=16000, hop_length=hop_length)[0]
                
                violations = []
                
                # Calculate statistical baseline to ignore constant background hiss
                mean_energy = np.mean(rms)
                std_energy = np.std(rms)
                
                # Dynamic Threshold: Only catches sudden spikes or clear noise events that deviate from the baseline hiss
                # We filter out very silent segments first to prevent false alarms in quiet calls
                for i in range(len(rms)):
                    energy = rms[i]
                    
                    current_second = i
                    minutes = current_second // 60
                    seconds = current_second % 60
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    # LOGIC: If the second has an energy spike significantly higher than the average call background
                    if energy > (mean_energy + 1.2 * std_energy) and energy > 0.02:
                        # Extract the specific 1-second segment of the raw audio waveform
                        y_sec = y[i * sr : (i + 1) * sr]
                        if len(y_sec) > 0:
                            # Compute spectral features on-the-fly for this second to classify the noise type
                            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_sec, sr=sr)))
                            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_sec)))
                            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_sec, sr=sr)))
                            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_sec)))
                            noise_type = classify_noise_type(centroid, zcr, rolloff, flatness)
                        else:
                            noise_type = "📢 General background disturbance"

                        violations.append({
                            "Timestamp ⏱️": timestamp,
                            "Audio Status": "Detected Explicit Background Noise / Speech Disturbance",
                            "Detected Noise Type": noise_type,
                            "Severity Level 📊": "High 🚨" if energy > (mean_energy + 2.5 * std_energy) else "Medium ⚠️"
                        })
                
                st.markdown("### 💡 Audit Summary & Results")
                if len(violations) > 0:
                    # Extract unique noise types detected
                    unique_noises = sorted(list(set([v["Detected Noise Type"] for v in violations])))
                    
                    st.error("🚨 Call Quality Audit Failed: Environmental Noise Detected")
                    st.markdown("#### Identified Environmental Issues:")
                    for noise in unique_noises:
                        st.markdown(f"- {noise}")
                else:
                    st.success("✅ Quality Audit Passed: Call environment complies with quiet-workspace standards.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

st.markdown("---")
st.caption("Smart Audit Tool - Balanced Precision Edition.")

