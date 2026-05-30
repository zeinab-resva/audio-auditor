import io
import hashlib

import streamlit as st
import librosa
import numpy as np

# ---------------------------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Audio Quality Auditor",
    page_icon="🎧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 2. Styling
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #3b0764, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc !important;
        overflow-x: hidden;
    }
    @keyframes gradientBG {
        0%   { background-position: 0%   50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0%   50%; }
    }

    /* ── Wave animation ── */
    .audio-wave-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 5px;
        height: 50px;
        margin-bottom: 16px;
    }
    .wave-bar {
        width: 5px;
        background: linear-gradient(to top, #6366f1, #a855f7);
        border-radius: 3px;
        animation: wave 1.2s ease-in-out infinite;
    }
    .wave-bar:nth-child(1) { height: 12px; animation-delay: 0.0s; }
    .wave-bar:nth-child(2) { height: 28px; animation-delay: 0.2s; }
    .wave-bar:nth-child(3) { height: 40px; animation-delay: 0.4s; }
    .wave-bar:nth-child(4) { height: 20px; animation-delay: 0.6s; }
    .wave-bar:nth-child(5) { height: 34px; animation-delay: 0.8s; }
    .wave-bar:nth-child(6) { height: 16px; animation-delay: 1.0s; }
    .wave-bar:nth-child(7) { height: 30px; animation-delay: 0.3s; }
    @keyframes wave {
        0%, 100% { transform: scaleY(1);   }
        50%       { transform: scaleY(2.2); }
    }

    /* ── Typography ── */
    h1, h2, h3, h4, p, span, label { color: #f1f5f9 !important; }
    .subheader-text {
        color: #38bdf8 !important;
        font-weight: 500;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* ── File uploader ── */
    section[data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed #a855f7 !important;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 0 18px rgba(168, 85, 247, 0.25);
        transition: box-shadow 0.3s;
    }
    section[data-testid="stFileUploader"]:hover {
        box-shadow: 0 0 28px rgba(168, 85, 247, 0.5);
    }

    /* ── Buttons ── */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.5);
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.8);
    }

    /* ── Metric cards ── */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa !important; }
    .metric-label { font-size: 0.85rem; color: #94a3b8 !important; margin-top: 4px; }

    /* ── Expander header ── */
    details summary {
        background: rgba(30, 27, 75, 0.7) !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
    }

    /* ── Divider ── */
    hr { border-color: #334155 !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. Header
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="audio-wave-container">
        <div class="wave-bar"></div><div class="wave-bar"></div>
        <div class="wave-bar"></div><div class="wave-bar"></div>
        <div class="wave-bar"></div><div class="wave-bar"></div>
        <div class="wave-bar"></div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎧 Automated Audio Noise Auditor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-text'>High-precision, standalone bulk filtration system</p>", unsafe_allow_html=True)
st.markdown("---")


# ---------------------------------------------------------------------------
# 4. Core analysis logic
# ---------------------------------------------------------------------------

def _file_hash(data: bytes) -> str:
    """Return a short SHA-256 hex digest for cache keying."""
    return hashlib.sha256(data).hexdigest()[:16]


def classify_noise_type(centroid: float, zcr: float, flatness: float) -> str:
    """
    Map spectral features to a human-readable noise category.

    Categories (ordered by increasing centroid / randomness):
    - Low-frequency thud/impact   – very low centroid
    - Background chatter          – low-mid centroid, low ZCR (voiced)
    - White noise / static hiss   – high flatness (energy spread evenly)
    - Explicit ambient disturbance – everything else that passed the filters
    """
    if centroid < 500:
        return "💥 Low-frequency impact (thud / banging)"
    if flatness > 0.25:
        return "📻 White noise / electrical static hiss"
    if centroid < 1_400 and zcr < 0.08:
        return "🗣️ Background chatter / overlapping speech"
    return "📢 Loud ambient disturbance (door slam, alarm, etc.)"


@st.cache_data(show_spinner=False)
def analyze_audio(file_bytes: bytes, filename: str) -> dict:
    """
    Analyse a raw audio byte-string and return a result dictionary.

    Returns
    -------
    dict with keys:
        duration_sec  – total audio duration in seconds
        violations    – list of {"timestamp": str, "noise": str, "energy": float}
        passed        – bool
        error         – str | None
    """
    try:
        audio_io = io.BytesIO(file_bytes)
        sr_target = 16_000
        y, sr = librosa.load(audio_io, sr=sr_target, mono=True)

        # --- Per-second RMS using short frames, averaged into 1-s bins --------
        frame_length = 512            # ~32 ms at 16 kHz — better resolution
        hop_length   = 256            # 50 % overlap
        rms_frames   = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Map frames → seconds
        times = librosa.frames_to_time(
            np.arange(len(rms_frames)), sr=sr, hop_length=hop_length
        )
        n_seconds = int(np.ceil(times[-1])) if len(times) > 0 else 0

        # Aggregate RMS per integer second
        rms_per_sec = np.zeros(n_seconds + 1)
        counts      = np.zeros(n_seconds + 1)
        for t, r in zip(times, rms_frames):
            s = int(t)
            if s < len(rms_per_sec):
                rms_per_sec[s] += r
                counts[s]      += 1
        with np.errstate(invalid="ignore"):
            rms_per_sec = np.where(counts > 0, rms_per_sec / counts, 0.0)

        # --- Baseline statistics (robust: use median + MAD) -------------------
        median_rms = float(np.median(rms_per_sec[rms_per_sec > 0])) if np.any(rms_per_sec > 0) else 0.0
        mad        = float(np.median(np.abs(rms_per_sec - median_rms)))
        threshold  = max(median_rms + 2.0 * mad, 0.03)   # adaptive + hard floor

        violations = []

        for sec_idx, energy in enumerate(rms_per_sec):
            if energy < threshold:
                continue

            # Slice exact 1-second segment
            start = sec_idx * sr_target
            end   = start + sr_target
            y_sec = y[start:end]
            if len(y_sec) < 512:
                continue

            # Spectral features
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_sec, sr=sr)))
            zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=y_sec)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_sec)))

            # ── FILTER 1: breathing / mic bump (high ZCR + high flatness) ──
            if zcr > 0.18 and flatness > 0.10:
                continue

            # ── FILTER 2: normal agent speech (voiced, mid centroid) ──
            # Only suppress if energy is not extreme (< 3σ above median)
            is_speech_band   = 600 < centroid < 2_200 and 0.03 < zcr < 0.14
            is_moderate_spike = energy < (median_rms + 3.5 * mad)
            if is_speech_band and is_moderate_spike:
                continue

            # ── VERIFIED NOISE ──
            mm = sec_idx // 60
            ss = sec_idx % 60
            timestamp  = f"{mm:02d}:{ss:02d}"
            noise_type = classify_noise_type(centroid, zcr, flatness)

            violations.append({
                "timestamp": timestamp,
                "noise":     noise_type,
                "energy":    round(float(energy), 5),
                "centroid":  round(centroid, 1),
                "zcr":       round(zcr, 4),
                "flatness":  round(flatness, 4),
            })

        return {
            "duration_sec": round(len(y) / sr_target, 1),
            "violations":   violations,
            "passed":       len(violations) == 0,
            "error":        None,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "duration_sec": 0,
            "violations":   [],
            "passed":       False,
            "error":        str(exc),
        }


# ---------------------------------------------------------------------------
# 5. UI — file upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "👇 Drop one or more call recordings here (MP3 / WAV)",
    type=["mp3", "wav"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()

st.info(f"📂 **{len(uploaded_files)}** file(s) loaded — press the button below to begin.")

col_btn, _ = st.columns([1, 3])
with col_btn:
    run = st.button("🔍 Start Bulk Audio Audit", type="primary")

if not run:
    st.stop()


# ---------------------------------------------------------------------------
# 6. Run analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 💡 Bulk Audit Results")

progress = st.progress(0, text="Analysing…")
summary_rows = []   # for the summary table

for idx, file in enumerate(uploaded_files):
    file_bytes = file.read()                    # read once; reuse for both audio + analysis
    result     = analyze_audio(file_bytes, file.name)
    status_icon = "✅" if result["passed"] else "❌"

    with st.expander(f"{status_icon}  {file.name}", expanded=not result["passed"]):

        # ── Audio player (correct MIME type) ──────────────────────────────
        ext  = file.name.rsplit(".", 1)[-1].lower()
        mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
        st.audio(io.BytesIO(file_bytes), format=mime)

        if result["error"]:
            st.error(f"❌ Analysis failed: {result['error']}")
        else:
            dur = result["duration_sec"]
            n_v = len(result["violations"])

            # ── Quick stats row ────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{dur}s</div>"
                    f"<div class='metric-label'>Duration</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                colour = "#f87171" if n_v > 0 else "#4ade80"
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='color:{colour} !important'>{n_v}</div>"
                    f"<div class='metric-label'>Noise Events</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                verdict = "FAIL" if n_v > 0 else "PASS"
                v_colour = "#f87171" if n_v > 0 else "#4ade80"
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='color:{v_colour} !important'>{verdict}</div>"
                    f"<div class='metric-label'>Verdict</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")   # spacer

            if result["violations"]:
                st.error("**Noise events detected:**")
                import pandas as pd
                vdf = pd.DataFrame(result["violations"])[
                    ["timestamp", "noise", "energy", "centroid", "zcr", "flatness"]
                ].rename(columns={
                    "timestamp": "⏱ Timestamp",
                    "noise":     "🔊 Noise Type",
                    "energy":    "Energy (RMS)",
                    "centroid":  "Centroid (Hz)",
                    "zcr":       "ZCR",
                    "flatness":  "Flatness",
                })
                st.dataframe(vdf, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Quality Audit Passed — environment complies with quiet-workspace standards.")

    # ── Accumulate summary row ─────────────────────────────────────────────
    unique_types = list({v["noise"] for v in result["violations"]})
    summary_rows.append({
        "File":          file.name,
        "Duration (s)":  result["duration_sec"],
        "Noise Events":  len(result["violations"]),
        "Noise Types":   "; ".join(unique_types) if unique_types else "—",
        "Verdict":       "FAIL" if result["violations"] else ("ERROR" if result["error"] else "PASS"),
    })

    progress.progress((idx + 1) / len(uploaded_files), text=f"Processed {idx + 1}/{len(uploaded_files)}")

# ---------------------------------------------------------------------------
# 7. Summary dashboard
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Audit Summary")

import pandas as pd
summary_df = pd.DataFrame(summary_rows)
total      = len(summary_df)
passed     = int((summary_df["Verdict"] == "PASS").sum())
failed     = int((summary_df["Verdict"] == "FAIL").sum())

m1, m2, m3, m4 = st.columns(4)
for col, label, value, colour in [
    (m1, "Total Files",    total,  "#a78bfa"),
    (m2, "Passed ✅",       passed, "#4ade80"),
    (m3, "Failed ❌",       failed, "#f87171"),
    (m4, "Pass Rate",      f"{passed/total*100:.0f}%" if total else "—", "#38bdf8"),
]:
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{colour} !important'>{value}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── CSV download ────────────────────────────────────────────────────────────
csv_bytes = summary_df.to_csv(index=False).encode()
st.download_button(
    label="⬇️  Download Full Report (CSV)",
    data=csv_bytes,
    file_name="audio_audit_report.csv",
    mime="text/csv",
)

st.balloons()
st.markdown("---")
st.caption("Smart Audit Tool — Bulk Precision Edition.")
