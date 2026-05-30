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

    /* ── Timestamp pill ── */
    .ts-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid #6366f1;
        color: #c4b5fd !important;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.85rem;
        font-family: 'Courier New', monospace;
        margin: 3px 4px;
        font-weight: 600;
    }

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


@st.cache_data(show_spinner=False)
def analyze_audio(file_bytes: bytes, filename: str) -> dict:
    """
    Analyse a raw audio byte-string and return a result dictionary.

    Detection targets:
      - Physical impacts: banging, door slams, loud thumps
      - Background chatter: other people speaking in the environment

    Suppressed (never flagged):
      - Agent speech at any volume (normal or raised)
      - Headset static / line hiss
      - Breathing / mic proximity pops
      - Isolated mouth transients (clicks, plosives)

    Core technique: Harmonic-Percussive Source Separation (HPSS).
      - Percussive component  →  captures physical impacts / transients
      - Harmonic component    →  captures tonal signals (speech, music)
    A "clean solo voice" has very low percussive energy and clean harmonics.
    Background chatter has higher spectral complexity (multiple overlapping
    voices = higher flatness) that distinguishes it from a single dominant
    speaker even when both occupy the same frequency band.

    Returns
    -------
    dict with keys:
        duration_sec – total audio length in seconds
        violations   – list of {"timestamp": str, "noise": str}
        passed       – bool
        error        – str | None
    """
    try:
        audio_io  = io.BytesIO(file_bytes)
        sr_target = 16_000
        y, sr     = librosa.load(audio_io, sr=sr_target, mono=True)

        # ── 1. Per-second RMS ────────────────────────────────────────────────
        frame_length = 512   # ~32 ms at 16 kHz
        hop_length   = 256   # 50 % overlap
        rms_frames   = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(rms_frames)), sr=sr, hop_length=hop_length
        )
        n_seconds   = int(np.ceil(times[-1])) + 1 if len(times) > 0 else 1
        rms_per_sec = np.zeros(n_seconds)
        counts      = np.zeros(n_seconds)
        for t, r in zip(times, rms_frames):
            s = int(t)
            if s < n_seconds:
                rms_per_sec[s] += r
                counts[s]      += 1
        with np.errstate(invalid="ignore"):
            rms_per_sec = np.where(counts > 0, rms_per_sec / counts, 0.0)

        # ── 2. Adaptive baseline ─────────────────────────────────────────────
        active     = rms_per_sec[rms_per_sec > 0]
        median_rms = float(np.median(active)) if len(active) else 0.0
        mad        = float(np.median(np.abs(active - median_rms))) if len(active) else 0.0

        # Two tiers:
        #   impact_threshold  – high bar required for physical-impact detection
        #   chatter_threshold – lower bar to also catch sustained background voices
        impact_threshold  = max(median_rms + 3.0 * mad, 0.040)
        chatter_threshold = max(median_rms + 1.5 * mad, 0.020)

        # ── 3. Candidate seconds ─────────────────────────────────────────────
        violations = []

        for sec_idx, energy in enumerate(rms_per_sec):
            # Skip truly silent / baseline-level seconds
            if energy < chatter_threshold:
                continue

            # Extract 1-second audio slice
            start = sec_idx * sr_target
            end   = start + sr_target
            y_sec = y[start:end]
            if len(y_sec) < 512:
                continue

            # ── Spectral features ────────────────────────────────────────────
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_sec, sr=sr)))
            zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=y_sec)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_sec)))

            # ── HPSS: separate transient from tonal energy ───────────────────
            # Percussive component = impacts / transients
            # Harmonic  component = speech / music (tonal)
            y_harm, y_perc = librosa.effects.hpss(y_sec, margin=3.0)
            perc_rms   = float(np.sqrt(np.mean(y_perc ** 2)))
            harm_rms   = float(np.sqrt(np.mean(y_harm ** 2)))
            perc_ratio = perc_rms / (perc_rms + harm_rms + 1e-9)

            # ── FILTER A: headset static / line hiss ─────────────────────────
            # Broadband noise has uniformly high spectral flatness.
            if flatness > 0.18:
                continue

            # ── FILTER B: breathing / mic proximity pops ─────────────────────
            # Unvoiced bursts: very high ZCR, low centroid.
            if zcr > 0.18 and centroid < 2_000:
                continue

            # ── PATH 1: Physical impact (detected before the speech filter) ──
            # Bangs and impacts are percussive-dominant.
            # We check this FIRST so it can never be masked by Filter C.
            if perc_ratio > 0.38 and energy >= impact_threshold:
                mm = sec_idx // 60
                ss = sec_idx % 60
                violations.append({
                    "timestamp": f"{mm:02d}:{ss:02d}",
                    "noise":     "💥 Physical impact (bang / thud / door slam)",
                })
                continue

            # ── FILTER C: clean solo agent voice (any volume) ────────────────
            # The agent speaking — even loudly — produces all three:
            #   • structured harmonics  → low flatness  (< 0.10)
            #   • almost no transients  → low perc_ratio (< 0.20)
            #   • speech-band centroid and ZCR
            # Background chatter fails because overlapping voices raise flatness
            # above 0.10 and create spectral complexity a solo speaker does not.
            is_speech_band = 400 < centroid < 3_500 and 0.02 < zcr < 0.22
            is_clean_solo  = flatness < 0.10 and perc_ratio < 0.20
            if is_speech_band and is_clean_solo:
                continue

            # ── FILTER D: isolated single-frame transient (mouth click) ──────
            # Background noise persists; an isolated spike is usually a plosive.
            sustain_thr = max(median_rms + 1.2 * mad, 0.018)
            left_ok     = sec_idx > 0              and rms_per_sec[sec_idx - 1] > sustain_thr
            right_ok    = sec_idx < n_seconds - 1  and rms_per_sec[sec_idx + 1] > sustain_thr
            if not (left_ok or right_ok):
                # Low-frequency physical bangs are often isolated — allow them.
                if centroid >= 500:
                    continue

            # ── VERIFIED NOISE — classify and record ─────────────────────────
            if centroid < 500:
                noise_type = "💥 Low-frequency impact (bang / thud)"
            elif centroid < 1_800 and zcr < 0.13:
                noise_type = "🗣️ Background chatter / overlapping voices"
            else:
                noise_type = "📢 Loud ambient disturbance (alarm / crash)"

            mm = sec_idx // 60
            ss = sec_idx % 60
            violations.append({"timestamp": f"{mm:02d}:{ss:02d}", "noise": noise_type})

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

        # ── Audio player ───────────────────────────────────────────────────
        ext  = file.name.rsplit(".", 1)[-1].lower()
        mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
        st.audio(io.BytesIO(file_bytes), format=mime)

        if result["error"]:
            st.error(f"❌ Analysis failed: {result['error']}")
        elif result["violations"]:
            # Group by noise type so we can label each timestamp cluster
            from collections import defaultdict
            by_type: dict = defaultdict(list)
            for v in result["violations"]:
                by_type[v["noise"]].append(v["timestamp"])

            lines = []
            for noise_type, timestamps in by_type.items():
                ts_pills = " ".join(
                    f"<span class='ts-pill'>{ts}</span>" for ts in timestamps
                )
                lines.append(f"**{noise_type}**<br>{ts_pills}")

            st.error(
                "**Noise detected at:**\n\n" + "\n\n".join(lines),
                icon="🔊",
            )
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
total  = len(summary_df)
passed = int((summary_df["Verdict"] == "PASS").sum())
failed = int((summary_df["Verdict"] == "FAIL").sum())

st.markdown(
    f"**{total}** file(s) audited — "
    f"<span style='color:#4ade80'>**{passed} passed**</span> · "
    f"<span style='color:#f87171'>**{failed} failed**</span>",
    unsafe_allow_html=True,
)
st.dataframe(
    summary_df[["File", "Noise Events", "Verdict"]],
    use_container_width=True,
    hide_index=True,
)

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
