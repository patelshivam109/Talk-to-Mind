import io
import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image


sys.path.append(
    os.path.abspath("../")
)


from utils.fusion import (
    calculate_wellbeing,
    generate_recommendation
)

from utils.model_loader import (
    predict_face,
    predict_speech
)


st.set_page_config(
    page_title="Talk To Mind",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
#  GODMODE STYLE SYSTEM — "Neural Diagnostics Console"
# =========================================================

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>

:root{
  --void:        #06080F;
  --panel:       #0E1422;
  --panel-edge:  #1B2438;
  --signal:      #7C9EFF;
  --signal-dim:  #44558A;
  --alert:       #FF6B9D;
  --calm:        #3DDC97;
  --text:        #E7EBF6;
  --muted:       #8B93A8;
  --faint:       #4B5470;
}

@keyframes pulseLine{
  0%   { stroke-dashoffset: 1000; }
  100% { stroke-dashoffset: 0; }
}
@keyframes drift{
  0%,100% { transform: translateY(0px); }
  50%     { transform: translateY(-4px); }
}
@keyframes glowPulse{
  0%,100% { opacity:.55; filter:drop-shadow(0 0 2px var(--signal)); }
  50%     { opacity:1;   filter:drop-shadow(0 0 10px var(--signal)); }
}
@keyframes sweep{
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes fadeUp{
  from { opacity:0; transform: translateY(14px); }
  to   { opacity:1; transform: translateY(0); }
}
@keyframes spinSlow{
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes blink{
  0%,100% { opacity:1; }
  50%     { opacity:.25; }
}

html, body, [class*="css"]{
  background-color: var(--void) !important;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

.stApp{
  background:
    radial-gradient(ellipse 60% 50% at 50% -10%, rgba(124,158,255,0.10), transparent 60%),
    radial-gradient(ellipse 50% 40% at 100% 100%, rgba(255,107,157,0.06), transparent 60%),
    var(--void);
}

/* kill default streamlit chrome noise */
#MainMenu, footer, header{ visibility:hidden; }
.block-container{ padding-top: 1.2rem; max-width: 1180px; }

/* ---------- masthead ---------- */
.tm-mast{
  display:flex; align-items:center; justify-content:space-between;
  padding: 18px 26px;
  border:1px solid var(--panel-edge);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(124,158,255,0.06), rgba(255,255,255,0.01));
  margin-bottom: 26px;
  animation: fadeUp .6s ease both;
  position:relative;
  overflow:hidden;
}
.tm-mast::before{
  content:"";
  position:absolute; inset:0;
  background: linear-gradient(100deg, transparent 40%, rgba(124,158,255,0.10) 50%, transparent 60%);
  background-size: 200% 100%;
  animation: sweep 6s linear infinite;
  pointer-events:none;
}
.tm-mast-id{
  font-family:'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 2.5px;
  color: var(--signal);
  text-transform: uppercase;
}
.tm-mast-title{
  font-family:'Space Grotesk', sans-serif;
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.5px;
  margin: 2px 0 0 0;
  background: linear-gradient(90deg, #fff, var(--signal) 60%, var(--calm));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.tm-mast-sub{
  font-family:'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}
.tm-live{
  display:flex; align-items:center; gap:8px;
  font-family:'JetBrains Mono', monospace;
  font-size: 11px; color: var(--calm);
  letter-spacing: 1.5px; text-transform:uppercase;
}
.tm-live-dot{
  width:8px; height:8px; border-radius:50%;
  background: var(--calm);
  box-shadow: 0 0 8px var(--calm);
  animation: blink 1.6s ease-in-out infinite;
}

/* ---------- neural pulse waveform strip ---------- */
.tm-pulse-wrap{
  border:1px solid var(--panel-edge);
  border-radius: 12px;
  background: var(--panel);
  margin-bottom: 28px;
  padding: 0;
  overflow:hidden;
  animation: fadeUp .6s ease .05s both;
}
.tm-pulse-svg path{
  animation: glowPulse 2.4s ease-in-out infinite;
}

/* ---------- section module (instrument panel) ---------- */
.tm-module{
  position: relative;
  border: 1px solid var(--panel-edge);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.002));
  padding: 26px 28px 22px 28px;
  margin-bottom: 22px;
  animation: fadeUp .55s ease both;
}
.tm-module::before, .tm-module::after,
.tm-corner-br::before, .tm-corner-br::after{
  content:""; position:absolute; width:16px; height:16px;
  border-color: var(--signal-dim); opacity:.8;
}
.tm-module::before{ top:-1px; left:-1px; border-top:2px solid var(--signal); border-left:2px solid var(--signal); border-radius: 6px 0 0 0; }
.tm-module::after{ top:-1px; right:-1px; border-top:2px solid var(--signal); border-right:2px solid var(--signal); border-radius: 0 6px 0 0; }

.tm-module-head{
  display:flex; align-items:baseline; gap:12px;
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--panel-edge);
}
.tm-module-tag{
  font-family:'JetBrains Mono', monospace;
  font-size: 11px; color: var(--signal);
  border: 1px solid var(--signal-dim);
  border-radius: 5px;
  padding: 3px 8px;
  letter-spacing: 1.5px;
}
.tm-module-title{
  font-family:'Space Grotesk', sans-serif;
  font-size: 20px; font-weight: 600;
  color: var(--text);
  letter-spacing: -0.2px;
}
.tm-module-desc{
  font-family:'JetBrains Mono', monospace;
  font-size: 11.5px; color: var(--muted);
  margin-left: auto;
  text-align:right;
}

/* ---------- readout result chip ---------- */
.tm-readout{
  display:inline-flex; align-items:center; gap:10px;
  border: 1px solid var(--calm);
  background: rgba(61,220,151,0.08);
  border-radius: 10px;
  padding: 10px 16px;
  font-family:'JetBrains Mono', monospace;
  font-size: 13px; color: var(--calm);
  margin: 6px 0 14px 0;
  animation: fadeUp .4s ease both;
}
.tm-readout-warn{
  border-color: var(--alert);
  background: rgba(255,107,157,0.08);
  color: var(--alert);
}
.tm-readout-dot{
  width:7px; height:7px; border-radius:50%; background: currentColor;
  box-shadow: 0 0 6px currentColor;
  animation: blink 1.4s ease-in-out infinite;
}

/* ---------- streamlit widget reskins ---------- */
[data-testid="stFileUploaderDropzone"]{
  background: rgba(124,158,255,0.04) !important;
  border: 1.5px dashed var(--signal-dim) !important;
  border-radius: 12px !important;
  transition: border-color .25s ease, background .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color: var(--signal) !important;
  background: rgba(124,158,255,0.08) !important;
}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{
  border-radius: 12px !important;
  border: 1px solid var(--panel-edge);
}

.stTabs [data-baseweb="tab-list"]{
  gap: 6px;
  background: var(--panel);
  padding: 5px;
  border-radius: 10px;
  border: 1px solid var(--panel-edge);
}
.stTabs [data-baseweb="tab"]{
  font-family:'JetBrains Mono', monospace;
  font-size: 12.5px;
  color: var(--muted);
  border-radius: 7px;
  padding: 8px 16px;
}
.stTabs [aria-selected="true"]{
  background: var(--signal) !important;
  color: #06080F !important;
  font-weight: 600;
}

.stSelectbox label, .stSelectbox p{
  font-family:'Inter', sans-serif !important;
  color: var(--text) !important;
  font-size: 14.5px !important;
  font-weight: 500 !important;
}
.stSelectbox > div > div{
  background: var(--panel) !important;
  border: 1px solid var(--panel-edge) !important;
  border-radius: 9px !important;
  color: var(--text) !important;
}
.stSelectbox [data-baseweb="select"]:hover > div{
  border-color: var(--signal-dim) !important;
}

[data-testid="stMetric"]{
  background: var(--panel);
  border: 1px solid var(--panel-edge);
  border-radius: 12px;
  padding: 16px 18px 14px 18px;
  transition: transform .2s ease, border-color .2s ease;
}
[data-testid="stMetric"]:hover{
  transform: translateY(-3px);
  border-color: var(--signal-dim);
}
[data-testid="stMetricLabel"]{
  font-family:'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 1px;
  color: var(--muted) !important;
  text-transform: uppercase;
}
[data-testid="stMetricValue"]{
  font-family:'Space Grotesk', sans-serif !important;
  color: var(--signal) !important;
  font-size: 30px !important;
}

div.stButton > button{
  font-family:'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 15px;
  letter-spacing: 0.3px;
  color: #06080F;
  background: linear-gradient(90deg, var(--signal), var(--calm));
  border: none;
  border-radius: 10px;
  padding: 13px 28px;
  width: 100%;
  box-shadow: 0 0 0px rgba(124,158,255,0);
  transition: box-shadow .25s ease, transform .15s ease;
}
div.stButton > button:hover{
  box-shadow: 0 0 26px rgba(124,158,255,0.45);
  transform: translateY(-1px);
}
div.stButton > button:active{ transform: translateY(0px) scale(.99); }

.stAlert{
  border-radius: 11px !important;
  font-family: 'Inter', sans-serif !important;
  border: 1px solid var(--panel-edge) !important;
}

hr, [data-testid="stDivider"]{ border-color: var(--panel-edge) !important; opacity:.7; }

/* recommendation list items */
.tm-rec{
  display:flex; align-items:flex-start; gap:12px;
  padding: 11px 14px;
  border-radius: 10px;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--panel-edge);
  margin-bottom: 8px;
  font-size: 14.5px;
  animation: fadeUp .4s ease both;
  transition: border-color .2s ease, background .2s ease;
}
.tm-rec:hover{ border-color: var(--signal-dim); background: rgba(124,158,255,0.05); }
.tm-rec-mark{
  font-family:'JetBrains Mono', monospace;
  color: var(--calm); flex-shrink:0; margin-top:1px;
}

.tm-footer{
  text-align:center; padding: 28px 0 10px 0;
  font-family:'JetBrains Mono', monospace;
  font-size: 11px; color: var(--faint);
  letter-spacing: 1.5px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
#  MASTHEAD
# =========================================================

st.markdown("""
<div class="tm-mast">
  <div>
    <div class="tm-mast-id">SYSTEM · MULTIMODAL AFFECT ENGINE · v2.4</div>
    <div class="tm-mast-title">◈ Talk To Mind</div>
    <div class="tm-mast-sub">face signal + voice signal + self-report → fused wellbeing readout</div>
  </div>
  <div class="tm-live"><span class="tm-live-dot"></span> session active</div>
</div>
""", unsafe_allow_html=True)


# animated neural pulse waveform — signature element
st.markdown("""
<div class="tm-pulse-wrap">
  <svg class="tm-pulse-svg" viewBox="0 0 1200 70" width="100%" height="70" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="pg" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#7C9EFF" stop-opacity="0"/>
        <stop offset="15%" stop-color="#7C9EFF"/>
        <stop offset="50%" stop-color="#3DDC97"/>
        <stop offset="85%" stop-color="#7C9EFF"/>
        <stop offset="100%" stop-color="#7C9EFF" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <path d="M0,35 L80,35 L100,35 L115,10 L130,60 L145,35 L165,35 L300,35 L320,35 L335,15 L350,55 L365,35 L385,35 L520,35 L540,35 L555,8 L570,62 L585,35 L605,35 L740,35 L760,35 L775,12 L790,58 L805,35 L825,35 L960,35 L980,35 L995,10 L1010,60 L1025,35 L1045,35 L1200,35"
      fill="none" stroke="url(#pg)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</div>
""", unsafe_allow_html=True)


# =========================
# Facial Emotion
# =========================

st.markdown("""
<div class="tm-module">
  <div class="tm-module-head">
    <span class="tm-module-tag">01</span>
    <span class="tm-module-title">Facial Emotion Detection</span>
    <span class="tm-module-desc">VISUAL CHANNEL<br>camera or static frame</span>
  </div>
""", unsafe_allow_html=True)


face_tab_camera, face_tab_upload = st.tabs(
    ["📷 Live Camera", "🖼️ Upload Image"]
)

face_image = None

with face_tab_camera:
    camera = st.camera_input(
        "Take a picture",
        key="face_capture"
    )
    if camera:
        face_image = Image.open(camera)

with face_tab_upload:
    uploaded_img = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="face_upload"
    )
    if uploaded_img:
        face_image = Image.open(uploaded_img)


if face_image:

    try:

        face_emotion, face_conf = predict_face(face_image)

        st.markdown(f"""
        <div class="tm-readout">
          <span class="tm-readout-dot"></span>
          FACE EMOTION DETECTED · <b>{face_emotion.upper()}</b>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Confidence",
            f"{face_conf:.2f}%"
        )

        st.image(
            face_image,
            caption="Analysed Face",
            width=300
        )

    except Exception as e:

        st.error(
            f"Face detection failed: {e}"
        )

        face_emotion = "neutral"
        face_conf = 0

else:

    face_emotion = "neutral"
    face_conf = 0

st.markdown("</div>", unsafe_allow_html=True)  # close module 01


# =========================
# Speech Emotion
# =========================

st.markdown("""
<div class="tm-module">
  <div class="tm-module-head">
    <span class="tm-module-tag">02</span>
    <span class="tm-module-title">Speech Emotion Detection</span>
    <span class="tm-module-desc">AUDIO CHANNEL<br>any common format, auto-converted</span>
  </div>
""", unsafe_allow_html=True)


SUPPORTED_AUDIO_TYPES = [
    "wav", "mp3", "ogg", "flac",
    "m4a", "aac", "wma", "aiff",
    "opus", "webm"
]

audio = st.file_uploader(
    "Upload voice sample (wav, mp3, ogg, flac, m4a, aac, …)",
    type=SUPPORTED_AUDIO_TYPES
)


def _to_wav_bytes(uploaded_file):
    """Convert any uploaded audio format to WAV bytes using ffmpeg directly."""
    import tempfile
    import subprocess
    import imageio_ffmpeg
    import shutil

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Get extension
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else "tmp"

    # Create temp files
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f_in:
        in_path = f_in.name
        f_in.write(uploaded_file.read())

    out_path = in_path + ".wav"

    try:
        # Run ffmpeg to convert to WAV
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", in_path, out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Read WAV bytes
        with open(out_path, "rb") as f_out:
            wav_data = f_out.read()

        return io.BytesIO(wav_data)

    finally:
        # Cleanup temp files
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


if audio:

    try:
        wav_bytes = _to_wav_bytes(audio)

        speech_emotion = predict_speech(
            wav_bytes
        )

        st.markdown(f"""
        <div class="tm-readout">
          <span class="tm-readout-dot"></span>
          SPEECH EMOTION DETECTED · <b>{speech_emotion.upper()}</b>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        speech_emotion = "neutral"

        st.warning(
            f"Speech model error: {e}"
        )

else:

    speech_emotion = "neutral"

st.markdown("</div>", unsafe_allow_html=True)  # close module 02


# =========================
# Questionnaire
# =========================

st.markdown("""
<div class="tm-module">
  <div class="tm-module-head">
    <span class="tm-module-tag">03</span>
    <span class="tm-module-title">Self Assessment</span>
    <span class="tm-module-desc">SUBJECTIVE CHANNEL<br>weighted questionnaire</span>
  </div>
""", unsafe_allow_html=True)


questions = pd.read_csv(
    "../datasets/questionnaire/mental_health_questions.csv"
)


score = 0


options = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}


for _, row in questions.iterrows():

    answer = st.selectbox(
        row["Question"],
        list(options.keys()),
        key=int(row["QuestionID"])
    )

    value = options[answer]

    if str(row["ReverseScore"]) == "Yes":
        value = 4 - value

    score += value

st.markdown("</div>", unsafe_allow_html=True)  # close module 03


# =========================
# Final Analysis
# =========================

st.markdown('<div style="margin: 30px 0 18px 0;">', unsafe_allow_html=True)

if st.button(
    "⚡ Generate Result"
):

    result = calculate_wellbeing(
        face_emotion,
        speech_emotion,
        score
    )

    risk = str(result["risk_level"]).lower()
    is_high_risk = any(w in risk for w in ["high", "severe", "elevated"])

    # =========================
    # Tracking Over Time
    # =========================
    import datetime
    
    history_file = "../datasets/history.csv"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    new_record = pd.DataFrame([{
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Wellbeing_Score": result["wellbeing_score"],
        "Risk_Level": result["risk_level"]
    }])
    
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, new_record], ignore_index=True)
    else:
        history_df = new_record
        
    history_df.to_csv(history_file, index=False)


    st.markdown(f"""
    <div class="tm-readout {'tm-readout-warn' if is_high_risk else ''}" style="margin-top:18px;">
      <span class="tm-readout-dot"></span>
      ANALYSIS COMPLETE · FUSED READOUT GENERATED
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tm-module" style="animation-delay:.05s;">
      <div class="tm-module-head">
        <span class="tm-module-tag">◈</span>
        <span class="tm-module-title">Fused Readout</span>
        <span class="tm-module-desc">FACE + VOICE + SELF-REPORT</span>
      </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Wellbeing Score",
            result["wellbeing_score"]
        )

    with col2:
        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    with col3:
        st.metric(
            "Questionnaire Score",
            score
        )

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-module-desc" style="text-align:left; margin-bottom:10px;">EMOTION SUMMARY</div>', unsafe_allow_html=True)

    st.write(f"🙂 Facial Emotion: **{face_emotion}**")
    st.write(f"🎤 Speech Emotion: **{speech_emotion}**")

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-module-desc" style="text-align:left; margin-bottom:10px;">RECOMMENDATIONS</div>', unsafe_allow_html=True)

    recommendations = generate_recommendation(
        result["risk_level"]
    )

    for item in recommendations:
        st.markdown(f"""
        <div class="tm-rec"><span class="tm-rec-mark">▸</span><span>{item}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close fused readout module

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Mental Wellness Progress
# =========================
st.markdown("""
<div class="tm-module">
  <div class="tm-module-head">
    <span class="tm-module-tag">04</span>
    <span class="tm-module-title">Mental Wellness Progress</span>
    <span class="tm-module-desc">HISTORICAL TRACKING</span>
  </div>
""", unsafe_allow_html=True)

history_file = "../datasets/history.csv"
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    if len(history_df) > 0:
        st.line_chart(history_df.set_index("Date")["Wellbeing_Score"])
    else:
        st.info("Take the assessment to start tracking your progress!")
else:
    st.info("Take the assessment to start tracking your progress over time!")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="tm-footer">TALK TO MIND · MULTIMODAL EMOTION FUSION SYSTEM</div>', unsafe_allow_html=True)