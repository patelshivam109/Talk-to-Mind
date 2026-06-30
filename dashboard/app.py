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
    layout="wide"
)



st.title("🧠 Talk To Mind")

st.subheader(
    "AI Based Multimodal Emotion Analysis"
)


st.divider()



# =========================
# Facial Emotion
# =========================
st.header("1. Facial Emotion Detection")


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


        st.success(
            f"Detected Face Emotion: {face_emotion}"
        )


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


# =========================
# Speech Emotion
# =========================

st.header(
    "2. Speech Emotion Detection"
)


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

        st.success(
            f"Detected Speech Emotion: {speech_emotion}"
        )


    except Exception as e:

        speech_emotion = "neutral"

        st.warning(
            f"Speech model error: {e}"
        )


else:

    speech_emotion = "neutral"




# =========================
# Questionnaire
# =========================

st.header(
    "3. Self Assessment"
)



questions = pd.read_csv(
    "../datasets/questionnaire/mental_health_questions.csv"
)



score = 0


options = {
    "Never":0,
    "Rarely":1,
    "Sometimes":2,
    "Often":3,
    "Always":4
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




# =========================
# Final Analysis
# =========================


st.divider()


if st.button(
    "Generate Result"
):


    result = calculate_wellbeing(

        face_emotion,

        speech_emotion,

        score

    )



    st.success(
        "Analysis Complete"
    )



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



    st.subheader(
        "Emotion Summary"
    )


    st.write(
        f"🙂 Facial Emotion: **{face_emotion}**"
    )

    st.write(
        f"🎤 Speech Emotion: **{speech_emotion}**"
    )



    st.subheader(
        "Recommendations"
    )



    recommendations = generate_recommendation(
        result["risk_level"]
    )



    for item in recommendations:

        st.write(
            "•",
            item
        )