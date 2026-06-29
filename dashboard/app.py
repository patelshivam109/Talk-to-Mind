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


camera = st.camera_input(
    "Take a picture",
    key="face_capture"
)


if camera:

    img = Image.open(camera)


    try:

        face_emotion, face_conf = predict_face(img)


        st.success(
            f"Detected Face Emotion: {face_emotion}"
        )


        st.metric(
            "Confidence",
            f"{face_conf:.2f}%"
        )


        # optional display captured image
        st.image(
            img,
            caption="Captured Face",
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


audio = st.file_uploader(
    "Upload voice sample",
    type=["wav"]
)



if audio:


    try:

        speech_emotion = predict_speech(
            audio
        )


        st.success(
            f"Detected Speech Emotion: {speech_emotion}"
        )


    except Exception:

        speech_emotion = "neutral"

        st.warning(
            "Speech model error"
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