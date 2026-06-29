import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image


sys.path.append(os.path.abspath("../"))


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
st.subheader("AI Based Multimodal Emotion Analysis")


st.divider()


st.header("1. Facial Emotion")


camera = st.camera_input(
    "Take a picture"
)


if camera:

    img = Image.open(camera)

    face_emotion = predict_face(img)

    st.write(
        "Detected Face Emotion:",
        face_emotion
    )

else:

    face_emotion = "neutral"


st.header("2. Speech Emotion")

audio = st.file_uploader(
    "Upload voice sample",
    type=["wav"]
)


if audio:

    speech_emotion = predict_speech(audio)

    st.write(
        "Detected Speech Emotion:",
        speech_emotion
    )

else:

    speech_emotion = "neutral"

st.header("3. Self Assessment")


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
        options.keys(),
        key=row["QuestionID"]
    )

    value = options[answer]

    if row["ReverseScore"] == "Yes":
        value = 4 - value

    score += value



if st.button("Generate Result"):


    result = calculate_wellbeing(
        face_emotion,
        speech_emotion,
        score
    )


    st.success("Analysis Complete")


    st.metric(
        "Well Being Score",
        result["wellbeing_score"]
    )


    st.write(
        "Risk Level:",
        result["risk_level"]
    )


    st.write(
        "Questionnaire Score:",
        result["questionnaire_score"]
    )


    st.subheader("Recommendations")


    for item in generate_recommendation(
        result["risk_level"]
    ):

        st.write("•", item)