import numpy as np


# Emotion risk mapping
EMOTION_SCORES = {
    "happy": 0,
    "neutral": 1,
    "surprise": 1,
    "surprised": 1,
    "sad": 3,
    "disgust": 3,
    "fear": 4,
    "angry": 4
}


def emotion_to_score(emotion):
    """
    Convert detected emotion into risk score
    """

    emotion = emotion.lower()

    return EMOTION_SCORES.get(emotion, 2)



def questionnaire_score(score):
    """
    Convert questionnaire score (0-80)
    into risk value (0-10)
    """

    score = float(score)

    return (score / 80) * 10



def calculate_wellbeing(
        face_emotion,
        speech_emotion,
        questionnaire_total
):

    face_score = emotion_to_score(face_emotion)
    speech_score = emotion_to_score(speech_emotion)

    q_score = questionnaire_score(questionnaire_total)


    # Weighted fusion
    final_risk = (
        (face_score / 4) * 30 +
        (speech_score / 4) * 30 +
        (q_score / 10) * 40
    )


    wellbeing_score = round(100 - final_risk)


    if wellbeing_score >= 75:
        level = "Low Concern"

    elif wellbeing_score >= 50:
        level = "Moderate Concern"

    else:
        level = "High Concern"


    return {
        "face_emotion": face_emotion,
        "speech_emotion": speech_emotion,
        "questionnaire_score": questionnaire_total,
        "wellbeing_score": wellbeing_score,
        "risk_level": level
    }



def generate_recommendation(level):

    recommendations = {

        "Low Concern": [
            "Maintain your current routine",
            "Continue healthy sleep habits",
            "Keep engaging in positive activities"
        ],


        "Moderate Concern": [
            "Try relaxation techniques",
            "Maintain a regular sleep schedule",
            "Take breaks during stressful tasks"
        ],


        "High Concern": [
            "Consider talking with someone you trust",
            "Practice stress management activities",
            "Focus on rest and self-care"
        ]

    }


    return recommendations.get(
        level,
        []
    )