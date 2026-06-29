import os
import numpy as np
import tensorflow as tf
import joblib
import librosa
import cv2


# ===============================
# Project root path
# ===============================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ===============================
# Model Paths
# ===============================

face_model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "facial_emotion_mobilenet.keras"
)


speech_model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "speech_emotion_svm.pkl"
)



# ===============================
# Load Models
# ===============================

face_model = tf.keras.models.load_model(
    face_model_path
)


speech_model = joblib.load(
    speech_model_path
)



# ===============================
# Classes
# ===============================

face_classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


speech_classes = [
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised"
]



# ===============================
# Face Detector
# ===============================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)



# ===============================
# Facial Prediction
# ===============================

def predict_face(image):

    img = np.array(image.convert("RGB"))

    img = cv2.resize(
        img,
        (96,96)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )


    pred = face_model.predict(
        img,
        verbose=0
    )


    idx = np.argmax(pred[0])

    confidence = float(
        pred[0][idx] * 100
    )


    return (
        face_classes[idx],
        confidence
    )




# ===============================
# Speech Prediction
# ===============================

def predict_speech(audio):

    features = extract_features(
        audio
    )


    prediction = speech_model.predict(
        features
    )


    return speech_classes[
        prediction[0]
    ]