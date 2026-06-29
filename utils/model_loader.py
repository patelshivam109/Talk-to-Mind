import os
import numpy as np
import tensorflow as tf
import joblib
import librosa


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
    "facial_emotion_model.keras"
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
# Facial Prediction
# ===============================

def predict_face(image):

    image = image.convert("L")

    image = image.resize(
        (48,48)
    )

    image = np.array(image)

    image = image / 255.0


    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.expand_dims(
        image,
        axis=-1
    )


    prediction = face_model.predict(
        image,
        verbose=0
    )


    index = np.argmax(
        prediction
    )


    return face_classes[index]



# ===============================
# Speech Feature Extraction
# ===============================

def extract_features(audio):

    y, sr = librosa.load(
        audio,
        duration=3
    )


    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )


    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )


    mfcc_std = np.std(
        mfcc,
        axis=1
    )


    delta = librosa.feature.delta(
        mfcc
    )


    delta_mean = np.mean(
        delta,
        axis=1
    )


    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )


    delta2_mean = np.mean(
        delta2,
        axis=1
    )


    features = np.hstack(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta2_mean
        ]
    )


    # SVM was trained on 189 features
    if len(features) < 189:

        features = np.pad(
            features,
            (0,189-len(features))
        )


    return features.reshape(
        1,-1
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