import io
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
    "facial_emotion_model.keras"
)


speech_model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "speech_emotion_svm.pkl"
)


speech_scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "encoders",
    "speech_scaler.pkl"
)



# ===============================
# Load Models
# ===============================




speech_model = joblib.load(
    speech_model_path
)


speech_scaler = joblib.load(
    speech_scaler_path
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

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_model = tf.keras.models.load_model(
    face_model_path,
    compile=False
)



def predict_face(image):
    # Convert PIL image to RGB array
    img = image.convert("RGB")
    img_array = np.array(img)

    # Detect faces with OpenCV
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return "neutral", 0

    # Use largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_img = img_array[y:y+h, x:x+w]

    # Prepare CLAHE for contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _prepare_variant(gray_face, scale):
        # resize with scale, then center-crop or pad to 48x48
        size = max(1, int(48 * scale))
        tmp = cv2.resize(gray_face, (size, size), interpolation=cv2.INTER_AREA)

        if size > 48:
            start = (size - 48) // 2
            tmp = tmp[start:start+48, start:start+48]
        elif size < 48:
            pad = 48 - size
            top = pad // 2
            bottom = pad - top
            left = top
            right = bottom
            tmp = cv2.copyMakeBorder(tmp, top, bottom, left, right, cv2.BORDER_REFLECT)

        tmp = clahe.apply(tmp)
        tmp = tmp.astype("float32") / 255.0
        tmp = np.expand_dims(tmp, axis=-1)  # (48,48,1)
        return tmp

    # Convert to grayscale face crop
    try:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    except Exception:
        return "neutral", 0

    # Test-time augmentations: scales and horizontal flip
    scales = [0.9, 1.0, 1.1]
    prob_list = []

    for s in scales:
        variant = _prepare_variant(gray_face, s)
        # ensure variant has shape (48,48,1)
        if variant.ndim == 2:
            variant = np.expand_dims(variant, axis=-1)

        batch = np.expand_dims(variant, axis=0)
        preds = face_model.predict(batch)
        prob_list.append(preds[0])

        # horizontal flip using numpy to preserve channels
        flip = np.flip(variant, axis=1)
        preds_f = face_model.predict(np.expand_dims(flip, axis=0))
        prob_list.append(preds_f[0])

    # Average probabilities across variants
    avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)

    emotion_index = int(np.argmax(avg_probs))
    confidence = float(np.max(avg_probs) * 100)

    return (
        face_classes[emotion_index],
        confidence
    )



# ===============================
# Speech Prediction
# ===============================

def extract_features(audio_input):

    if isinstance(audio_input, (bytes, bytearray)):
        audio_input = io.BytesIO(audio_input)

    if hasattr(audio_input, "read"):
        audio_input.seek(0)
        audio_input = io.BytesIO(audio_input.read())

    signal, sr = librosa.load(
        audio_input,
        sr=22050
    )

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    rms = np.mean(
        librosa.feature.rms(y=signal)
    )

    feature_vector = np.hstack([
        mfcc,
        chroma,
        mel,
        contrast,
        zcr,
        rms
    ])

    return feature_vector


# ===============================
# Speech Prediction

def predict_speech(audio):

    features = extract_features(
        audio
    )

    features = features.reshape(1, -1)

    features = speech_scaler.transform(features)

    prediction = speech_model.predict(
        features
    )


    return speech_classes[
        prediction[0]
    ]