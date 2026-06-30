# 🧠 Talk To Mind
### AI-Based Multimodal Mental Health Assessment System

Talk To Mind is an AI-powered multimodal mental health assessment system that combines **Facial Emotion Recognition**, **Speech Emotion Recognition**, and a **Self-Assessment Questionnaire** to estimate an individual's emotional wellbeing and generate personalized recommendations.

The project was developed as an AI/ML internship project using Deep Learning, Machine Learning, Computer Vision, Audio Signal Processing, and Streamlit.

---

# Features

- 😊 Facial Emotion Recognition using CNN
- 🎤 Speech Emotion Recognition using Support Vector Machine (SVM)
- 📋 Mental Health Self-Assessment Questionnaire
- 🧠 Multimodal Fusion Engine
- 📊 Wellbeing Score Calculation
- ⚠️ Risk Level Prediction
- 💡 Personalized Recommendations
- 🌐 Interactive Streamlit Dashboard

---

# Project Workflow

```
                Camera Image
                      │
                      ▼
        Facial Emotion Recognition
                      │
                      ▼

Microphone Audio ─► Speech Emotion Recognition
                      │
                      ▼

        Mental Health Questionnaire
                      │
                      ▼

           Multimodal Fusion Engine
                      │
                      ▼

      Wellbeing Score & Risk Analysis
                      │
                      ▼

      Personalized Recommendations
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- Scikit-Learn
- OpenCV
- Librosa
- NumPy
- Pandas
- Matplotlib
- Joblib
- Streamlit

---

# Models Used

## Facial Emotion Recognition

- Custom CNN
- MobileNetV2 (Transfer Learning)
- Final Dashboard Model:
  - MobileNetV2

Dataset:
- FER2013

---

## Speech Emotion Recognition

Models Evaluated

- Logistic Regression
- Random Forest
- Support Vector Machine (Best Model)

Final Dashboard Model

- Support Vector Machine (SVM)

Dataset

- RAVDESS

---

# Project Structure

```
Talk-To-Mind/

│
├── app/
│   └── app.py
│
├── datasets/
│   ├── facial/
│   ├── speech/
│   └── questionnaire/
│
├── models/
│   ├── encoders/
│   │     └── speech_scaler.pkl
│   │
│   └── saved_models/
│         ├── facial_emotion_model.keras
│         ├── facial_emotion_mobilenet.keras
│         └── speech_emotion_svm.pkl
│
├── notebooks/
│   ├── Facial Emotion Recognition.ipynb
│   ├── Speech Emotion Recognition.ipynb
│   └── Mental Health Questionnaire.ipynb
│
├── reports/
│   ├── figures/
│   └── presentation/
│
├── utils/
│   ├── fusion.py
│   └── model_loader.py
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

## 1. FER2013 (Facial Emotion Dataset)

Download from Kaggle

https://www.kaggle.com/datasets/msambare/fer2013

Place the extracted dataset inside

```
datasets/facial/
```

Expected Structure

```
datasets/

└── facial/

    ├── train/
    │      ├── angry
    │      ├── disgust
    │      ├── fear
    │      ├── happy
    │      ├── neutral
    │      ├── sad
    │      └── surprise
    │
    └── test/
           ├── angry
           ├── disgust
           ├── fear
           ├── happy
           ├── neutral
           ├── sad
           └── surprise
```

---

## 2. RAVDESS Speech Emotion Dataset

Download

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Extract into

```
datasets/speech/
```

Expected Folder

```
datasets/

└── speech/

      Audio_Speech_Actors_01-24/
```

---

## 3. Mental Health Questionnaire

Create

```
datasets/questionnaire/

mental_health_questions.csv
```

Columns

```
QuestionID
Question
ReverseScore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Talk-To-Mind.git
```

Move into project

```bash
cd Talk-To-Mind
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Open terminal

```bash
cd app
```

Run

```bash
streamlit run app.py
```

---

# Model Performance

## Facial Emotion Recognition

| Model | Accuracy |
|--------|----------|
| CNN | 57.2% |
| MobileNetV2 | 54.5% |

---

## Speech Emotion Recognition

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 51.7% |
| Random Forest | 52.8% |
| Support Vector Machine | 68.1% |

---

# Dashboard Features

The Streamlit application provides

- Facial emotion detection
- Speech emotion detection
- Mental health questionnaire
- Wellbeing score
- Risk prediction
- Personalized recommendations

---

# Reports

Project reports include

- Exploratory Data Analysis
- Model Evaluation
- Confusion Matrix
- Classification Report
- Accuracy Curves
- Loss Curves
- Model Comparison Charts

Generated figures are stored in

```
reports/figures/
```

---

# Future Improvements

- Real-time video emotion tracking
- Real-time microphone emotion analysis
- Transformer-based speech models
- Clinical dataset integration
- Mobile application deployment
- Cloud deployment

---

# Author

Shivam Patel

AI & Machine Learning Internship Project

---

# Acknowledgements

Datasets

FER2013

https://www.kaggle.com/datasets/msambare/fer2013

RAVDESS

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Libraries

- TensorFlow
- Scikit-Learn
- OpenCV
- Librosa
- Streamlit
- NumPy
- Pandas

---

# License

This project is intended for educational and internship purposes.