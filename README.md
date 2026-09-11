# 🏥 AI Healthcare ML System

### Intelligent Multi-Disease Prediction Platform

An AI-powered healthcare application that uses Machine Learning to predict the likelihood of multiple diseases based on user-provided medical data.

## 🚀 Overview

The AI Healthcare ML System is a multi-disease prediction platform designed to assist users in understanding potential health risks through Machine Learning models.

The system provides prediction modules for:

- 🩸 Diabetes
- ❤️ Heart Disease
- 🧠 Parkinson's Disease
- 🎗️ Breast Cancer
- 🧠 Stroke

## ✨ Features

- Multi-disease prediction
- Machine Learning based predictions
- Interactive healthcare dashboard
- Disease-specific prediction modules
- Pre-trained ML models
- Dataset-based model training
- Simple and user-friendly interface

## 🧠 Machine Learning

The project uses Python and Machine Learning techniques to train and evaluate disease prediction models.

### Supported Models

- Diabetes Prediction
- Heart Disease Prediction
- Stroke Prediction
- Parkinson's Disease Prediction
- Breast Cancer Prediction

## 🛠️ Technologies Used

- Python
- Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib
- HTML & CSS

## 📂 Project Structure

```text
healthcare-ml-system/
│
├── datasets/
│   ├── breast_cancer.csv
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── parkinsons.csv
│   └── stroke.csv
│
├── models/
│   ├── cancer_model.pkl
│   ├── heart_model.pkl
│   ├── parkinsons_model.pkl
│   └── stroke_model.pkl
│
├── pages/
│   ├── 1_Diabetes_Prediction.py
│   ├── 2_Heart_Disease_Prediction.py
│   ├── 3_Stroke_Prediction.py
│   ├── 4_Parkinsons_Prediction.py
│   ├── 5_Breast_Cancer_Prediction.py
│   └── 6_Health_Dashboard.py
│
├── app.py
├── styles.css
├── train_cancer_model.py
├── train_diabetes_model.py
├── train_heart_model.py
├── train_parkinsons_model.py
└── train_stroke_model.py
