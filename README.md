# Placement Predictor

A Machine Learning-based web application that predicts student placement using CGPA and IQ as input features.

## Overview
This project implements an end-to-end Machine Learning pipeline including data preprocessing, feature scaling, model training, and deployment. The model is integrated into an interactive web application using Streamlit for real-time predictions.

## Features
- Logistic Regression model with ~85% accuracy
- Feature scaling using StandardScaler
- Probability-based prediction output
- Interactive user interface using Streamlit

## Tech Stack
- Python
- NumPy
- Scikit-learn
- Streamlit

## Machine Learning Workflow
- Data preprocessing and cleaning
- Train-test split
- Feature scaling using StandardScaler
- Model training using Logistic Regression
- Model evaluation and accuracy calculation
- Model serialization using Pickle

## Project Structure
placement-predictor/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md

## How to Run
pip install -r requirements.txt  
streamlit run app.py

## Results
- Model Accuracy: ~85%
- Real-time prediction with probability score

## Future Improvements
- Use advanced models (Random Forest, SVM)
- Improve dataset quality
- Add more input features
- Deploy on cloud platform
