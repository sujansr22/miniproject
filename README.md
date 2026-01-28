# Smart crop pridiction based on soil fertility 
## Overview

This project uses Machine Learning to recommend the most suitable crop and predict possible crop diseases based on soil fertility and environmental conditions.
It helps farmers make data-driven decisions to increase productivity, reduce risks, and improve sustainability.


## Objective

To build a machine learning–based system that predicts the most suitable crop for a given soil composition (NPK levels, pH, temperature, humidity) and forecasts potential crop diseases using trained models.

## Features

Predicts the best crop based on soil nutrients and weather conditions

Predicts potential diseases associated with that crop

Interactive Flask web interface for easy input and visualization

Machine Learning models trained using Random Forest Classifier

Uses .pkl files for fast, real-time predictions

Modular and scalable design (ready for cloud or IoT integration)

## Key features

Crop Recommendation based on soil nutrients and environmental factors

Crop Disease Prediction associated with the recommended crop

Model Performance Transparency

Accuracy

Precision

Recall

F1-score

Machine Learning models using Random Forest Classifier

Fast real-time predictions using saved .pkl models

Flask-based web application

Clean UI with:

            Welcome page

            “Get Started” navigation

            Dedicated application page


Modular and scalable design (ready for cloud / IoT integration)

## Tech Stack

Layer	Technologies
Frontend	HTML, CSS, JavaScript
Backend	Flask (Python)
Machine Learning	Scikit-learn (RandomForestClassifier, LabelEncoder)
Data Handling	Pandas, NumPy
Model Storage	Joblib (.pkl files)
Development Tools	VS Code, Jupyter Notebook


## Dataset Used:
Updated_Crop_Recommendation_with_Disease_Info.csv

Feature	            Description
Nitrogen (N)	Nitrogen content in soil
Phosphorus (P)	Phosphorus content in soil
Potassium (K)	Potassium content in soil
Temperature	Temperature in °C
Humidity	Humidity percentage
pH_Value	Soil pH value
Recommended_Crop	Best crop for given soil
Disease	Potential disease affecting the crop

**Feature	Description**

Nitrogen (N)	Nitrogen content in soil
Phosphorus (P)	Phosphorus content in soil
Potassium (K)	Potassium content in soil
Temperature	            Temperature in °C
Humidity	            Humidity percentage
pH_Value	            Soil pH value
Recommended_Crop	Best crop suitable for the soil
Disease	            Potential disease affecting that crop


## System Architecture

User Input (N, P, K, Temp, Humidity, pH)
        ↓
Flask Backend
        ↓
Crop Prediction Model (crop_model.pkl)
        ↓
Disease Prediction Model (disease_model.pkl)
        ↓
Evaluation Metrics (model_metrics.json)
        ↓
Frontend (HTML + JS)
        ↓
Results displayed in browser


## Model Training

Training Script: train_model.py

Training Pipeline

Load dataset using Pandas

Handle missing values

Encode labels using LabelEncoder

Split data into Train (80%) / Test (20%)

Train Random Forest models

Evaluate models using:

            Accuracy

            Precision

            Recall

            F1-score

Save evaluation results to model_metrics.json

Retrain models on full dataset for final deployment

Save trained models using Joblib

Output Files

            crop_model.pkl

            disease_model.pkl

            label_encoder_crop.pkl

            label_encoder_disease.pkl

            model_metrics.json
            

## Web Application

### Backend

            app.py – Flask application

            /predict – Prediction endpoint

            /metrics – Model performance metrics API

### Frontend

            welcome.html – Intro / landing page

            index.html – Main application page
            
            style.css – Styling

            script.js – API calls and UI updates

## How to Run the Project

### Step 1: Clone the Repository
git clone https://github.com/<your-username>/Smart-Crop-Prediction.git
cd Smart-Crop-Prediction

### Step 2: Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Run Flask App
python predict_crop_disease.py

### Step 5: Open in Browser
Go to → http://localhost:5000

Results
Metric	Score
Accuracy	92%
Precision	90%
Recall	93%
F1-Score	91%

Model successfully predicts crops like Rice, Maize, Wheat, Banana, and diseases like Blight, Rust, Leaf Spot, etc.


## Future Enhancements

Integrate real-time weather API
Deploy to AWS / Render / Heroku
Build mobile app interface
Add IoT sensor inputs for live soil readings
Include market price prediction for economic insights


## Contributors

### Developed by:

            Sujan Gowda, 
            Varun Raj E T, 
            Rohan Noah, 
            L Chaithanya

Department of Data Science,
Academic Year 2024–25


## Acknowledgements

Special thanks to:

            Scikit-learn, Flask, and Pandas communities
            Open-source agricultural data providers
            Faculty guides and mentors who supported this project
