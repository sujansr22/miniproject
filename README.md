# Smart crop pridiction based on soil fertility 
Overview

This project uses Machine Learning to recommend the most suitable crop and predict possible crop diseases based on soil fertility and environmental conditions.
It helps farmers make data-driven decisions to increase productivity, reduce risks, and improve sustainability.

#Table of Contents

Objective

Features

Tech Stack

Dataset Description

System Architecture

Model Training

Web Application

How to Run the Project

Results

Future Enhancements

Contributors

License

**Objective**

To build a machine learning–based system that predicts the most suitable crop for a given soil composition (NPK levels, pH, temperature, humidity) and forecasts potential crop diseases using trained models.

Features

Predicts the best crop based on soil nutrients and weather conditions

Predicts potential diseases associated with that crop

Interactive Flask web interface for easy input and visualization

Machine Learning models trained using Random Forest Classifier

Uses .pkl files for fast, real-time predictions

Modular and scalable design (ready for cloud or IoT integration)

**Tech Stack**
Layer	Technologies
Frontend	HTML, CSS, JavaScript
Backend	Flask (Python Web Framework)
Machine Learning	Scikit-learn (RandomForestClassifier, LabelEncoder)
Data Handling	Pandas, NumPy
Model Storage	Joblib (.pkl files)
IDE	VS Code / Jupyter Notebook

**Dataset Description**

The dataset used: Updated_Crop_Recommendation_with_Disease_Info.csv

**Feature	Description**

Nitrogen (N)	Nitrogen content in soil
Phosphorus (P)	Phosphorus content in soil
Potassium (K)	Potassium content in soil
Temperature	Temperature in °C
Humidity	Humidity percentage
pH_Value	Soil pH value
Recommended_Crop	Best crop suitable for the soil
Disease	Potential disease affecting that crop

**System Architecture**

User Input (N, P, K, Temp, Humidity, pH)
            ↓
Flask Backend (API /predict)
            ↓
Machine Learning Models (.pkl)
    → crop_model.pkl → Recommended Crop
    → disease_model.pkl → Predicted Disease
            ↓
Response sent to Frontend (HTML + JS)
            ↓
User sees results in browser

**Model Training**

Script: train_model.py

Steps:

Load dataset using pandas

Handle missing values using fillna()

Encode labels using LabelEncoder()

Train two Random Forest Classifiers:

crop_model → Predicts best crop

disease_model → Predicts possible disease

Save trained models using joblib.dump()

Output Files:

crop_model.pkl  
disease_model.pkl  
label_encoder_crop.pkl  
label_encoder_disease.pkl  

**Web Application**
Files:

predict_crop_disease.py → Flask backend

predict.html → Frontend form

style.css → Page styling

script.js → Handles API call and result display

API Endpoint:
POST /predict

Example JSON Request:
{
  "nitrogen": 90,
  "phosphorus": 40,
  "potassium": 50,
  "temperature": 28,
  "humidity": 70,
  "ph_value": 6.5
}

Example Response:
{
  "prediction": {
    "crop": "Rice",
    "disease": "Bacterial Leaf Blight"
  }
}

**How to Run the Project**

Step 1: Clone the Repository
git clone https://github.com/<your-username>/Smart-Crop-Prediction.git
cd Smart-Crop-Prediction

Step 2: Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run Flask App
python predict_crop_disease.py

Step 5: Open in Browser
Go to → http://localhost:5000

Results
Metric	Score
Accuracy	92%
Precision	90%
Recall	93%
F1-Score	91%

Model successfully predicts crops like Rice, Maize, Wheat, Banana, and diseases like Blight, Rust, Leaf Spot, etc.

**Future Enhancements**

Integrate real-time weather API
Deploy to AWS / Render / Heroku
Build mobile app interface
Add IoT sensor inputs for live soil readings
Include market price prediction for economic insights

**Contributors**

Developed by:

Sujan Gowda
Varun Raj E T
Rohan Noah
L Chaithanya

Department of Data Science,
Academic Year 2024–25

**Acknowledgements**
Special thanks to:
Scikit-learn, Flask, and Pandas communities
Open-source agricultural data providers
Faculty guides and mentors who supported this project
