import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('Updated_Crop_Recommendation_with_Disease_Info.csv')

# Check for missing values and handle them
if df.isnull().sum().any():
    print("Dataset contains missing values. Filling with default values.")
    df = df.fillna("Unknown Disease")

# Feature columns and target labels
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value']]
y_crop = df['Recommended_Crop']
y_disease = df['Disease']

# Encode crop and disease labels
label_encoder_crop = LabelEncoder()
y_crop_encoded = label_encoder_crop.fit_transform(y_crop)

label_encoder_disease = LabelEncoder()
y_disease_encoded = label_encoder_disease.fit_transform(y_disease)

# Train models
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(X, y_crop_encoded)

disease_model = RandomForestClassifier(random_state=42)
disease_model.fit(X, y_disease_encoded)

# Save models and encoders
joblib.dump(crop_model, 'crop_model.pkl')
joblib.dump(disease_model, 'disease_model.pkl')
joblib.dump(label_encoder_crop, 'label_encoder_crop.pkl')
joblib.dump(label_encoder_disease, 'label_encoder_disease.pkl')

print("Models and LabelEncoders saved successfully.")
# Ensure no multiple crops in the label
df['Recommended_Crop'] = df['Recommended_Crop'].str.split(',').str[0].str.strip()
