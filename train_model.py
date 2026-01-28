import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# --- EVALUATION PHASE ---
# Split data for evaluation (80% train, 20% test)
X_train, X_test, y_crop_train, y_crop_test, y_disease_train, y_disease_test = train_test_split(
    X, y_crop_encoded, y_disease_encoded, test_size=0.2, random_state=42
)

# Train temporary models for evaluation
eval_crop_model = RandomForestClassifier(random_state=42)
eval_crop_model.fit(X_train, y_crop_train)

eval_disease_model = RandomForestClassifier(random_state=42)
eval_disease_model.fit(X_train, y_disease_train)

# Predict on test set
y_crop_pred = eval_crop_model.predict(X_test)
y_disease_pred = eval_disease_model.predict(X_test)

# Calculate metrics
metrics = {
    "crop_model": {
        "accuracy": float(accuracy_score(y_crop_test, y_crop_pred)),
        "precision": float(precision_score(y_crop_test, y_crop_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_crop_test, y_crop_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_crop_test, y_crop_pred, average='weighted', zero_division=0))
    },
    "disease_model": {
        "accuracy": float(accuracy_score(y_disease_test, y_disease_pred)),
        "precision": float(precision_score(y_disease_test, y_disease_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_disease_test, y_disease_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_disease_test, y_disease_pred, average='weighted', zero_division=0))
    }
}

# --- EXPLAINABILITY METADATA ---
# Calculate mean feature values for each crop to help with explanations
# "Why Rice? Because Nitrogen is usually around 80..."
crop_profiles = {}
features_list = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value']

# Group by crop and calculate mean
# We need to use the original dataframe 'df' but we need the decoded labels for keys
# It's easier to just iterate over unique crops
for crop in df['Recommended_Crop'].unique():
    crop_data = df[df['Recommended_Crop'] == crop][features_list]
    crop_profiles[crop] = crop_data.mean().to_dict()

# Add to metrics/metadata file
metrics['crop_profiles'] = crop_profiles

# Save metrics (updated with profiles)
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Evaluation metrics and crop profiles saved to model_metrics.json")
print("First 2 crop profiles:", list(crop_profiles.items())[:2])


# --- FINAL TRAINING PHASE ---
# Retrain on FULL dataset for production
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
