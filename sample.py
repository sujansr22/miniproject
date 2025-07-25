# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Updated_Crop_Recommendation.csv')

# Check the first few rows to confirm the dataset is loaded
print(data.head())
# Step 2: Encoding the target variable (Recommended_Crop)
label_encoder = LabelEncoder()
data['Recommended_Crop'] = label_encoder.fit_transform(data['Recommended_Crop'])

# Separate features and target
X = data.drop('Recommended_Crop', axis=1)  # Features
y = data['Recommended_Crop']  # Target variable
# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Visualize feature importance
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
features = X.columns

plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Crop Prediction")
plt.show()
import joblib

# Save the trained model to a file
joblib.dump(model, 'crop_recommendation_model.pkl')

