from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from flask_mysqldb import MySQL
import bcrypt
import joblib
import json
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = 'your-secret-key-change-this-in-production'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Sujan@123'
app.config['MYSQL_DB'] = 'crop_prediction_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL
mysql = MySQL(app)

# Load models and label encoders
crop_model = joblib.load('crop_model.pkl')
disease_model = joblib.load('disease_model.pkl')
label_encoder_crop = joblib.load('label_encoder_crop.pkl')
label_encoder_disease = joblib.load('label_encoder_disease.pkl')

# Load crop profiles for explanation
crop_profiles = {}
try:
    with open('model_metrics.json', 'r') as f:
        metrics_data = json.load(f)
        crop_profiles = metrics_data.get('crop_profiles', {})
except Exception as e:
    print(f"Warning: Could not load crop profiles: {e}")

# Load disease info
disease_info = {}
try:
    with open('disease_info.json', 'r') as f:
        disease_info = json.load(f)
except Exception as e:
    print(f"Warning: Could not load disease info: {e}")

# Load Indian State-Crop mapping
indian_crops = {}
try:
    with open('indian_crops.json', 'r') as f:
        indian_crops = json.load(f)
except Exception as e:
    print(f"Warning: Could not load Indian crops data: {e}")

def generate_explanation(crop_name, input_features):
    if crop_name not in crop_profiles:
        return "Suitable conditions match this crop's requirements."
    
    profile = crop_profiles[crop_name]
    reasons = []
    
    # Feature mapping: Input key -> Profile key
    feature_map = {
        'nitrogen': 'Nitrogen',
        'phosphorus': 'Phosphorus',
        'potassium': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph_value': 'pH_Value'
    }
    
    diffs = []
    for input_key, profile_key in feature_map.items():
        if profile_key in profile:
            input_val = float(input_features[input_key])
            profile_val = float(profile[profile_key])
            if profile_val > 0:
                diff = abs(input_val - profile_val) / profile_val
                diffs.append((profile_key, diff))
    
    # Sort by smallest difference (closest match)
    diffs.sort(key=lambda x: x[1])
    
    # Pick top 2 matches
    top_matches = [d[0] for d in diffs[:2]]
    
    if top_matches:
        return f"{crop_name} is recommended because {top_matches[0]} and {top_matches[1]} levels are optimal."
    
    return f"{crop_name} fits the current soil and weather profile."

@app.route('/')
def welcome():
    # Redirect to login page
    return redirect(url_for('login_page'))

@app.route('/app')
def main_app():
    # Check if user is logged in
    if 'user_email' not in session:
        return redirect(url_for('login_page'))
    # Serve the Main Application Page
    return send_from_directory('.', 'app.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph_value']
        
        # Validate input data
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract values
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph_value = float(data['ph_value'])
        state = data.get('state', '') # State is optional

        # --- VALIDATION ---
        # Define ranges: (min, max)
        ranges = {
            'Nitrogen': (0, 140),
            'Phosphorus': (0, 145),
            'Potassium': (0, 205),
            'Temperature': (0, 60),
            'Humidity': (0, 100),
            'pH': (0, 14)
        }

        # Check ranges
        if not (ranges['Nitrogen'][0] <= nitrogen <= ranges['Nitrogen'][1]):
            return jsonify({'error': f'Nitrogen must be between {ranges["Nitrogen"][0]} and {ranges["Nitrogen"][1]}'}), 400
        if not (ranges['Phosphorus'][0] <= phosphorus <= ranges['Phosphorus'][1]):
            return jsonify({'error': f'Phosphorus must be between {ranges["Phosphorus"][0]} and {ranges["Phosphorus"][1]}'}), 400
        if not (ranges['Potassium'][0] <= potassium <= ranges['Potassium'][1]):
            return jsonify({'error': f'Potassium must be between {ranges["Potassium"][0]} and {ranges["Potassium"][1]}'}), 400
        if not (ranges['Temperature'][0] <= temperature <= ranges['Temperature'][1]):
            return jsonify({'error': f'Temperature must be between {ranges["Temperature"][0]}°C and {ranges["Temperature"][1]}°C'}), 400
        if not (ranges['Humidity'][0] <= humidity <= ranges['Humidity'][1]):
            return jsonify({'error': f'Humidity must be between {ranges["Humidity"][0]}% and {ranges["Humidity"][1]}%'}), 400
        if not (ranges['pH'][0] <= ph_value <= ranges['pH'][1]):
            return jsonify({'error': f'pH value must be between {ranges["pH"][0]} and {ranges["pH"][1]}'}), 400

        # Prepare input for models
        features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph_value]]

        # Make predictions
        crop_probs = crop_model.predict_proba(features)[0]
        
        # Disease prediction with probability
        disease_probs = disease_model.predict_proba(features)[0]
        start_disease_idx = np.argsort(disease_probs)[-1] 
        disease_confidence = float(disease_probs[start_disease_idx]) * 100
        predicted_disease = label_encoder_disease.inverse_transform([start_disease_idx])[0]
        
        # Determine Risk Level
        risk_level = "Low"
        if disease_confidence > 70:
            risk_level = "High"
        elif disease_confidence > 30:
            risk_level = "Medium"

        # Get top 3 crops
        top_3_indices = np.argsort(crop_probs)[-3:][::-1]
        top_3_crops = []
        
        # Input features dictionary for explanation
        input_data = {
            'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium,
            'temperature': temperature, 'humidity': humidity, 'ph_value': ph_value
        }
        
        # Get regional crops if state is provided
        regional_crops = []
        if state and state in indian_crops:
            regional_crops = [c.lower() for c in indian_crops[state]]
        
        for idx in top_3_indices:
            crop_name = label_encoder_crop.inverse_transform([idx])[0]
            confidence = float(crop_probs[idx]) * 100
            explanation = generate_explanation(crop_name, input_data)
            
            # Check regional match
            is_regional = False
            if regional_crops:
                # Simple containment check
                if crop_name.lower() in regional_crops:
                    is_regional = True
            
            top_3_crops.append({
                "crop": crop_name,
                "confidence": round(confidence, 2),
                "explanation": explanation,
                "regional_match": is_regional
            })

        # Enrich disease info (same as before)
        disease_details = disease_info.get(predicted_disease, {
            "description": "Information not available.",
            "symptoms": [],
            "treatment_organic": "N/A",
            "treatment_chemical": "N/A",
            "prevention": "N/A"
        })

        return jsonify({
            'prediction': {
                'top_3_crops': top_3_crops,
                'disease': {
                    'name': predicted_disease,
                    'confidence': round(disease_confidence, 2),
                    'risk_level': risk_level,
                    'details': disease_details
                }
            }
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        metrics_path = 'model_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/login')
def login_page():
    # Serve the Login Page
    return send_from_directory('.', 'login.html')

@app.route('/signup')
def signup_page():
    # Serve the Signup Page
    return send_from_directory('.', 'signup.html')

@app.route('/forgot-password')
def forgot_password_page():
    # Serve the Forgot Password Page
    return send_from_directory('.', 'forgot_password.html')

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'All fields are required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Basic email validation
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if email already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()
        
        if existing_user:
            cur.close()
            return jsonify({'error': 'Email already registered'}), 400
        
        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert new user
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'message': 'User registered successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        
        # Get user from database
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, email, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            # Create session
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['username'] = user['username']
            
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'username': user['username'],
                    'email': user['email']
                }
            }), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('new_password'):
            return jsonify({'error': 'Email and new password are required'}), 400
        
        email = data['email'].strip().lower()
        new_password = data['new_password']
        
        # Check if user exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        
        if not user:
            cur.close()
            return jsonify({'error': 'Email not found'}), 404
        
        # Hash the new password
        password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        # Update password
        cur.execute(
            "UPDATE users SET password_hash = %s WHERE email = %s",
            (password_hash, email)
        )
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'message': 'Password updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# ==================== END AUTHENTICATION ROUTES ====================

if __name__ == '__main__':
    app.run(debug=True, port=5003)
