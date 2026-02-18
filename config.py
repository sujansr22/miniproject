import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'Sujan@123') # Default from app.py
    MYSQL_DB = os.getenv('MYSQL_DB', 'crop_prediction_db')
    MYSQL_CURSORCLASS = 'DictCursor'
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    
    # Google Gemini Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCoJANfXwGKpUuOVrLOMYSBJL8Sej4wYyI')
