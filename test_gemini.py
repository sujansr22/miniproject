import google.generativeai as genai
from config import Config
import traceback

def test_gemini():
    api_key = Config.GEMINI_API_KEY
    print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("Sending test message to Gemini...")
        response = model.generate_content("Hello, this is a test from AgriBot Pro.")
        print("\n--- RESPONSE START ---")
        print(response.text)
        print("--- RESPONSE END ---\n")
        print("Gemini connectivity test: SUCCESS")
        
    except Exception as e:
        print("\n--- GEMINI TEST FAILED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        print("--------------------------\n")

if __name__ == "__main__":
    test_gemini()
