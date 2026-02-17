import google.generativeai as genai
from config import Config
import os

def test_gemini():
    api_key = Config.GEMINI_API_KEY
    print(f"Testing Gemini API...")
    print(f"Key present: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"Key starts with: {api_key[:4]}...")

    try:
        genai.configure(api_key=api_key)
        
        # List all models to a file
        with open('d:/VSCodes/miniproject/available_models.txt', 'w') as f:
            for m in genai.list_models():
                f.write(f"{m.name}\n")
        print("Model list saved to available_models.txt")

        target_models = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-1.5-pro-latest']
        
        for model_name in target_models:
            print(f"\nAttempting to generate content with: {model_name}")
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                print(f"SUCCESS with {model_name}!")
                print(response.text)
                break
            except Exception as e:
                print(f"Failed with {model_name}: {e}")

    except Exception as e:
        print(f"\nERROR DETECTED:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini()
