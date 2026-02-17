import requests
import os
from config import Config

def test_hf_connection():
    api_key = Config.HUGGINGFACE_API_KEY
    model = "gpt2"
    
    # Try the specific router URL that was 404 for Mistral
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
    
    models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3-8B-Instruct" 
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"inputs": "Hello"}

    for m in models:
        url = f"https://router.huggingface.co/hf-inference/models/{m}"
        print(f"\nTesting: {url}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            print(f"Status: {response.status_code}")
            # print(f"Text: {response.text[:200]}")
            if response.status_code == 200:
                print("SUCCESS!")
            elif response.status_code == 503:
                 print("Loading (503) - But endpoint exists!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_hf_connection()
