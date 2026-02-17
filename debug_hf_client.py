from huggingface_hub import InferenceClient
from config import Config
import os

def test_client():
    api_key = Config.HUGGINGFACE_API_KEY
    candidate_models = [
        "tiiuae/falcon-7b-instruct",
        "google/gemma-7b-it",
        "meta-llama/Llama-2-7b-chat-hf",
        "bigscience/bloom"
    ]
    
    client = InferenceClient(api_key=api_key)
    
    for m in candidate_models:
        print(f"\nTesting Model: {m}")
        try:
            response = client.text_generation(
                model=m,
                prompt="Hello",
                max_new_tokens=10
            )
            print("SUCCESS!")
            print(f"Working Model Found: {m}")
            print(response)
            break
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    test_client()
