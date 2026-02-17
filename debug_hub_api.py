import requests
from config import Config

def get_model_info():
    model = Config.HUGGINGFACE_MODEL
    api_url = f"https://huggingface.co/api/models/{model}"
    
    print(f"Querying Hub API: {api_url}")
    try:
        response = requests.get(api_url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # Check for specific keys
            print(f"Model ID: {data.get('id')}")
            print(f"Pipeline Tag: {data.get('pipeline_tag')}")
            print(f"Tags: {data.get('tags')}")
            # Look for inference info? usually not directly exposed but maybe hints
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    get_model_info()
