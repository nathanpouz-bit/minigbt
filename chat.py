import requests
from model import MODEL_NAME

def generate_response(user_input):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": user_input,
            "stream": False
        }
    )

    return response.json()["response"]
