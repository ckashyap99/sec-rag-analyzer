import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(prompt, model="mistral"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama Error: {response.text}")