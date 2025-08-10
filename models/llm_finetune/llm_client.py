import os, httpx, json

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

async def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload)
        r.raise_for_status()
        data= r.json()
        return data.get("response", "")
    
# What Happens
# The function sends:

# json

# {
#   "model": "qwen2.5:3b-instruct",
#   "prompt": "Write a haiku about AI.",
#   "stream": false,
#   "options": {"temperature": 0.2, "num_predict": 512}
# }
# to:


# http://localhost:11434/api/generate
# Ollama processes the prompt using the model and returns something like:


# {
#   "model": "qwen2.5:3b-instruct",
#   "created_at": "2025-08-10T10:00:00Z",
#   "response": "Silent circuits hum,\nThoughts like rivers intertwine,\nDreams of code take flight."
# }
# Your Python code extracts the "response" value and prints:


# Silent circuits hum,
# Thoughts like rivers intertwine,
# Dreams of code take flight.