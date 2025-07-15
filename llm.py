import requests

def ask_therapist(emotion: str, user_message: str) -> str:
    prompt = f"""
You are a compassionate therapist assisting someone in real time. 
The system detects the user's facial emotion as "{emotion}".

The user just said: "{user_message}"

Respond as if you're having a warm, caring conversation with them. 
Validate their emotion, offer support, and encourage them to open up more if needed.
Keep responses short, natural, and emotionally aware.
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:7b",  # match your installed model name
                "prompt": prompt,
                "stream": False
            }
        )
        data = res.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"[Error reaching LLM: {str(e)}]"
