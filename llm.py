import requests

def ask_therapist(emotion: str, user_message: str) -> str:
    prompt = f"""
    You are a grounded, emotionally supportive therapist chatting with someone in real time.

    Your system can detect the user's facial expression — and right now, they seem to be feeling: "{emotion.lower()}".

    The user just said: "{user_message}"

    Respond like a real therapist. Let their words guide your reply. You may gently acknowledge their emotion if it seems relevant, but don't overdo it. Avoid cheesy lines or overly positive reactions.

    Be warm, human, and attentive. Keep it short and focused.
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
