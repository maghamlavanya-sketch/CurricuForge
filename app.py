import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# Router chat endpoint for fallback HTTP requests
ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Build headers only if we have a token; otherwise warn the user
if not HF_TOKEN:
    print("Warning: HUGGINGFACE_API_KEY not set. External requests will likely fail.")

headers = {"Content-Type": "application/json"}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

# Initialize huggingface_hub client if available
hf_client = None
if InferenceClient is not None:
    try:
        hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else InferenceClient()
    except Exception:
        hf_client = None

def generate_timetable(subject, free_time, level, plan_type):

    prompt = f"""
You are an expert study planner.

Create a structured {plan_type} timetable.

Subject: {subject}
Available time per day: {free_time}
Level: {level}

Make it practical, clear, and well-organized.
"""

    # Prefer using the official InferenceClient if available
    if hf_client is not None:
        try:
            out = hf_client.text_generation(prompt, model=MODEL_ID, max_new_tokens=500, temperature=0.7)
            if isinstance(out, str):
                return out
            if isinstance(out, dict) and "generated_text" in out:
                return out["generated_text"]
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
            return str(out)
        except Exception as e:
            # Fall back to HTTP router below
            fallback_error = f"InferenceClient error: {e}"

    # Fallback: use router chat completions endpoint (OpenAI-compatible)
    chat_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        response = requests.post(ROUTER_CHAT_URL, headers=headers, json=chat_payload, timeout=30)
    except requests.RequestException as e:
        return f"Request error: {e}"

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return response.text

        # OpenAI-compatible response parsing
        try:
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and len(choices) > 0:
                msg = choices[0].get("message") or choices[0].get("text") or choices[0].get("delta")
                if isinstance(msg, dict):
                    return msg.get("content") or str(msg)
                if isinstance(msg, str):
                    return msg
        except Exception:
            pass

        # Try other shapes
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        return str(data)
    else:
        # include prior InferenceClient error if present
        msg = f"Error: {response.status_code} - {response.text}"
        if 'fallback_error' in locals():
            msg = fallback_error + " | " + msg
        return msg


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        subject = request.form["subject"]
        free_time = request.form["free_time"]
        level = request.form["level"]
        plan_type = request.form["plan_type"]

        result = generate_timetable(subject, free_time, level, plan_type)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)