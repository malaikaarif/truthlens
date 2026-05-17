from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
from groq import Groq

app = FastAPI(
    title="Fake News Detector API",
    description="ML + LLM powered fake news detection. Trained on 44K articles.",
    version="1.0.0"
)

print("Loading model...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded.")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class NewsInput(BaseModel):
    text: str

class DetectionResult(BaseModel):
    label: str
    confidence: float
    verdict: str
    explanation: str

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/detect", response_model=DetectionResult)
def detect(news: NewsInput):
    if not news.text or len(news.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short.")

    text = news.text.strip()

    proba = model.predict_proba([text])[0]
    pred = model.predict([text])[0]

    confidence = round(float(max(proba)) * 100, 2)

    if pred == 1:
        label = "Real"
        verdict = "This article appears credible."
    elif confidence < 70:
        label = "Uncertain"
        verdict = "Low confidence — model is uncertain. Verify from multiple sources."
    else:
        label = "Fake"
        verdict = "This article shows signs of misinformation."

    prompt = f"""You are a fake news analysis expert.

A machine learning model classified this news as: {label} ({confidence}% confidence)

News text:
\"\"\"{text[:1000]}\"\"\"

In 3-4 sentences, explain specifically:
- Which words, phrases, or patterns in THIS text triggered the {label} classification
- What linguistic red flags or trust signals you can see
- One specific thing the reader should verify

Be specific to this text. No generic advice."""

    groq_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    explanation = groq_response.choices[0].message.content.strip()

    return DetectionResult(
        label=label,
        confidence=confidence,
        verdict=verdict,
        explanation=explanation
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}