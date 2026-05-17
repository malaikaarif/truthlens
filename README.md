# TruthLens — AI Fake News Detector

An AI-powered fake news detection system that combines a trained Machine Learning model with Groq's LLaMA 3.3 70B to classify news as real or fake — and explain exactly why.

## Live Demo
[truthlens-1rzw.onrender.com](https://truthlens-1rzw.onrender.com)

## How It Works

1. User pastes any news article or headline
2. Trained Logistic Regression classifier predicts real or fake with confidence score
3. Groq LLaMA 3.3 70B explains which specific patterns triggered the classification
4. Results returned as clean JSON via REST API

## Tech Stack

- **Python** — core language
- **scikit-learn** — Logistic Regression classifier trained on 44,898 articles
- **FastAPI** — REST API backend with auto Swagger docs
- **Groq API** — LLaMA 3.3 70B for explainability
- **HTML/CSS/JS** — frontend UI
- **Render** — deployment

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.71% |
| Fake Precision | 0.99 |
| Real Precision | 0.98 |
| F1-Score | 0.99 |
| Training articles | 44,898 |

## Dataset

ISOT Fake News Dataset — University of Victoria
- Real news: Reuters.com
- Fake news: PolitiFact.com
- Total articles: 44,898

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /detect | Analyze news text |
| GET | /health | API health check |
| GET | /docs | Swagger UI |

## Run Locally

```bash
# Clone repo
git clone https://github.com/malaikaarif/truthlens.git
cd truthlens

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run server
uvicorn main:app --reload
```

## Limitations

- Trained on 2016-2017 English news data
- Detects language patterns, not current facts
- May struggle with very short texts
- Limited Urdu language support

## Future Improvements

- Retrain on recent Pakistani news data
- Add Urdu language support
- Continuous retraining pipeline
- BERT-based model for better contextual understanding

## Built By

**Malaika Arif** — CS Student, COMSATS University Islamabad
- GitHub: [github.com/malaikaarif](https://github.com/malaikaarif)
- LinkedIn: [linkedin.com/in/malaika-arif204718295](https://linkedin.com/in/malaika-arif204718295)