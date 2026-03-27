# TruthLens — AI vs Human Text Detection System

> Dual-model ensemble ML system with linguistic fingerprinting and a professional web UI.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Custom dataset (`ai_vs_human_text-1.xlsx`) |
| Total Samples | 1,299 |
| Label Distribution | Human: 651 / AI: 648 (balanced) |
| Features Used | `text`, `label` |

---

## Preprocessing Steps

1. Lowercase conversion
2. Punctuation removal using regex
3. CountVectorizer with **bigrams** (`ngram_range=(1,2)`, `max_features=5000`)
4. 80/20 train-test split with `random_state=42`

---

## Models & Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | ~95.8% |
| Naive Bayes | ~96.9% |

---

## Advanced Features (Beyond Problem Statement)

| Feature | Description |
|---|---|
| **Bigram Vectorizer** | Captures phrase-level patterns, not just single words |
| **Ensemble Confidence** | Weighted combination (55% LR + 45% NB) for more robust prediction |
| **Model Agreement Check** | Flags when both models disagree — indicates ambiguous text |
| **Linguistic Fingerprinting** | Extracts 8 features: word count, sentence count, avg word/sentence length, vocab richness, punctuation density, burstiness |
| **Risk Score** | Derived from `(1 - ensemble_confidence) × 100` |
| **Web UI** | Flask-based professional dark UI with animated progress bars |
| **REST API** | `/predict` endpoint accepts JSON, returns structured response |

---

## Decision Layer

| Condition | Verdict |
|---|---|
| Ensemble ≥ 0.85 AND both models agree | ✅ Acceptable |
| Ensemble ≥ 0.65 | ❓ Needs Review |
| Ensemble < 0.65 | ⚠️ Likely AI-generated |

---

## How to Run

```bash
# Step 1: Install dependencies
pip install flask scikit-learn pandas openpyxl

# Step 2: Train and save models
python model.py

# Step 3: Launch the web app
python app.py

# Step 4: Open browser
# → http://127.0.0.1:5000
```

---

## API Usage

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here"}'
```

**Response:**
```json
{
  "logistic_regression": { "prediction": "ai", "confidence": 0.91, "ai_probability": 0.91 },
  "naive_bayes":         { "prediction": "ai", "confidence": 0.99, "ai_probability": 0.99 },
  "decision": {
    "verdict": "Likely AI-generated",
    "ensemble_confidence": 0.948,
    "model_agreement": true,
    "risk_score": 5.2
  },
  "linguistic_features": {
    "word_count": 21, "vocab_richness": 0.952, "burstiness": 1.5, ...
  }
}
```

---

## Tech Stack

- Python 3.x
- scikit-learn (ML models)
- Flask (web server + REST API)
- HTML / CSS / Vanilla JS (frontend)
- No external AI APIs used ✅
- Runs fully offline ✅
