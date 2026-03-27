from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import math
import os

app = Flask(__name__)

# ─── Load Models ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
lr_model   = pickle.load(open(os.path.join(BASE, "lr_model.pkl"), "rb"))
nb_model   = pickle.load(open(os.path.join(BASE, "nb_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE, "vectorizer.pkl"), "rb"))

# ─── Linguistic Feature Extractor ───────────────────────────────────────────
def extract_features(text):
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    punct_count = len(re.findall(r'[^\w\s]', text))
    unique_words = set(w.lower() for w in words)

    word_count      = len(words)
    char_count      = len(text)
    sentence_count  = max(len(sentences), 1)
    avg_word_len    = round(sum(len(w) for w in words) / max(word_count, 1), 2)
    avg_sent_len    = round(word_count / sentence_count, 2)
    punct_density   = round(punct_count / max(word_count, 1), 3)
    vocab_richness  = round(len(unique_words) / max(word_count, 1), 3)

    # Burstiness: std deviation of sentence lengths
    sent_lengths = [len(s.split()) for s in sentences]
    if len(sent_lengths) > 1:
        mean_sl = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
        burstiness = round(math.sqrt(variance), 2)
    else:
        burstiness = 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sent_len,
        "punctuation_density": punct_density,
        "vocab_richness": vocab_richness,
        "burstiness": burstiness,
    }

# ─── Preprocess ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# ─── Decision Layer ─────────────────────────────────────────────────────────
def decision_layer(lr_conf, nb_conf, lr_pred, nb_pred):
    # Ensemble confidence
    ensemble_conf = round(0.55 * lr_conf + 0.45 * nb_conf, 4)

    # Agreement check
    agreement = lr_pred == nb_pred

    # Confidence difference (important for uncertainty)
    diff = abs(lr_conf - nb_conf)

    # FINAL LOGIC
    if ensemble_conf >= 0.85 and agreement:
        verdict = "Acceptable"
        verdict_icon = "✅"
        verdict_color = "green"

    elif not agreement or diff > 0.15:
        verdict = "Needs Review"
        verdict_icon = "❓"
        verdict_color = "yellow"

    elif ensemble_conf >= 0.65:
        verdict = "Needs Review"
        verdict_icon = "❓"
        verdict_color = "yellow"

    else:
        verdict = "Likely AI-generated"
        verdict_icon = "⚠️"
        verdict_color = "red"

    risk_score = round((1 - ensemble_conf) * 100, 1)

    return {
        "verdict": verdict,
        "verdict_icon": verdict_icon,
        "verdict_color": verdict_color,
        "ensemble_confidence": ensemble_conf,
        "model_agreement": agreement,
        "confidence_difference": diff,
        "risk_score": risk_score,
    }

# ─── API Endpoint ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    # 🔹 Input validation
    if not text or len(text.split()) < 3:
        return jsonify({"error": "Please enter at least 3 words."}), 400

    # 🔹 Feature extraction
    features = extract_features(text)
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    # 🔹 Logistic Regression
    lr_pred = lr_model.predict(vec)[0]
    lr_proba = lr_model.predict_proba(vec)[0]
    lr_ai_index = list(lr_model.classes_).index("ai") if "ai" in lr_model.classes_ else 1
    lr_ai_p = float(lr_proba[lr_ai_index])
    lr_conf = float(max(lr_proba))

    # 🔹 Naive Bayes
    nb_pred = nb_model.predict(vec)[0]
    nb_proba = nb_model.predict_proba(vec)[0]
    nb_ai_index = list(nb_model.classes_).index("ai") if "ai" in nb_model.classes_ else 1
    nb_ai_p = float(nb_proba[nb_ai_index])
    nb_conf = float(max(nb_proba))

    # 🔥 FINAL DECISION (NO MISMATCH)
    final_pred = lr_pred   # use Logistic Regression as main
    final_conf = lr_ai_p   # AI probability

    if str(final_pred).lower() in ["ai", "1"]:
        verdict = "AI"
        verdict_icon = "⚠️"
        verdict_color = "red"
    else:
        verdict = "Human"
        verdict_icon = "✅"
        verdict_color = "green"

    decision = {
        "verdict": verdict,
        "verdict_icon": verdict_icon,
        "verdict_color": verdict_color,
        "ensemble_confidence": round(final_conf, 4),
        "model_agreement": lr_pred == nb_pred,
        "risk_score": round(final_conf * 100, 1),
    }

    # 🔹 Final response
    return jsonify({
        "logistic_regression": {
            "prediction": str(lr_pred),
            "confidence": round(lr_conf, 4),
            "ai_probability": round(lr_ai_p, 4),
        },
        "naive_bayes": {
            "prediction": str(nb_pred),
            "confidence": round(nb_conf, 4),
            "ai_probability": round(nb_ai_p, 4),
        },
        "decision": decision,
        "linguistic_features": features,
    })
# ─── Serve Frontend ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# ─── HTML Frontend ───────────────────────────────────────────────────────────
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TruthLens — AI vs Human Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #060910;
    --surface:  #0d1117;
    --card:     #111827;
    --border:   #1f2937;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --green:    #10b981;
    --yellow:   #f59e0b;
    --red:      #ef4444;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --glow:     0 0 40px rgba(0,229,255,0.12);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── GRID BG ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
  }

  /* ── HEADER ── */
  header {
    position: relative; z-index: 10;
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 2.5rem;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: rgba(6,9,16,0.85);
  }
  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .logo span { opacity: 0.5; -webkit-text-fill-color: var(--muted); font-weight: 400; font-size: 0.85rem; margin-left: 6px; font-family: 'DM Mono'; }
  .badge {
    font-size: 0.7rem; padding: 4px 10px;
    border: 1px solid var(--accent); color: var(--accent);
    border-radius: 100px; letter-spacing: 0.1em;
  }

  /* ── MAIN LAYOUT ── */
  main {
    position: relative; z-index: 1;
    max-width: 960px; margin: 0 auto;
    padding: 3rem 1.5rem 4rem;
  }

  /* ── HERO ── */
  .hero { text-align: center; margin-bottom: 3rem; }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800; line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 0.8rem;
  }
  .hero h1 em {
    font-style: normal;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: var(--muted); font-size: 0.9rem; line-height: 1.7; max-width: 480px; margin: 0 auto; }

  /* ── INPUT CARD ── */
  .input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
  }
  .input-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem;
  }
  .input-label { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
  .char-counter { font-size: 0.72rem; color: var(--muted); }
  .char-counter span { color: var(--accent); }

  textarea {
    width: 100%; min-height: 160px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem; line-height: 1.7;
    padding: 1rem 1.1rem;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--muted); }

  /* ── CONTROLS ── */
  .controls { display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }

  .btn-analyze {
    flex: 1; min-width: 180px;
    background: linear-gradient(135deg, var(--accent), #0099bb);
    color: #000; font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 0.9rem;
    padding: 0.85rem 1.5rem;
    border: none; border-radius: 10px;
    cursor: pointer; letter-spacing: 0.02em;
    transition: transform 0.15s, opacity 0.15s;
    position: relative; overflow: hidden;
  }
  .btn-analyze:hover { transform: translateY(-1px); opacity: 0.92; }
  .btn-analyze:active { transform: translateY(0); }
  .btn-analyze:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  .btn-clear {
    padding: 0.85rem 1.2rem;
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); border-radius: 10px;
    font-family: 'DM Mono'; font-size: 0.85rem;
    cursor: pointer; transition: border-color 0.2s, color 0.2s;
  }
  .btn-clear:hover { border-color: var(--red); color: var(--red); }

  /* ── LOADER ── */
  .loader {
    display: none; text-align: center; padding: 2rem;
    color: var(--muted); font-size: 0.82rem;
  }
  .loader.active { display: block; }
  .spinner {
    width: 32px; height: 32px; margin: 0 auto 1rem;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── RESULTS ── */
  #results { display: none; }
  #results.show { display: block; animation: fadeUp 0.4s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── VERDICT BANNER ── */
  .verdict-banner {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 1.5rem;
    border: 1px solid;
    position: relative; overflow: hidden;
  }
  .verdict-banner::before {
    content: ''; position: absolute; inset: 0;
    background: currentColor; opacity: 0.06;
  }
  .verdict-banner.green { color: var(--green); border-color: rgba(16,185,129,0.3); }
  .verdict-banner.yellow { color: var(--yellow); border-color: rgba(245,158,11,0.3); }
  .verdict-banner.red { color: var(--red); border-color: rgba(239,68,68,0.3); }

  .verdict-icon { font-size: 2.4rem; flex-shrink: 0; }
  .verdict-text h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: var(--text); margin-bottom: 0.2rem;
  }
  .verdict-text p { font-size: 0.8rem; color: var(--muted); }

  .risk-badge {
    margin-left: auto; flex-shrink: 0;
    text-align: center;
  }
  .risk-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    line-height: 1;
  }
  .risk-lbl { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }

  /* ── GRID ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  @media(max-width: 600px){ .grid-2 { grid-template-columns: 1fr; } }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem; }
  @media(max-width: 640px){ .grid-3 { grid-template-columns: 1fr 1fr; } }

  /* ── STAT CARD ── */
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--accent); }
  .stat-card .sc-label {
    font-size: 0.68rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .stat-card .sc-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: var(--text);
  }
  .stat-card .sc-sub {
    font-size: 0.72rem; color: var(--muted); margin-top: 2px;
  }

  /* ── MODEL CARD ── */
  .model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
  }
  .model-card .mc-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem;
  }
  .mc-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 0.9rem;
  }
  .mc-pred {
    font-size: 0.72rem; padding: 3px 10px;
    border-radius: 100px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .mc-pred.ai { background: rgba(239,68,68,0.15); color: var(--red); }
  .mc-pred.human { background: rgba(16,185,129,0.15); color: var(--green); }

  /* ── PROGRESS BAR ── */
  .prog-wrap { margin-bottom: 0.7rem; }
  .prog-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-bottom: 4px; }
  .prog-bar { height: 6px; background: var(--border); border-radius: 100px; overflow: hidden; }
  .prog-fill {
    height: 100%; border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
  }
  .fill-ai { background: linear-gradient(90deg, var(--red), #ff6b6b); }
  .fill-human { background: linear-gradient(90deg, var(--green), #34d399); }
  .fill-conf { background: linear-gradient(90deg, var(--accent), var(--accent2)); }

  /* ── AGREEMENT PILL ── */
  .agreement {
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.78rem;
  }
  .agree-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0;
  }
  .agree-dot.yes { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .agree-dot.no  { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }

  /* ── FEATURE TABLE ── */
  .feat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
  }
  .feat-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1rem;
  }
  .feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.8rem;
  }
  .feat-item {
    display: flex; justify-content: space-between;
    align-items: center;
    background: var(--surface);
    border-radius: 8px; padding: 0.55rem 0.8rem;
    font-size: 0.78rem;
  }
  .feat-item .fi-name { color: var(--muted); }
  .feat-item .fi-val { color: var(--accent); font-weight: 500; }

  /* ── INTERPRETATION ── */
  .interp-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(0,229,255,0.04));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-size: 0.83rem;
    line-height: 1.75;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .interp-card strong { color: var(--text); }

  /* ── SECTION TITLE ── */
  .sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 8px;
  }
  .sec-title::after {
    content: ''; flex: 1; height: 1px;
    background: var(--border);
  }

  /* ── ENSEMBLE BAR ── */
  .ensemble-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem; margin-bottom: 1rem;
  }
  .ens-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; margin-bottom: 8px;
  }
  .ens-label span:first-child { color: var(--muted); }
  .ens-label span:last-child { color: var(--text); font-family: 'Syne'; font-weight: 700; }

  /* ── ERROR ── */
  .error-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: var(--red);
    border-radius: 10px; padding: 0.8rem 1rem;
    font-size: 0.82rem; margin-bottom: 1rem;
    display: none;
  }
  .error-box.show { display: block; }

  footer {
    text-align: center; padding: 2rem;
    font-size: 0.72rem; color: var(--muted);
    border-top: 1px solid var(--border);
    position: relative; z-index: 1;
  }
</style>
</head>
<body>

<header>
  <div class="logo">TruthLens <span>v2.0</span></div>
  <div class="badge">AI DETECTION ENGINE</div>
</header>

<main>
  <!-- HERO -->
  <div class="hero">
    <h1>Is it <em>Human</em> or <em>AI?</em></h1>
    <p>Dual-model ensemble analysis with linguistic fingerprinting. Paste any text to get a confidence-scored verdict.</p>
  </div>

  <!-- INPUT -->
  <div class="input-card">
    <div class="input-header">
      <span class="input-label">// Input Text</span>
      <span class="char-counter"><span id="wc">0</span> words · <span id="cc">0</span> chars</span>
    </div>
    <textarea id="textInput" placeholder="Paste or type your text here — minimum 3 words required for analysis..."></textarea>
    <div class="controls">
      <button class="btn-analyze" id="analyzeBtn" onclick="analyze()">⚡ Analyze Text</button>
      <button class="btn-clear" onclick="clearAll()">✕ Clear</button>
    </div>
  </div>

  <div class="error-box" id="errBox"></div>
  <div class="loader" id="loader">
    <div class="spinner"></div>
    Running dual-model inference...
  </div>

  <!-- RESULTS -->
  <div id="results">

    <!-- VERDICT -->
    <div class="verdict-banner" id="verdictBanner">
      <div class="verdict-icon" id="verdictIcon"></div>
      <div class="verdict-text">
        <h2 id="verdictTitle"></h2>
        <p id="verdictSub"></p>
      </div>
      <div class="risk-badge">
        <div class="risk-val" id="riskVal"></div>
        <div class="risk-lbl">Risk Score</div>
      </div>
    </div>

    <!-- ENSEMBLE BAR -->
    <div class="ensemble-wrap">
      <div class="ens-label">
        <span>Ensemble Confidence (55% LR · 45% NB weighted)</span>
        <span id="ensConf"></span>
      </div>
      <div class="prog-bar"><div class="prog-fill fill-conf" id="ensBar" style="width:0%"></div></div>
    </div>

    <!-- MODEL SECTION -->
    <div class="sec-title">Model Breakdown</div>
    <div class="agreement" id="agreementRow">
      <div class="agree-dot" id="agreeDot"></div>
      <span id="agreeText"></span>
    </div>
    <div class="grid-2">
      <!-- LR CARD -->
      <div class="model-card">
        <div class="mc-head">
          <span class="mc-name">Logistic Regression</span>
          <span class="mc-pred" id="lrPredBadge"></span>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>AI probability</span><span id="lrAiP"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-ai" id="lrAiBar" style="width:0%"></div></div>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>Human probability</span><span id="lrHumanP"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-human" id="lrHumanBar" style="width:0%"></div></div>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>Confidence</span><span id="lrConf"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-conf" id="lrConfBar" style="width:0%"></div></div>
        </div>
      </div>

      <!-- NB CARD -->
      <div class="model-card">
        <div class="mc-head">
          <span class="mc-name">Naïve Bayes</span>
          <span class="mc-pred" id="nbPredBadge"></span>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>AI probability</span><span id="nbAiP"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-ai" id="nbAiBar" style="width:0%"></div></div>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>Human probability</span><span id="nbHumanP"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-human" id="nbHumanBar" style="width:0%"></div></div>
        </div>
        <div class="prog-wrap">
          <div class="prog-labels"><span>Confidence</span><span id="nbConf"></span></div>
          <div class="prog-bar"><div class="prog-fill fill-conf" id="nbConfBar" style="width:0%"></div></div>
        </div>
      </div>
    </div>

    <!-- LINGUISTIC FEATURES -->
    <div class="sec-title">Linguistic Fingerprint</div>
    <div class="feat-card">
      <div class="feat-grid" id="featGrid"></div>
    </div>

    <!-- INTERPRETATION -->
    <div class="sec-title">AI Interpretation</div>
    <div class="interp-card" id="interpBox"></div>

  </div><!-- /results -->
</main>

<footer>
  TruthLens · Dual-Model Ensemble (LR + NB) · Bigram CountVectorizer · Linguistic Analysis<br>
  Built for KCE Hackathon 2028 Batch · No external AI APIs used
</footer>

<script>
const $ = id => document.getElementById(id);
const pct = v => (v * 100).toFixed(1) + '%';
const fmt = v => (v * 100).toFixed(1) + '%';

// Word/char counter
$('textInput').addEventListener('input', function() {
  const t = this.value.trim();
  $('wc').textContent = t ? t.split(/\s+/).length : 0;
  $('cc').textContent = this.value.length;
});

function clearAll() {
  $('textInput').value = '';
  $('wc').textContent = '0';
  $('cc').textContent = '0';
  $('results').classList.remove('show');
  $('errBox').classList.remove('show');
}

function showError(msg) {
  $('errBox').textContent = msg;
  $('errBox').classList.add('show');
  $('loader').classList.remove('active');
  $('analyzeBtn').disabled = false;
}

async function analyze() {
  const text = $('textInput').value.trim();
  $('errBox').classList.remove('show');
  $('results').classList.remove('show');

  if (!text) { showError('Please enter some text.'); return; }

  $('analyzeBtn').disabled = true;
  $('loader').classList.add('active');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const d = await res.json();

    if (d.error) { showError(d.error); return; }

    renderResults(d);
  } catch(e) {
    showError('Connection error. Make sure the Flask server is running.');
  } finally {
    $('loader').classList.remove('active');
    $('analyzeBtn').disabled = false;
  }
}

function renderResults(d) {
  const { logistic_regression: lr, naive_bayes: nb, decision: dec, linguistic_features: lf } = d;

  // VERDICT BANNER
  const banner = $('verdictBanner');
  banner.className = 'verdict-banner ' + dec.verdict_color;
  $('verdictIcon').textContent = dec.verdict_icon;
  $('verdictTitle').textContent = dec.verdict;
  $('verdictSub').textContent =
    dec.verdict === 'Acceptable'         ? 'High confidence — this text appears to be human-written.' :
    dec.verdict === 'Needs Review'       ? 'Moderate confidence — manual review recommended.' :
                                           'Low confidence — this text shows strong AI-generation patterns.';
  $('riskVal').textContent = dec.risk_score + '%';
  $('riskVal').style.color = dec.verdict_color === 'green' ? 'var(--green)' :
                              dec.verdict_color === 'yellow' ? 'var(--yellow)' : 'var(--red)';

  // ENSEMBLE BAR
  const ep = (dec.ensemble_confidence * 100).toFixed(1);
  $('ensConf').textContent = ep + '% confidence';
  setTimeout(() => $('ensBar').style.width = ep + '%', 50);

  // AGREEMENT
  const agree = dec.model_agreement;
  $('agreeDot').className = 'agree-dot ' + (agree ? 'yes' : 'no');
  $('agreeText').textContent = agree
    ? 'Both models agree — prediction is more reliable.'
    : 'Models disagree — interpret result with caution.';

  // LR CARD
  const lrAi = (lr.ai_probability * 100).toFixed(1);
  const lrHuman = (100 - lr.ai_probability * 100).toFixed(1);
  $('lrPredBadge').textContent = lr.prediction;
  $('lrPredBadge').className = 'mc-pred ' + lr.prediction;
  $('lrAiP').textContent = lrAi + '%';
  $('lrHumanP').textContent = lrHuman + '%';
  $('lrConf').textContent = (lr.confidence * 100).toFixed(1) + '%';
  setTimeout(() => {
    $('lrAiBar').style.width = lrAi + '%';
    $('lrHumanBar').style.width = lrHuman + '%';
    $('lrConfBar').style.width = (lr.confidence * 100).toFixed(1) + '%';
  }, 80);

  // NB CARD
  const nbAi = (nb.ai_probability * 100).toFixed(1);
  const nbHuman = (100 - nb.ai_probability * 100).toFixed(1);
  $('nbPredBadge').textContent = nb.prediction;
  $('nbPredBadge').className = 'mc-pred ' + nb.prediction;
  $('nbAiP').textContent = nbAi + '%';
  $('nbHumanP').textContent = nbHuman + '%';
  $('nbConf').textContent = (nb.confidence * 100).toFixed(1) + '%';
  setTimeout(() => {
    $('nbAiBar').style.width = nbAi + '%';
    $('nbHumanBar').style.width = nbHuman + '%';
    $('nbConfBar').style.width = (nb.confidence * 100).toFixed(1) + '%';
  }, 120);

  // LINGUISTIC FEATURES
  const featMap = {
    'word_count':          ['Words',             lf.word_count],
    'char_count':          ['Characters',        lf.char_count],
    'sentence_count':      ['Sentences',         lf.sentence_count],
    'avg_word_length':     ['Avg Word Len',      lf.avg_word_length],
    'avg_sentence_length': ['Avg Sent Len',      lf.avg_sentence_length],
    'vocab_richness':      ['Vocab Richness',    lf.vocab_richness],
    'punctuation_density': ['Punct Density',     lf.punctuation_density],
    'burstiness':          ['Burstiness',        lf.burstiness],
  };
  $('featGrid').innerHTML = Object.entries(featMap).map(([k, [label, val]]) =>
    `<div class="feat-item"><span class="fi-name">${label}</span><span class="fi-val">${val}</span></div>`
  ).join('');

  // INTERPRETATION
  const aiChance = (lr.ai_probability * 100).toFixed(0);
  const vocabNote = lf.vocab_richness > 0.8
    ? 'High vocabulary richness suggests diverse, non-repetitive word choices — a human trait.'
    : lf.vocab_richness < 0.5
    ? 'Low vocabulary richness may indicate repetitive AI phrasing patterns.'
    : 'Moderate vocabulary richness observed.';
  const burstNote = lf.burstiness > 3
    ? 'High sentence-length variation (burstiness) is characteristic of human writing.'
    : 'Low burstiness — sentences are uniformly structured, which AI models tend to produce.';

  $('interpBox').innerHTML = `
    <strong>Analysis Summary:</strong> The ensemble model assigns a <strong>${aiChance}% AI probability</strong>
    to this text. ${vocabNote} ${burstNote}
    ${agree ? 'Both models reached the same conclusion, reinforcing the result.' :
              'The two models disagree, which may indicate ambiguous or mixed-origin content.'}
    Final decision threshold applied: <strong>${dec.verdict}</strong>.
  `;

  $('results').classList.add('show');
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Enter key shortcut
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 TruthLens running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
