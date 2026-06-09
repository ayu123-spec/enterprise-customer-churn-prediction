"""FastAPI prediction service for the Enterprise Customer Churn model.

Serves a styled prediction front page at /, a JSON prediction endpoint at
/predict, and a health check at /health. Incoming customers are transformed
exactly the way the training pipeline did (via the shared engineer() function).

Run from the project root (with venv active):
    uvicorn src.api.app:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.features.feature_engineering import engineer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODELS_DIR / "xgboost.joblib")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class Customer(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 5
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 79.0
    TotalCharges: float = 395.0


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str
    recommendation: str


def _to_features(customer: Customer) -> pd.DataFrame:
    df = pd.DataFrame([customer.model_dump()])
    df = engineer(df)
    X = pd.get_dummies(df, dtype=int)
    return X.reindex(columns=feature_columns, fill_value=0)


def _recommend(c: Customer, proba: float) -> str:
    if proba < 0.4:
        return "Low risk - maintain standard engagement."
    tips = []
    if c.Contract == "Month-to-month":
        tips.append("offer a discounted 1- or 2-year contract")
    if c.tenure < 12:
        tips.append("prioritise onboarding and early engagement")
    if c.MonthlyCharges > 70:
        tips.append("review the plan for a better-value bundle")
    if c.TechSupport == "No" and c.InternetService != "No":
        tips.append("add tech support / online security")
    return "High risk - " + "; ".join(tips) if tips else "Elevated risk - reach out proactively."


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "n_features": len(feature_columns)}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = _to_features(customer)
    proba = float(model.predict_proba(X)[0, 1])
    level = "high" if proba >= 0.6 else "medium" if proba >= 0.4 else "low"
    return Prediction(
        churn_probability=round(proba, 4),
        will_churn=proba >= 0.5,
        risk_level=level,
        recommendation=_recommend(customer, proba),
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Risk Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#16202B; --muted:#65727F; --line:#E3E8EF; --bg:#EDF0F4; --card:#FFFFFF;
    --brand:#2F4A7C; --low:#179A6B; --med:#DD9B2E; --high:#D44A38;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--ink);
    line-height:1.5;-webkit-font-smoothing:antialiased;padding:32px 20px}
  .wrap{max-width:980px;margin:0 auto}
  header{margin-bottom:28px}
  .eyebrow{font-family:'Space Grotesk';font-weight:600;font-size:13px;letter-spacing:.14em;
    text-transform:uppercase;color:var(--brand)}
  h1{font-family:'Space Grotesk';font-weight:700;font-size:34px;letter-spacing:-.02em;margin:6px 0 4px}
  header p{color:var(--muted);font-size:15px}
  .grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}
  @media(max-width:760px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px}
  .card h2{font-family:'Space Grotesk';font-size:15px;font-weight:600;margin-bottom:18px;
    letter-spacing:.01em}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .field{margin-bottom:14px}
  label{display:block;font-size:12.5px;font-weight:500;color:var(--muted);margin-bottom:6px}
  input,select{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:9px;
    font-family:inherit;font-size:14px;color:var(--ink);background:#FBFCFD;outline:none}
  input:focus,select:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(47,74,124,.12)}
  button{width:100%;margin-top:8px;padding:13px;border:none;border-radius:10px;
    background:var(--brand);color:#fff;font-family:'Space Grotesk';font-weight:600;font-size:15px;
    cursor:pointer;transition:filter .15s}
  button:hover{filter:brightness(1.08)}
  .result{display:flex;flex-direction:column;justify-content:center;min-height:100%}
  .empty{color:var(--muted);font-size:14.5px;text-align:center;padding:30px 10px}
  .empty svg{width:40px;height:40px;stroke:var(--line);margin-bottom:12px}
  .score{font-family:'Space Grotesk';font-weight:700;font-size:62px;line-height:1;letter-spacing:-.03em}
  .band{display:inline-block;margin-top:10px;padding:4px 12px;border-radius:999px;font-size:12.5px;
    font-weight:600;letter-spacing:.04em;text-transform:uppercase}
  .meter{height:10px;border-radius:999px;margin:22px 0 6px;position:relative;
    background:linear-gradient(90deg,var(--low) 0%,var(--med) 55%,var(--high) 100%)}
  .needle{position:absolute;top:-5px;width:4px;height:20px;border-radius:2px;background:var(--ink);
    transform:translateX(-2px);transition:left .5s cubic-bezier(.2,.8,.2,1)}
  .scale{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
  .rec{margin-top:20px;padding:14px 16px;background:#F6F8FB;border:1px solid var(--line);
    border-radius:11px;font-size:14px}
  .rec b{font-family:'Space Grotesk';display:block;margin-bottom:4px;font-size:12.5px;
    text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
  .hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Churn intelligence</div>
    <h1>Will this customer stay?</h1>
    <p>Enter a customer's plan and account details to estimate their churn risk and get a retention play.</p>
  </header>
  <div class="grid">
    <div class="card">
      <h2>Customer details</h2>
      <div class="row">
        <div class="field"><label>Tenure (months)</label><input id="tenure" type="number" value="3" min="0" max="72"></div>
        <div class="field"><label>Senior citizen</label><select id="senior"><option value="0">No</option><option value="1">Yes</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Monthly charges</label><input id="mc" type="number" value="89" step="0.05"></div>
        <div class="field"><label>Total charges</label><input id="tc" type="number" value="270" step="0.05"></div>
      </div>
      <div class="field"><label>Contract</label><select id="contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
      <div class="row">
        <div class="field"><label>Internet service</label><select id="internet"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
        <div class="field"><label>Payment method</label><select id="payment"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Tech support</label><select id="techsupport"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
        <div class="field"><label>Online security</label><select id="security"><option>No</option><option>Yes</option><option>No internet service</option></select></div>
      </div>
      <div class="field"><label>Paperless billing</label><select id="paperless"><option>Yes</option><option>No</option></select></div>
      <button onclick="check()">Check churn risk</button>
    </div>
    <div class="card result">
      <div id="empty" class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.6"><path d="M3 12h4l3 8 4-16 3 8h4"/></svg>
        <div>Fill in the details and run the check to see this customer's churn risk.</div>
      </div>
      <div id="out" class="hidden">
        <div class="score" id="score">--</div>
        <span class="band" id="band"></span>
        <div class="meter"><div class="needle" id="needle" style="left:0%"></div></div>
        <div class="scale"><span>0%</span><span>Likely to churn</span><span>100%</span></div>
        <div class="rec"><b>Retention play</b><span id="rec"></span></div>
      </div>
    </div>
  </div>
</div>
<script>
  const v = id => document.getElementById(id).value;
  const colors = {low:'#179A6B', medium:'#DD9B2E', high:'#D44A38'};
  async function check(){
    const payload = {
      tenure:+v('tenure'), SeniorCitizen:+v('senior'),
      MonthlyCharges:+v('mc'), TotalCharges:+v('tc'),
      Contract:v('contract'), InternetService:v('internet'),
      PaymentMethod:v('payment'), TechSupport:v('techsupport'),
      OnlineSecurity:v('security'), PaperlessBilling:v('paperless')
    };
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const d = await r.json();
    const pct = (d.churn_probability*100).toFixed(1);
    const c = colors[d.risk_level];
    document.getElementById('empty').classList.add('hidden');
    document.getElementById('out').classList.remove('hidden');
    const score = document.getElementById('score');
    score.textContent = pct + '%'; score.style.color = c;
    const band = document.getElementById('band');
    band.textContent = d.risk_level + ' risk'; band.style.color = c; band.style.background = c+'1f';
    document.getElementById('needle').style.left = pct + '%';
    document.getElementById('rec').textContent = d.recommendation;
  }
</script>
</body>
</html>"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    