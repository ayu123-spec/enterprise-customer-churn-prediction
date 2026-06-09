"""Generate a standalone executive churn dashboard at reports/dashboard.html.

Reads the cleaned dataset, computes the headline churn metrics and the churn
rate across the key segments, and writes a self-contained HTML page (charts
via Chart.js CDN). Open the file in any browser - no server needed.

Run from the project root (with venv active):
    python src/dashboard/build_dashboard.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "dashboard.html"
TARGET = "Churn"


def compute(df: pd.DataFrame) -> dict:
    def rate_by(col: str) -> dict:
        s = (df.groupby(col)[TARGET].mean() * 100).round(1).sort_values()
        return {"labels": list(s.index), "values": [float(v) for v in s.values]}

    bands = pd.cut(df["tenure"], [0, 12, 24, 48, 72],
                   labels=["0-12m", "12-24m", "24-48m", "48-72m"], include_lowest=True)
    tb = (df.groupby(bands, observed=True)[TARGET].mean() * 100).round(1)

    return {
        "n_customers": int(len(df)),
        "churn_rate": round(float(df[TARGET].mean() * 100), 1),
        "churned": int(df[TARGET].sum()),
        "revenue_at_risk": round(float(df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()), 0),
        "by_contract": rate_by("Contract"),
        "by_internet": rate_by("InternetService"),
        "by_payment": rate_by("PaymentMethod"),
        "by_tenure": {"labels": list(tb.index), "values": [float(v) for v in tb.values]},
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root{--ink:#16202B;--muted:#65727F;--line:#E3E8EF;--bg:#EDF0F4;--card:#FFF;--brand:#2F4A7C}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--ink);padding:32px 20px}
  .wrap{max-width:1080px;margin:0 auto}
  .eyebrow{font-family:'Space Grotesk';font-weight:600;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--brand)}
  h1{font-family:'Space Grotesk';font-weight:700;font-size:32px;letter-spacing:-.02em;margin:6px 0 4px}
  .sub{color:var(--muted);font-size:15px;margin-bottom:26px}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}
  @media(max-width:720px){.kpis{grid-template-columns:repeat(2,1fr)}}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
  .kpi .v{font-family:'Space Grotesk';font-weight:700;font-size:30px;letter-spacing:-.02em}
  .kpi .l{font-size:12.5px;color:var(--muted);margin-top:4px}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:720px){.charts{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
  .card h2{font-family:'Space Grotesk';font-size:14.5px;font-weight:600;margin-bottom:14px}
  .foot{color:var(--muted);font-size:12px;margin-top:20px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Executive overview</div>
  <h1>Customer churn dashboard</h1>
  <div class="sub">Where the business is losing customers, and which segments drive it.</div>
  <div class="kpis">
    <div class="kpi"><div class="v" id="k_customers"></div><div class="l">Total customers</div></div>
    <div class="kpi"><div class="v" id="k_rate" style="color:#D44A38"></div><div class="l">Overall churn rate</div></div>
    <div class="kpi"><div class="v" id="k_churned"></div><div class="l">Customers churned</div></div>
    <div class="kpi"><div class="v" id="k_rev" style="color:#D44A38"></div><div class="l">Monthly revenue at risk</div></div>
  </div>
  <div class="charts">
    <div class="card"><h2>Churn rate by contract</h2><canvas id="c1"></canvas></div>
    <div class="card"><h2>Churn rate by tenure</h2><canvas id="c2"></canvas></div>
    <div class="card"><h2>Churn rate by internet service</h2><canvas id="c3"></canvas></div>
    <div class="card"><h2>Churn rate by payment method</h2><canvas id="c4"></canvas></div>
  </div>
  <div class="foot">Generated from the processed Telco dataset &middot; churn rate shown as % within each segment.</div>
</div>
<script>
  const DATA = __DATA__;
  const fmt = n => n.toLocaleString();
  document.getElementById('k_customers').textContent = fmt(DATA.n_customers);
  document.getElementById('k_rate').textContent = DATA.churn_rate + '%';
  document.getElementById('k_churned').textContent = fmt(DATA.churned);
  document.getElementById('k_rev').textContent = '$' + fmt(DATA.revenue_at_risk);

  const colorFor = v => v >= 35 ? '#D44A38' : v >= 20 ? '#DD9B2E' : '#179A6B';
  function bar(id, d){
    new Chart(document.getElementById(id), {
      type:'bar',
      data:{labels:d.labels, datasets:[{data:d.values,
        backgroundColor:d.values.map(colorFor), borderRadius:6, barThickness:24}]},
      options:{indexAxis:'y', plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>c.parsed.x+'% churn'}}},
        scales:{x:{ticks:{callback:v=>v+'%'},grid:{color:'#EEF1F5'}},y:{grid:{display:false}}},
        responsive:true, maintainAspectRatio:true, aspectRatio:1.8}
    });
  }
  bar('c1', DATA.by_contract);
  bar('c2', DATA.by_tenure);
  bar('c3', DATA.by_internet);
  bar('c4', DATA.by_payment);
</script>
</body>
</html>"""


def run() -> None:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Clean data not found at {CLEAN_PATH}. Run the pipeline first.")
    df = pd.read_csv(CLEAN_PATH)
    data = compute(df)
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", OUTPUT_PATH)
    logger.info("Key metrics: %s", json.dumps({k: v for k, v in data.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    run()
    