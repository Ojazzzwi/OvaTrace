# OvaTrace: PCOS Early Risk Detection Dashboard

Flask + Chart.js dashboard built on the Kaggle PCOS dataset (541 patients).
Ten tabs: Introduction, Data Wrangling, Preprocessing, EDA, Hypothesis Testing,
PCA/LDA, Model Performance, Hidden Insights, Risk Prediction, Lifestyle Plan.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis.py     # optional: regenerates artifacts/ from data/*.xlsx (already included)
python app.py          # http://localhost:5000
```

## Structure
| File | Purpose |
|---|---|
| `analysis.py` | Cleaning, stats, PCA/LDA, Random Forest, lifestyle stats. Writes `artifacts/results.json` and `artifacts/early_rf.joblib` |
| `app.py` | Flask server: `/` dashboard, `/api/results`, `/api/predict`, `/health` |
| `templates/index.html` | Front end (vanilla JS + Chart.js from cdnjs) |
| `data/` | Source Excel file |
| `render.yaml` | One-click Render deploy |

## API
`POST /api/predict` with JSON containing the 12 features listed in `artifacts/results.json` → `calc.f`
(Age (yrs), BMI, Waist:Hip Ratio, Cycle(R/I) 0/1, Cycle length(days), five symptom flags, Fast food, Reg.Exercise).
Returns `{"rf_probability": 0.68, "risk_level": "elevated"}`. Out-of-range or missing values return HTTP 400.

## Deploy on Render
Push to GitHub → New Web Service → it reads `render.yaml`
(build: `pip install -r requirements.txt && python analysis.py`, start: `gunicorn app:app`).

## Notes
- Cycle(R/I) is coded 2 = regular, 4 = irregular in the raw data → recoded to 0/1.
- Charts load Chart.js from a CDN; for offline use, download it and change the `<script src>` in `templates/index.html`.
- Educational project, not a medical device. Single-source data, no external validation.
