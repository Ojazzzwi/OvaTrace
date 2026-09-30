"""OvaTrace Flask app.  Run:  python app.py   ->  http://localhost:5000"""
import json, os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
if not (ART / "results.json").exists() or not (ART / "early_rf.joblib").exists():
    import analysis  # noqa: F401  (regenerates artifacts from data/*.xlsx)

RESULTS = (ART / "results.json").read_text(encoding="utf-8")
BUNDLE = joblib.load(ART / "early_rf.joblib")
MODEL, FEATURES = BUNDLE["model"], BUNDLE["features"]
PAGE = (ROOT / "templates" / "index.html").read_text(encoding="utf-8").replace("__DATA__", RESULTS)

# plausible input ranges, used to validate API requests
RANGES = {"Age (yrs)": (10, 70), "BMI": (10, 60), "Waist:Hip Ratio": (0.4, 1.5),
          "Cycle length(days)": (0, 30)}

app = Flask(__name__)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/results")
def results():
    return Response(RESULTS, mimetype="application/json")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    row = {}
    for f in FEATURES:
        try:
            v = float(body[f])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"missing or invalid feature: {f}"), 400
        lo, hi = RANGES.get(f, (0, 1))
        if not lo <= v <= hi:
            return jsonify(error=f"{f} out of range ({lo}-{hi})"), 400
        row[f] = v
    p = float(MODEL.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    level = ("low" if p < .2 else "moderate" if p < .5 else "elevated" if p < .8 else "high")
    return jsonify(rf_probability=round(p, 4), risk_level=level, features=FEATURES)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
