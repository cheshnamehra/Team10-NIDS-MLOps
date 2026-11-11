"""FastAPI inference service. Model is loaded once at process startup."""
from __future__ import annotations
import json, logging, os, time
from pathlib import Path
from typing import Dict, Annotated
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, BeforeValidator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.joblib"))
META_PATH = Path(os.getenv("META_PATH", ROOT / "model" / "metadata.json"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("team10-nids")

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("Model files missing. Run `python train.py` before starting the API.")
MODEL = joblib.load(MODEL_PATH)  # one-time startup load
METADATA = json.loads(META_PATH.read_text())
FEATURES = METADATA["features"]

# Build an explicit Pydantic feature schema with a numeric lower/upper bound for every field.
def _numeric_json_value(value):
    # Accept JSON numbers (integer or decimal), but reject numeric strings and booleans.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Input must be a JSON number.")
    return float(value)

feature_fields = {
    name: (
        Annotated[float, BeforeValidator(_numeric_json_value)],
        Field(..., ge=spec["min"], le=spec["max"], allow_inf_nan=False, description=spec["why"])
    )
    for name, spec in FEATURES.items()
}
FeatureValues = create_model("FeatureValues", __config__=ConfigDict(extra="forbid"), **feature_fields)

class PredictionRequest(BaseModel):
    features: FeatureValues

PREDICTIONS = Counter("nids_predictions_total", "Total NIDS predictions", ["label"])
PREDICTION_ERRORS = Counter("nids_prediction_errors_total", "Rejected or failed prediction requests")
PREDICTION_LATENCY = Histogram("nids_prediction_latency_seconds", "NIDS prediction latency")

app = FastAPI(
    title="Team 10 — Network Intrusion Detection API",
    description="CICIDS-style binary classifier: 0 = BENIGN, 1 = ATTACK.",
    version="1.0.0"
)

def user_tips(features: Dict[str, float], label: str):
    tips = []
    if label == "ATTACK":
        tips.append("Treat this flow as suspicious and review the source/destination endpoints.")
    else:
        tips.append("No attack was flagged; continue normal network monitoring.")
    duration = next((v for k,v in features.items() if k.strip().lower()=="flow duration"), None)
    if duration is not None and duration > 1_000_000:
        tips.append("This flow has a long duration; compare it with the host's normal traffic baseline.")
    else:
        tips.append("Check unusual packet/byte volume against the normal baseline for this network segment.")
    tips.append("Correlate this result with firewall, IDS and endpoint logs before taking action.")
    return tips[:3]

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = ROOT / "static" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.get("/health")
def health():
    return {"status": "ok", "model": METADATA["best_algorithm"]}

@app.post("/predict")
def predict(request: PredictionRequest):
    features = request.features.model_dump()
    start = time.perf_counter()
    try:
        frame = __import__("pandas").DataFrame([[features[name] for name in METADATA["feature_names"]]],
                                               columns=METADATA["feature_names"])
        prediction = int(MODEL.predict(frame)[0])
        probability = float(MODEL.predict_proba(frame)[0][1])
        label = "ATTACK" if prediction == 1 else "BENIGN"
        PREDICTIONS.labels(label=label).inc()
        response = {
            "prediction": prediction, "label": label,
            "attack_probability": round(probability, 6),
            "tips": user_tips(features, label)
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(), "input": features, "prediction": prediction,
                "attack_probability": probability, "latency_seconds": time.perf_counter()-start
            }) + "\n")
        return response
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; see service logs.") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter()-start)

@app.get("/metrics")
def metrics():
    """Prometheus-compatible service metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
