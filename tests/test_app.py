from fastapi.testclient import TestClient
from app import app, METADATA, FEATURES

client = TestClient(app)

def sample_payload():
    # Use a real valid data point retained in the training metadata for boundary-safe input.
    pair = METADATA["domain_pair"]["weaker_candidate"]
    return {"features": {k: float(v) for k,v in pair.items()}}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Network Intrusion Detection System" in response.text

def test_prediction_returns_valid_class_and_tips():
    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"BENIGN", "ATTACK"}
    assert 0.0 <= body["attack_probability"] <= 1.0
    assert 2 <= len(body["tips"]) <= 3

def test_domain_logic_attack_example_scores_above_benign_example():
    pair = METADATA["domain_pair"]
    def score(row):
        payload = {"features": row}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        return response.json()["attack_probability"]
    weaker = score(pair["weaker_candidate"])
    stronger = score(pair["stronger_candidate"])
    assert stronger > weaker, f"Expected attack-like sample {stronger} > benign-like sample {weaker}"

def test_out_of_range_feature_returns_422():
    payload = sample_payload()
    feature = next(iter(FEATURES))
    payload["features"][feature] = float(FEATURES[feature]["max"]) + 1e6
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_missing_feature_returns_422():
    payload = sample_payload()
    payload["features"].pop(next(iter(FEATURES)))
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "nids_predictions_total" in response.text

def test_string_feature_value_returns_422():
    payload = sample_payload()
    payload["features"][next(iter(FEATURES))] = "not-a-number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
