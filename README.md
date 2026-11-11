# Team 10 — Machine Learning-Based Network Intrusion Detection System

**Course:** CSL236 — Introduction to AI and Machine Learning, Semester 5  
**Team:** Cheshna Mehra (24csu044), Faguni (24csu290), Riya Fulambarkar (24csu174)

## 1. Problem statement
Classify a network flow as `BENIGN` (0) or `ATTACK` (1) using numerical flow statistics from CICIDS2017. The goal is to provide a reproducible binary network-intrusion classification service, not an automated blocking decision.

## 2. Dataset and source
CICIDS2017 MachineLearningCVE network-flow CSV files, Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html. Dataset placement and handling instructions are in `data/README.md`. Training samples at most 10,000 records per source label for a manageable reproducible working set, then removes duplicates. Dataset files are not committed.

## 3. Data contract
The API's complete feature contract is generated from the training split and saved in `model/metadata.json`. Each numeric feature has an explicit lower and upper bound, type, and rationale. Pydantic builds a constrained request schema at startup; missing, unknown, nonnumeric, NaN/Infinity and out-of-range fields are rejected with HTTP 422. The full table is generated after training by `python scripts/export_contract.py` to create `DATA_CONTRACT.md` (or inspect `model/metadata.json`).

| Feature | Type | Allowed range / values | Why it matters |
|---|---|---|---|
<!-- DATA_CONTRACT_START -->
Run `python scripts/export_contract.py` after training to replace this block with the complete generated table (one row for every numeric feature plus target). The API enforces the same generated bounds using Pydantic.
<!-- DATA_CONTRACT_END -->

Bounds are learned from the training partition only. They are an input-domain guardrail, not a claim that values outside the observed training range are impossible in real networks. Re-train and regenerate the schema when the data contract changes.

## 4. Preprocessing and model experiments
The training pipeline performs:
- stratified 80:20 train/test split before fitting transformations;
- replacement of infinite values with missing values and median imputation;
- training-fitted `log1p` transformation for non-negative, strongly skewed features;
- training-derived quartile binning of `Flow Duration` where present;
- standard scaling, zero-variance feature removal and PCA retaining 95% variance;
- comparison of Logistic Regression, Random Forest and Extra Trees.

Three configurations/runs are logged to MLflow using the local SQLite backend. The winning run is also registered in the MLflow Model Registry as `Team10NIDS` with the `champion` alias when the registry backend is available. The best model is selected by test F1-score (accuracy breaks ties) and saved as `model/model.joblib`. Run metrics are saved to `training_results.json` and the MLflow experiment is available in `mlflow.db`.

## 5. Setup and run locally
```bash
python3 -m venv venv
# Windows PowerShell: venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
# Put CICIDS CSVs into data/ (see data/README.md)
python train.py --n_estimators 100 --max_depth 12
python scripts/export_contract.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn app:app --reload --port 8000
```
Open `http://127.0.0.1:8000/`, `/docs`, and `/health`.

## 6. API
- `GET /` — browser UI
- `GET /health` — service/model health
- `POST /predict` — validated feature dictionary; returns class, attack probability and 2–3 rule-based tips
- `GET /metrics` — Prometheus-compatible prediction counters, error count and latency histogram

The model is loaded once at application startup, not once per request. Every successful prediction input, output and latency is appended to `logs/predictions.jsonl` for later drift analysis. The log is ignored by Git.

## 7. Tests
```bash
pytest -v --tb=short
```
Tests cover health, UI, prediction output, domain-logic ordering, out-of-range rejection (422), and missing-feature rejection (422).

## 8. Docker
```bash
python train.py
docker build -t team10-nids .
docker run -d -p 8000:8000 --restart unless-stopped team10-nids
```
The serving image uses `python:3.11-slim` and `requirements-serve.txt` only. Record the built image size after running `docker images team10-nids` and update this README for final submission.

## 9. GitHub Actions CI
The workflow `.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It installs dependencies, creates a documented deterministic synthetic fixture for CI smoke-testing only, trains three MLflow experiments and runs pytest. The synthetic fixture is not used for the final reported results. For final model training, use the real CICIDS CSV files in `data/`; do not commit a dataset larger than 50 MB.

## 10. Deployment and monitoring
`render.yaml` is a Render Docker-service starter with `/health` as its health check. Push the trained `model/model.joblib` and `model/metadata.json` with the repository (these are model artifacts, not raw dataset files), configure the service, and verify the public URL. The app logs prediction inputs to JSONL. The service host must be configured to restart the container after failure.

**Live deployment URL:** Pending team deployment.  
**GitHub repository:** Pending team repository creation.

## 11. MLflow results
Run `python train.py` and copy the three rows from `training_results.json` into this table before final submission.

| Run | Algorithm | Hyperparameters | Accuracy | F1 |
|---|---|---|---:|---:|
| 1 | Logistic Regression | `C` from CLI | Generated by training | Generated by training |
| 2 | Random Forest | `n_estimators`, `max_depth` | Generated by training | Generated by training |
| 3 | Extra Trees | `n_estimators`, `max_depth` | Generated by training | Generated by training |

## 12. Workshop connection
The implementation follows the BrainMentors *End-to-End MLOps: Production Deployment* workshop reference architecture: training script, MLflow experiment tracking, serialized model, FastAPI serving, Docker container, GitHub Actions CI and cloud deployment. Project-specific dataset handling, binary target, preprocessing, validation contract, tests and UI tips are Team 10's NIDS implementation.

## 13. Limitations and next steps
- Binary mapping merges all attack families into one class.
- Training-split min/max bounds may reject novel but legitimate traffic; monitor 422 rates and update the contract deliberately.
- Flow distributions can change over time; compare logged-input mean/std with training statistics and retrain when drift is material.
- Model output is a decision-support signal and must be correlated with other security telemetry.
- Next: multiclass classification, threshold calibration, SHAP explanations and automated post-CI deployment.

## 14. Team contribution
Fill this table with the actual contribution and GitHub commits of each member before submission.

| Member | Roll number | Contribution |
|---|---|---|
| Cheshna Mehra | 24csu044 | To be confirmed by team |
| Faguni | 24csu290 | To be confirmed by team |
| Riya Fulambarkar | 24csu174 | To be confirmed by team |

## 15. Evidence checklist
Add genuine screenshots of MLflow with 3+ runs, UI, `/docs`, pytest output, green GitHub Actions run, and live deployment. Do not use mock screenshots or claim a deployment that has not been tested.
