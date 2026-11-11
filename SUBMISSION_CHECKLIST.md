# End-Term submission checklist

## Before evaluation
- [ ] Download/place the real CICIDS2017 MachineLearningCVE CSV files in `data/`.
- [ ] Create venv and install `requirements.txt`.
- [ ] Run `python train.py --n_estimators 100 --max_depth 12` on the real dataset.
- [ ] Run `python scripts/export_contract.py` to write the complete feature table into README.md and DATA_CONTRACT.md.
- [ ] Open MLflow UI and confirm at least 3 runs; save a genuine screenshot.
- [ ] Run `pytest -v --tb=short`; save passing output.
- [ ] Run `uvicorn app:app --reload --port 8000`; test `/`, `/health`, `/docs`, valid `/predict`, and invalid input 422.
- [ ] Build and run Docker; record image size.
- [ ] Push code to a shared GitHub repository; ensure each member contributes commits.
- [ ] Confirm latest GitHub Actions workflow is green.
- [ ] Deploy the Docker service to Render/Railway/AWS; verify live URL and restart behaviour.
- [ ] Capture UI, docs, MLflow, pytest, CI and deployment screenshots.
- [ ] Add real repository/live URLs and screenshots to README, report and PPT.
- [ ] Fill section, individual contributions and file naming required by instructor.

## Important
The CI workflow uses a documented synthetic fixture only to verify that training, MLflow, API and tests run on GitHub without storing a large dataset. Final reported model metrics and deployment must be produced from the actual CICIDS2017 dataset.
