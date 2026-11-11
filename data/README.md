# Dataset placement

This project uses the same CICIDS-style `MachineLearningCVE` network-flow CSV dataset as Team 10's supplied notebook.

1. Obtain the public CICIDS2017 MachineLearningCVE CSV files from the official Canadian Institute for Cybersecurity dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
2. Put the CSV files in this `data/` folder.
3. Do not commit large CSV files to Git. The end-term guidelines prohibit committing datasets larger than 50 MB; keep them local or use a documented download script if redistribution is permitted.
4. `train.py` reads CSV files in chunks and requires a `Label` column, at least 1,000 rows and at least 5 numeric input features.

Target mapping: `BENIGN` → 0; every other CICIDS label → 1 (ATTACK). This intentionally defines a binary NIDS task.
