"""Generate a documented, deterministic small synthetic NIDS-like fixture for CI only.

This fixture is not used for reported project metrics or the final real-data model.
It lets GitHub Actions exercise the full train -> MLflow -> API -> pytest workflow
without committing the >50 MB public CICIDS2017 dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"data"/"ci_fixture.csv"
rng=np.random.default_rng(42)
n=1400
y=np.array([0]*(n//2)+[1]*(n//2))
rng.shuffle(y)
df=pd.DataFrame({
    "Flow Duration": np.where(y==1,rng.gamma(5,250000,n),rng.gamma(2,90000,n)),
    "Total Fwd Packets": np.where(y==1,rng.poisson(80,n),rng.poisson(12,n)),
    "Total Backward Packets": np.where(y==1,rng.poisson(65,n),rng.poisson(10,n)),
    "Total Length of Fwd Packets": np.where(y==1,rng.gamma(4,5000,n),rng.gamma(2,800,n)),
    "Flow Bytes/s": np.where(y==1,rng.gamma(4,80000,n),rng.gamma(2,9000,n)),
    "Flow Packets/s": np.where(y==1,rng.gamma(3,500,n),rng.gamma(2,40,n)),
    "Packet Length Mean": np.where(y==1,rng.normal(850,100,n),rng.normal(300,70,n)).clip(1),
    "Active Mean": np.where(y==1,rng.gamma(3,30000,n),rng.gamma(2,4000,n)),
    "Label": np.where(y==1,"ATTACK","BENIGN")
})
df.to_csv(out,index=False)
print(f"Created CI-only synthetic fixture: {out} ({len(df)} rows)")
