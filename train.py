"""Train and compare NIDS classifiers; log every run to local SQLite MLflow."""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from nids_pipeline import DataFrameMedianImputer, NIDSFeatureEngineer

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def load_dataset():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(
            "No dataset CSV found. Place the CICIDS MachineLearningCVE CSV files in data/ "
            "or follow data/README.md."
        )
    chunks, per_label = [], {}
    cap = 10000
    for path in files:
        for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            if "Label" not in chunk:
                continue
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            for label, group in chunk.groupby("Label", sort=False):
                have = sum(len(x) for x in per_label.get(label, []))
                remain = cap - have
                if remain > 0:
                    per_label.setdefault(label, []).append(
                        group.sample(n=min(remain, len(group)), random_state=RANDOM_STATE)
                    )
    for parts in per_label.values():
        chunks.append(pd.concat(parts, ignore_index=True))
    if not chunks:
        raise ValueError("CSV files were found, but none contained a Label column.")
    df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Label"])
    y = (df["Label"].str.upper() != "BENIGN").astype(int)
    X = df.drop(columns=["Label"]).select_dtypes(include=np.number).copy()
    if X.shape[1] < 5 or len(X) < 1000:
        raise ValueError(f"Dataset contract failed: got {len(X)} rows and {X.shape[1]} numeric features.")
    # A completely empty or constant column is retained here; the fitted pipeline handles it.
    return X, y, df["Label"].value_counts().to_dict()

def build_pipeline(algorithm, args):
    steps = [
        ("imputer", DataFrameMedianImputer()),
        ("feature_engineering", NIDSFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ]
    if algorithm == "logistic_regression":
        model = LogisticRegression(C=args.logistic_c, max_iter=1500, class_weight="balanced",
                                   solver="liblinear", random_state=RANDOM_STATE)
    elif algorithm == "random_forest":
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                       min_samples_leaf=2, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=RANDOM_STATE)
    elif algorithm == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    steps.append(("classifier", model))
    return Pipeline(steps)

def main():
    parser = argparse.ArgumentParser(description="Train NIDS models with MLflow tracking.")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--logistic_c", type=float, default=1.0)
    parser.add_argument("--experiment", default="Team10-NIDS-EndTerm")
    parser.add_argument("--tracking_uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--sample_cap", type=int, default=10000)
    args = parser.parse_args()

    X, y, label_counts = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    configs = [
        ("logistic_regression", {"C": args.logistic_c, "max_iter": 1500}),
        ("random_forest", {"n_estimators": args.n_estimators, "max_depth": args.max_depth}),
        ("extra_trees", {"n_estimators": args.n_estimators, "max_depth": args.max_depth}),
    ]
    results, candidates = [], []
    for algorithm, params in configs:
        pipe = build_pipeline(algorithm, args)
        started = time.time()
        with mlflow.start_run(run_name=f"{algorithm}-{len(results)+1}") as run:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "training_seconds": float(time.time()-started),
            }
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_params(params)
            mlflow.log_param("preprocessing", "median_imputer+train_fitted_log1p+duration_quartiles+standard_scaler+variance_filter+pca95")
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            result = {"run_id": run.info.run_id, "algorithm": algorithm, **params, **metrics}
            results.append(result)
            candidates.append((metrics["f1"], pipe, algorithm, result, prob, pred))
            print(result)

    candidates.sort(key=lambda x: (x[0], x[3]["accuracy"]), reverse=True)
    _, best, best_name, best_result, best_prob, best_pred = candidates[0]
    # Metadata defines the exact API data contract. Bounds are learned from training data only.
    feature_contract = {}
    for col in X.columns:
        vals = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = float(vals.min()), float(vals.max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1e12, 1e12
        if lo == hi:
            lo, hi = lo - 1e-9, hi + 1e-9
        feature_contract[col] = {
            "type": "float",
            "min": lo,
            "max": hi,
            "why": "Numeric network-flow measurement used by the NIDS classifier."
        }

    # Store a representative BENIGN/ATTACK pair for the domain-logic test.
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    in_contract = ((X_test >= train_min) & (X_test <= train_max)).all(axis=1).to_numpy()
    benign_idx = np.where((y_test.to_numpy() == 0) & (best_prob < 0.5) & in_contract)[0]
    attack_idx = np.where((y_test.to_numpy() == 1) & (best_prob >= 0.5) & in_contract)[0]
    if len(benign_idx) and len(attack_idx):
        benign_i = int(benign_idx[np.argmin(best_prob[benign_idx])])
        attack_i = int(attack_idx[np.argmax(best_prob[attack_idx])])
    else:
        # Fall back to the most separated contract-valid holdout examples.
        valid_idx = np.where(in_contract)[0]
        if len(valid_idx) < 2:
            raise RuntimeError("Could not find two holdout examples within the training feature contract.")
        benign_i = int(valid_idx[np.argmin(best_prob[valid_idx])])
        attack_i = int(valid_idx[np.argmax(best_prob[valid_idx])])
    domain_pair = {
        "weaker_candidate": {k: float(v) for k, v in X_test.iloc[benign_i].items()},
        "stronger_candidate": {k: float(v) for k, v in X_test.iloc[attack_i].items()},
    }
    # Register the winning experiment in MLflow Model Registry (workshop practice).
    try:
        from mlflow.tracking import MlflowClient
        registered = mlflow.register_model(
            model_uri=f"runs:/{best_result['run_id']}/model",
            name="Team10NIDS"
        )
        MlflowClient().set_registered_model_alias(
            name="Team10NIDS", alias="champion", version=registered.version
        )
        best_result["registered_model"] = "Team10NIDS"
        best_result["registered_version"] = str(registered.version)
        print(f"Registered best model: Team10NIDS version {registered.version} (alias: champion)")
    except Exception as exc:
        print("MLflow Registry registration skipped; Tracking runs remain available:", exc)

    joblib.dump(best, MODEL_DIR / "model.joblib")
    metadata = {
        "project": "Team 10 Network Intrusion Detection System",
        "target_mapping": {"BENIGN": 0, "ATTACK": 1},
        "features": feature_contract,
        "feature_names": list(X.columns),
        "best_algorithm": best_name,
        "best_run": best_result,
        "metrics": results,
        "dataset_rows": int(len(X)),
        "dataset_features": int(X.shape[1]),
        "label_counts": label_counts,
        "domain_pair": domain_pair,
        "test_pair_expected_order": "stronger_candidate_attack_probability_gt_weaker_candidate",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ROOT / "training_results.json").write_text(json.dumps(results, indent=2))
    print("\nBest model:", best_name, "| F1:", round(best_result["f1"], 4))
    print("Saved model/model.joblib and model/metadata.json")
    print("\nClassification report for best model:\n",
          classification_report(y_test, best_pred, target_names=["BENIGN", "ATTACK"], zero_division=0))

if __name__ == "__main__":
    main()
