import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(cfg):
    out_root = BASE_DIR / cfg["run"]["output_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{ts}_{cfg['run']['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_local_dataset(data_dir):
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    return X, y


def load_splits(splits_dir):
    idx_train = np.load(splits_dir / "idx_train.npy")
    idx_val = np.load(splits_dir / "idx_val.npy")
    idx_test = np.load(splits_dir / "idx_test.npy")
    return idx_train, idx_val, idx_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = BASE_DIR / cfg["dataset"]["local_dir"]
    splits_dir = data_dir / "splits"

    X, y = load_local_dataset(data_dir)
    idx_train, idx_val, idx_test = load_splits(splits_dir)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    m = cfg["model"]
    clf = RandomForestClassifier(
        n_estimators=int(m["n_estimators"]),
        max_depth=None if m["max_depth"] is None else int(m["max_depth"]),
        min_samples_leaf=int(m["min_samples_leaf"]),
        n_jobs=int(m["n_jobs"]),
        random_state=int(cfg["run"]["random_state"]),
    )

    print("Training RandomForest...")
    clf.fit(X_train, y_train)
    print("Training done.")

    def eval_split(name, Xs, ys):
        y_pred = clf.predict(Xs)
        acc = accuracy_score(ys, y_pred)

        proba = clf.predict_proba(Xs)
        conf = np.max(proba, axis=1)

        cm = confusion_matrix(ys, y_pred)
        report = classification_report(ys, y_pred, digits=4)

        return acc, y_pred, conf, cm, report

    val_acc, val_pred, val_conf, val_cm, val_report = eval_split("val", X_val, y_val)
    test_acc, test_pred, test_conf, test_cm, test_report = eval_split("test", X_test, y_test)

    run_dir = make_run_dir(cfg)

    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    metrics = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(val_cm).to_csv(run_dir / "confusion_val.csv", index=False)
    pd.DataFrame(test_cm).to_csv(run_dir / "confusion_test.csv", index=False)

    (run_dir / "classification_report_val.txt").write_text(val_report, encoding="utf-8")
    (run_dir / "classification_report_test.txt").write_text(test_report, encoding="utf-8")

    pd.DataFrame(
        {"split": "val", "idx": idx_val, "y_true": y_val, "y_pred": val_pred, "confidence": val_conf}
    ).to_csv(run_dir / "preds_val.csv", index=False)

    pd.DataFrame(
        {"split": "test", "idx": idx_test, "y_true": y_test, "y_pred": test_pred, "confidence": test_conf}
    ).to_csv(run_dir / "preds_test.csv", index=False)

    print("Run saved to:", run_dir)
    print(f"VAL accuracy:  {val_acc:.4f}")
    print(f"TEST accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
