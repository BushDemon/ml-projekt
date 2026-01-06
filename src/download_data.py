import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


def download_and_save_openml():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openml_name = cfg["dataset"]["openml_name"]

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg["dataset"]["local_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / "X.npy"
    y_path = out_dir / "y.npy"

    if X_path.exists() and y_path.exists():
        print("Dataset already saved locally.")
        return

    print("Downloading dataset from OpenML...")
    ds = fetch_openml(name=openml_name, as_frame=False)

    X = ds.data.astype(np.float32)
    y = ds.target.astype(int)

    np.save(X_path, X)
    np.save(y_path, y)

    print("Dataset saved to:", out_dir)

if __name__ == "__main__":
    download_and_save_openml()