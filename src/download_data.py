from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


def download_and_save_openml(openml_name: str, out_dir: Path):
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
