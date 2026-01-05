import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / cfg["dataset"]["local_dir"]

    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    rs = int(cfg["run"]["random_state"])
    test_size = float(cfg["dataset"]["test_size"])
    val_size = float(cfg["dataset"]["val_size"])

    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=rs, stratify=y
    )

    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size, random_state=rs, stratify=y_trainval
    )

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / "idx_train.npy", idx_train)
    np.save(splits_dir / "idx_val.npy", idx_val)
    np.save(splits_dir / "idx_test.npy", idx_test)

    print("Split saved to:", splits_dir)
    print("Sizes:")
    print("  train:", len(idx_train))
    print("  val:  ", len(idx_val))
    print("  test: ", len(idx_test))

    def class_counts(indices):
        vals, cnts = np.unique(y[indices], return_counts=True)
        return dict(zip(vals.tolist(), cnts.tolist()))

    print("Train class counts (first few):", list(class_counts(idx_train).items())[:3])
    print("Val class counts (first few):  ", list(class_counts(idx_val).items())[:3])
    print("Test class counts (first few): ", list(class_counts(idx_test).items())[:3])


if __name__ == "__main__":
    main()
