from pathlib import Path
import pandas as pd


def high_confidence_wrongs(preds_csv, percentile = 0.9):
    df = pd.read_csv(preds_csv)

    wrong = df[df["y_true"] != df["y_pred"]].copy()

    if len(wrong) == 0:
        print("No wrong predictions found.")
        return wrong

    threshold = wrong["confidence"].quantile(percentile)

    high_conf = wrong[wrong["confidence"] >= threshold]

    print("=" * 60)
    print(f"File: {preds_csv.name}")
    print(f"Total samples: {len(df)}")
    print(f"Wrong predictions: {len(wrong)}")
    print(f"Percentile used: {int(percentile * 100)}%")
    print(f"Confidence threshold: {threshold:.4f}")
    print(f"High-confidence wrongs: {len(high_conf)}")
    print()

    counts_true = (
        high_conf["y_true"]
        .value_counts()
        .sort_index()
    )

    print("High-confidence wrongs per y_true:")
    print(counts_true)
    print()

    return high_conf


BASE_DIR = Path(__file__).resolve().parent.parent

run_dir = BASE_DIR / "results" / "20260105-171946_exp1_pca_rf"

# TEST
hc_test = high_confidence_wrongs(run_dir / "preds_test.csv",percentile=0.9)
# VAL
hc_val = high_confidence_wrongs(run_dir / "preds_val.csv",percentile=0.9)
