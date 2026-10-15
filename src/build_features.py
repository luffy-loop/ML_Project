"""Build prediction features from the raw OULAD CSV files."""
import argparse
from pathlib import Path
import pandas as pd
from data import build_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--cutoff-day", type=int, default=28)
    parser.add_argument("--output", default="data/processed/features_day28.csv")
    parser.add_argument("--quality-report", default="reports/data_quality.json")
    args = parser.parse_args()
    features, quality = build_features(args.data_dir, args.cutoff_day)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    report = Path(args.quality_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    import json
    report.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Wrote {len(features):,} rows to {out}")
    print(json.dumps(quality, indent=2))
