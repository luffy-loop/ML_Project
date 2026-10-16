# Final evaluation report

This is a template. Do not publish metrics until the pipeline has run on the official dataset and the prediction cutoff, label timing, and split have been audited.

## Reproduction

```bash
python src/build_features.py --data-dir data/raw --cutoff-day 28
python src/evaluate_baseline.py --features data/processed/features_day28.csv
python src/evaluate_temporal.py --features data/processed/features_day28.csv
python src/train.py --features data/processed/features_day28.csv
pytest -q
```

## Required results

Populate from generated JSON reports: row and student counts per split; precision, recall, F1, PR-AUC, ROC-AUC; confusion matrix; Brier score and calibration analysis; threshold/workload trade-offs chosen using validation data; subgroup error review with sample sizes; and later-presentation holdout results.

## Limitations

A grouped random holdout does not prove future-presentation generalization. Final-result labels may include withdrawals that occurred before the prediction cutoff. Results support research and voluntary outreach only, not automatic student decisions.
