# Dataset setup

This project expects the four OULAD-style CSV files in `data/raw/`:

- `studentInfo.csv`
- `studentAssessment.csv`
- `assessments.csv`
- `studentVle.csv`

Download the dataset from its official distribution and review its terms of use. Do not commit raw student-level data to Git. The data directory is ignored by Git.

Build features available by day 28:

```bash
python src/build_features.py --data-dir data/raw --cutoff-day 28
```

The script writes `data/processed/features_day28.csv` and `reports/data_quality.json`. Verify the date semantics against the dataset documentation before interpreting this as an operational day-28 model.
