# Evaluation scaffold

`evaluate_baseline.py` compares baseline classifiers with a student-grouped split. It expects a prepared feature table with:
- `id_student`: used only to keep a student's rows together across the split;
- `At_Risk`: binary target (1 = fail/withdrawn, 0 = pass/distinction);
- engineered feature columns derived only from the prediction-time information available.

Run it with:

```bash
python src/evaluate_baseline.py --features data/prepared_cutoff_safe_features.csv
```

This is an evaluation scaffold, not a complete data pipeline. The training and evaluation scripts reject a raw `final_result` column because it directly reveals the target. Do not feed the current notebook's final-outcome aggregates until you have rebuilt those aggregates to respect the intended cutoff. A grouped split helps avoid student overlap but cannot by itself prevent other forms of target leakage. If the intended use is future presentations, also report a chronological/presentation holdout.
