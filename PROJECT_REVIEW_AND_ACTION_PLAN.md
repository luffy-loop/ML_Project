# Project review and implementation status

## Implemented in the repository

- Cutoff-aware feature generation for assessment submissions and VLE interactions.
- Student-grouped baseline training with classification metrics, Brier score, threshold/workload comparisons, subgroup error summaries, and permutation importance.
- Separate grouped model comparison script.
- Optional latest-presentation holdout evaluation.
- Shared prediction helper with required-column and empty-input validation.
- Streamlit CSV scoring dashboard and CSV export.
- Synthetic-data tests and GitHub Actions workflow.
- Reproducibility, responsible-use, and final-results documentation.

## Remaining validation requiring the official dataset

- Run all commands in README.md in a clean environment with official OULAD data.
- Confirm OULAD date semantics and whether each withdrawn outcome was already known by day 28. A retrospective final-result label may not support prospective withdrawal prediction.
- Verify presentation chronology and run an untouched later-presentation holdout; grouped random splitting alone is insufficient for future-cohort claims.
- Inspect data-quality reports, missingness, subgroup errors, calibration, threshold/workload trade-offs, and prediction errors.
- Run the publication notebook top-to-bottom and reconcile all claims in the report and presentation to reproducible outputs.

## Do not claim

- Notebook accuracy is a verified prospective early-warning result.
- Fixed score bands are calibrated probabilities.
- Subgroup summaries certify fairness.
- The application is production-ready without data protection, educator oversight, monitoring, and temporal validation.

## Definition of done

- [ ] Full data pipeline and tests run in a clean environment.
- [ ] Label timing is prospective and cutoff-safe.
- [ ] Later-presentation holdout is reported and reviewed.
- [ ] Metrics, calibration, and threshold trade-offs are reviewed.
- [ ] Presentation claims match saved reproducible results.
- [ ] Privacy, responsible use, and institutional review requirements are satisfied.
