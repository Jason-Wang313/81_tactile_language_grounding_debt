# 81 Tactile-Language Grounding Debt

Submission-hardening version: v5 expanded hostile-review audit.

Terminal decision: `KILL_ARCHIVE`.

Paper 81 was rebuilt into a CPU-only tactile-language manipulation benchmark with nine main splits, twelve methods, ten seeds, hard-regime aggregation, stronger tactile baselines, adversarial language/vision traps, component ablations, six-axis stress sweeps, fixed-risk false-safe calibration budgets, negative cases, generated evidence tables, and a validation script.

The expanded evidence is useful but not submission-ready. On `combined_hard_shift`, `grounding_debt_planner_v5` reaches `0.435 +/- 0.029` task success, compared with `0.421 +/- 0.032` for `risk_aware_touch_policy`, `0.415 +/- 0.051` for `budgeted_information_gain`, `0.381 +/- 0.051` for `passive_tactile_classifier`, and `0.219 +/- 0.046` for `strong_tactile_then_policy`. The hard-regime aggregate is `0.641 +/- 0.010`. However, fixed-risk coverage is nearly zero at the 0.05 false-safe budget, several ablations or simpler selectors match or beat the full method on hard splits, and the evidence remains generated tactile-language data rather than real tactile hardware or robot rollouts.

This artifact should be archived as a negative/diagnostic mechanism package, not submitted as an ICLR-main paper.

## Reproduce Evidence

```powershell
python src\run_experiment.py
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
python scripts\validate_submission_artifacts.py
```

The frozen full run writes:

- `results/rollouts.csv` with 51,840 main rollout rows.
- `results/dataset_summary.csv` with 4,320 support rows.
- `results/raw_seed_metrics.csv` with 1,080 seed-metric rows.
- `results/metrics.csv` with 1,188 aggregate metric rows.
- `results/pairwise_stats.csv` with 441 paired-comparison rows.
- `results/aggregate_seed_metrics.csv` with 120 hard-regime seed rows.
- `results/aggregate_metrics.csv` with 132 hard-regime metric rows.
- `results/aggregate_pairwise_stats.csv` with 49 hard-regime paired rows.
- `results/ablation_rollouts.csv` with 6,400 ablation rollout rows.
- `results/ablation_seed_metrics.csv` with 200 ablation seed rows.
- `results/ablation_metrics.csv` with 20 ablation metric rows.
- `results/stress_sweep_raw.csv` with 80,640 stress rollout rows.
- `results/stress_sweep.csv` with 336 stress aggregate rows.
- `results/fixed_risk_raw.csv` with 15,360 fixed-risk rollout rows.
- `results/fixed_risk_seed_metrics.csv` with 480 fixed-risk seed rows.
- `results/fixed_risk_metrics.csv` with 192 fixed-risk metric rows.
- `results/fixed_risk_pairwise.csv` with 40 fixed-risk comparison rows.
- `results/negative_cases.csv` with 16 curated negative cases.
- Figures under `figures/`.

## PDF Artifact

Canonical numbered PDF: `C:/Users/wangz/Downloads/81.pdf`

Validated page count: 50 pages.

Validated SHA256: `6447BA3CE2C62A0D36394D07AEF807ADF9F440D68AB40E55319D5D7844CB17C5`

The manuscript uses bright boxed citation links via `hyperref` border settings. No visible Desktop copy of `81.pdf` is produced.
