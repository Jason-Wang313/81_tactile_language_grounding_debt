# 81 Tactile-Language Grounding Debt

Submission-hardening version: v4

Terminal decision: STRONG_REVISE for ICLR main conference.

Paper 81 was rebuilt from a v3 synthetic archive into a local tactile-language manipulation benchmark. The runner generates language instructions, visual object evidence, hidden tactile/material facts, tactile probes with cost and damage risk, and robot action targets. It implements language-only, vision-language, conservative uncertainty, passive tactile, greedy active touch, all-channel tactile, grounding-debt planning, oracle upper bound, ablations, stress sweeps, negative cases, and figures.

The local evidence is promising but not final. On `combined_hard_shift`, `grounding_debt_planner` reaches `0.595 +/- 0.044` task success, compared with `0.506 +/- 0.029` for passive tactile classification, `0.432 +/- 0.056` for greedy active touch, and `0.304 +/- 0.034` for all-channel tactile probing. Damage is `0.173` for the proposed method versus `0.551` for all-channel tactile probing, with probe cost `0.227` versus `0.790`.

This is still not ICLR-main submission-ready because the evidence is a generated local benchmark, not real tactile hardware, real robot rollouts, or a recognized high-fidelity tactile simulator.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

This writes:

- `results/rollouts.csv` with 13,440 main rollout rows.
- `results/dataset_summary.csv`.
- `results/raw_seed_metrics.csv`.
- `results/metrics.csv`.
- `results/pairwise_stats.csv`.
- `results/ablation_rollouts.csv` and `results/ablation_metrics.csv`.
- `results/stress_sweep_raw.csv` and `results/stress_sweep.csv`.
- `results/negative_cases.csv`.
- Figures under `figures/`.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/81.pdf`

No visible Desktop PDF is required or produced.

