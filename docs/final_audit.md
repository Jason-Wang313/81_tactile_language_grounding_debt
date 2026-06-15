# Final Audit

1. Chosen thesis: Tactile-Language Grounding Debt explores `Measure what language-conditioned policies fail to ground until touch resolves it.` for tactile-language-action models.
2. ICLR-main decision: STRONG_REVISE.
3. Submission-hardening version: v4.
4. Reason: the v4 rebuild adds implemented local evidence and beats local tactile/language baselines, but still lacks real tactile hardware, real robot, or recognized high-fidelity benchmark validation.
5. Decisive result: on `combined_hard_shift`, `grounding_debt_planner` reaches `0.595 +/- 0.044` task success.
6. Strongest baseline pressure: all-channel tactile probing reaches `0.304 +/- 0.034` task success but has damage `0.551` and probe cost `0.790`.
7. Proposed tradeoff: damage `0.173` and probe cost `0.227`.
8. Caveat: conflict detection is nearly neutral, and calibration is not best-in-class.
9. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
10. Reproducibility: `python src\run_experiment.py` regenerates metrics and figures.
11. Claim-validity status: promising local mechanism; not submission-ready until external validation is added.
12. Exact Downloads PDF path: `C:/Users/wangz/Downloads/81.pdf`
13. GitHub URL: https://github.com/Jason-Wang313/81_tactile_language_grounding_debt
14. Confirmation: no visible Desktop copy was requested or made.

## 2026-06-15 Continuation Audit

1. Plan-first requirement: satisfied by `docs/paper81_iclr_submission_execution_plan_20260615.md` before the evidence gate was rerun.
2. Code gate: `python -m py_compile src/run_experiment.py` passed.
3. Experiment gate: `python src/run_experiment.py` completed with terminal recommendation STRONG_REVISE.
4. CSV integrity gate: audited 13,440 `rollouts.csv` rows, 1,680 `dataset_summary.csv` rows, 280 `raw_seed_metrics.csv` rows, 320 `metrics.csv` rows, 100 `pairwise_stats.csv` rows, 2,352 `ablation_rollouts.csv` rows, 7 `ablation_metrics.csv` rows, 31,500 `stress_sweep_raw.csv` rows, 150 `stress_sweep.csv` rows, and 16 `negative_cases.csv` rows.
5. Coverage gate: seeds 0 through 6, eight main methods, five main splits, seven ablations, five stress axes, and six stress levels are present.
6. Decisive split: on `combined_hard_shift`, `grounding_debt_planner` reaches 0.59524 +/- 0.04352 success, while `passive_tactile_classifier` reaches 0.50595 +/- 0.02917, `greedy_active_touch` reaches 0.43155 +/- 0.05615, and `strong_tactile_then_policy` reaches 0.30357 +/- 0.03435.
7. Paired statistics: the action-success gain over `strong_tactile_then_policy` is 0.29167 +/- 0.03674 with 7/7 better seeds. The success gain over `greedy_active_touch` is 0.16369 +/- 0.02587 with 7/7 better seeds.
8. Safety/cost gate: the proposed method has damage 0.17262 versus 0.55059 for strong tactile probing, and probe cost 0.22660 versus 0.79000.
9. Ablation gate: removing tactile belief update drops success from 0.59524 to 0.41667, removing active probe selection drops it to 0.52679, and removing the debt estimator drops it to 0.52976. Removing language/vision conflict detection is nearly neutral and should not be overclaimed.
10. Stress gate: at maximum combined stress, `grounding_debt_planner` reaches 0.40000 +/- 0.04508 success versus 0.30000 +/- 0.06837 for `vision_language_policy`, 0.27143 +/- 0.02640 for `greedy_active_touch`, and 0.16667 +/- 0.04032 for `strong_tactile_then_policy`.
11. PDF gate: `paper/main.pdf` rebuilt after BibTeX-author and float-placement cleanup, then copied to `C:/Users/wangz/Downloads/81.pdf`.
12. Artifact gate: `C:/Users/wangz/Downloads/81.pdf` SHA256 is `38A15AD79C700B29B5665633E68741EB534F52C9BF5FF8BC154A0E5135040D3A`; `C:/Users/wangz/Desktop/81.pdf` is absent.
13. Final decision: STRONG_REVISE. The local evidence is promising and reproducible, but the artifact is not ICLR-main-ready without real tactile hardware traces, robot rollouts, external tactile-language-action baselines, or recognized high-fidelity tactile benchmark validation.
