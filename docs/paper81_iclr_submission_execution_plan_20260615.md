# Paper 81 ICLR Submission-Readiness Execution Plan - 2026-06-15

## Objective

Re-audit Paper 81, `tactile_language_grounding_debt`, as an honest ICLR-main-target candidate. The paper may remain `STRONG_REVISE` if the local tactile-language benchmark reruns cleanly and the grounding-debt planner still improves the success/damage/cost tradeoff over strong tactile baselines. It cannot be marked ICLR-main-ready without real tactile hardware traces, robot rollouts, a recognized high-fidelity tactile benchmark, or external tactile-language-action baselines.

## Evidence Gate

1. Verify the runner with `python -m py_compile src/run_experiment.py`.
2. Re-run `python src/run_experiment.py` once with the existing deterministic benchmark, preserving all seven seeds, all baselines, ablations, stress axes, and row counts.
3. Audit CSV integrity and expected scale:
   - `rollouts.csv`: 13,440 main rollout rows.
   - `dataset_summary.csv`: generated episode coverage for the five main splits.
   - `raw_seed_metrics.csv`, `metrics.csv`, and `pairwise_stats.csv`: aggregate and paired uncertainty across seeds.
   - `ablation_rollouts.csv`: 2,352 rows.
   - `ablation_metrics.csv`: seven ablation rows.
   - `stress_sweep_raw.csv`: 31,500 rows.
   - `stress_sweep.csv`: stress-axis/method/level summaries.
   - `negative_cases.csv`: retained failure exemplars.
4. Confirm seeds 0 through 6, all eight main methods, all five main evaluation splits, all seven ablations, five stress axes, and six stress levels are present.

## Decision Criteria

Keep `STRONG_REVISE` only if all of the following still hold:

1. On `combined_hard_shift`, `grounding_debt_planner` beats the strongest non-oracle tactile baselines on task success by a meaningful margin.
2. Paired success gain over `strong_tactile_then_policy` is positive with seed-level uncertainty reported.
3. Damage and probe cost remain substantially lower than all-channel/strong tactile probing.
4. Ablations support the central grounding-debt mechanism, especially tactile belief update, active probe selection, debt estimation, safety gating, or probe-cost regularization.
5. Neutral components such as language/vision conflict detection are reported honestly rather than overclaimed.
6. Maximum combined stress does not reverse the proposed method below the strongest non-oracle baseline.
7. The paper explicitly states that generated local tactile-language evidence is insufficient for ICLR-main submission readiness.

Downgrade to `KILL_ARCHIVE` if the reproduced evidence fails strong baselines, if damage/cost tradeoffs are not favorable, if ablations contradict the mechanism, if stress sweeps reverse the claim, if CSVs disagree materially with the manuscript, or if the artifact cannot be rebuilt cleanly.

Do not upgrade to ICLR-main-ready unless new external/real tactile/robot/high-fidelity evidence is actually present in the repository.

## Artifact Gate

1. Rebuild the PDF with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
2. Fix recoverable LaTeX/BibTeX issues, including placeholder bibliography warnings or fragile float placement, without changing the empirical claim.
3. Copy only the canonical numbered PDF to `C:/Users/wangz/Downloads/81.pdf`.
4. Confirm `C:/Users/wangz/Desktop/81.pdf` is absent.
5. Record the Downloads PDF SHA256.

## Documentation And Repo Gate

1. Update child status, plan, final audit, hostile reviewer response, attack log, submission readiness docs, and version log with the continuation result.
2. Update root `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md` so the continuation audit is current through Paper 81.
3. Verify the public GitHub repository URL and visibility.
4. Commit and push the Paper 81 repository.
5. Verify local `HEAD` equals `origin/main` and the worktree is clean before moving to Paper 82.

## RAM Discipline

Run one Paper 81 experiment process at a time. Do not reduce seeds, baselines, ablations, stress levels, or row counts to save memory; preserve experiment quality and rely on the repo's lightweight deterministic generator.
