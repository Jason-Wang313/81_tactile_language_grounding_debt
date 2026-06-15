# Plan

Build paper 81 `tactile_language_grounding_debt` from the shared pool, compile PDF to Downloads only, and publish the exact-name public repo.

## 2026-06-15 Continuation Plan and Result

Plan before execution: re-audit Paper 81 under the ICLR-main-target gate without reducing experiment quality. The required checks were code compilation, full deterministic experiment rerun, CSV row/schema/seed/method coverage, main tactile baseline comparisons, paired uncertainty, damage/cost tradeoffs, ablations, stress sweeps, PDF/BibTeX cleanliness, Downloads-only artifact placement, Desktop exclusion, and public GitHub readiness.

Result: `python -m py_compile src/run_experiment.py` passed and `python src/run_experiment.py` regenerated 13,440 main rollout rows, 2,352 ablation rollout rows, and 31,500 stress rollout rows. The local evidence still supports STRONG_REVISE: `grounding_debt_planner` reaches 0.59524 +/- 0.04352 success on `combined_hard_shift`, compared with 0.50595 +/- 0.02917 for `passive_tactile_classifier`, 0.43155 +/- 0.05615 for `greedy_active_touch`, and 0.30357 +/- 0.03435 for `strong_tactile_then_policy`. The paired gain over strong tactile is 0.29167 +/- 0.03674 with 7/7 better seeds, while damage and probe cost are much lower. The paper remains not ICLR-main-ready because no real tactile hardware, robot rollout, external tactile-language-action baseline, or recognized high-fidelity tactile benchmark evidence is present.
