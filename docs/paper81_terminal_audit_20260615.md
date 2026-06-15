# Paper 81 Terminal Audit - 2026-06-15

## Scope

Paper 81, `tactile_language_grounding_debt`, was re-audited under the sequential ICLR-main-target continuation gate. The audit tested whether the local tactile-language evidence remains strong enough for `STRONG_REVISE`, and whether any actual evidence exists that could upgrade the paper to ICLR-main-ready.

## Plan

The execution plan was written first in `docs/paper81_iclr_submission_execution_plan_20260615.md`. The plan required code compilation, a full deterministic experiment rerun, CSV integrity checks, main tactile baseline analysis, paired uncertainty, damage/cost tradeoff analysis, ablations, stress sweeps, PDF hygiene, Downloads-only artifact placement, Desktop exclusion, public GitHub verification, and report updates.

## Verification

- Code compilation: `python -m py_compile src/run_experiment.py` passed.
- Experiment rerun: `python src/run_experiment.py` completed and returned terminal recommendation STRONG_REVISE.
- Main rows: 13,440 rollout rows, 1,680 dataset-summary rows, 280 seed-level metric rows, 320 aggregate-metric rows, and 100 pairwise-stat rows.
- Ablation rows: 2,352 ablation rollout rows and 7 ablation summary rows.
- Stress rows: 31,500 stress rollout rows and 150 stress summary rows.
- Negative cases: 16 rows.
- Seeds: 0 through 6.
- Methods: `greedy_active_touch`, `grounding_debt_planner`, `language_prior_policy`, `oracle_tactile_upper_bound`, `passive_tactile_classifier`, `strong_tactile_then_policy`, `uncertainty_threshold_policy`, and `vision_language_policy`.

## Central Evidence

On `combined_hard_shift`, `grounding_debt_planner` reaches 0.59524 +/- 0.04352 success. `passive_tactile_classifier` reaches 0.50595 +/- 0.02917, `greedy_active_touch` reaches 0.43155 +/- 0.05615, and `strong_tactile_then_policy` reaches 0.30357 +/- 0.03435.

The paired action-success gain over `strong_tactile_then_policy` is 0.29167 +/- 0.03674 with 7/7 better seeds. The proposed method also lowers damage from 0.55059 to 0.17262 and probe cost from 0.79000 to 0.22660 compared with strong tactile probing.

Ablations support the central mechanism. Removing tactile belief update drops success to 0.41667. Removing active probe selection drops success to 0.52679. Removing the debt estimator drops success to 0.52976. Removing language/vision conflict detection is nearly neutral and should not be overclaimed.

At maximum combined stress, `grounding_debt_planner` reaches 0.40000 +/- 0.04508 success, compared with 0.30000 +/- 0.06837 for `vision_language_policy`, 0.27143 +/- 0.02640 for `greedy_active_touch`, and 0.16667 +/- 0.04032 for `strong_tactile_then_policy`.

## Artifact Verification

- Downloads PDF: `C:/Users/wangz/Downloads/81.pdf`
- SHA256: `38A15AD79C700B29B5665633E68741EB534F52C9BF5FF8BC154A0E5135040D3A`
- Desktop PDF: absent at `C:/Users/wangz/Desktop/81.pdf`
- GitHub: `https://github.com/Jason-Wang313/81_tactile_language_grounding_debt`

## Decision

Final decision: STRONG_REVISE.

The local evidence is positive and reproducible, so the paper should not be archived. It is still not ICLR-main-ready because the evidence is generated local tactile-language data, with no real tactile sensor traces, no robot hardware validation, no recognized high-fidelity tactile benchmark, and no external tactile-language-action baseline.
