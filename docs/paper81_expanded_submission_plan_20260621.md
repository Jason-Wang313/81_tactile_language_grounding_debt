# Paper 81 Expanded v5 Submission-Hardening Plan

Date: 2026-06-21

Target: rebuild Paper 81 into the strongest honest CPU-only archive/submission artifact possible. Do not optimize for pretty results. Optimize for a result that survives hostile review. Because the workspace does not contain real tactile hardware logs, robot rollouts, learned tactile-language checkpoints, or a recognized high-fidelity tactile simulator, the maximum honest terminal state is `STRONG_REVISE`; assign `KILL_ARCHIVE` if the mechanism fails strong local baselines or safety gates.

## Claim Under Test

Language-conditioned robot manipulation policies accumulate hidden physical grounding debt when language and vision leave action-relevant tactile/material facts unresolved. A method should repay only the tactile debt that matters for the imminent action, rather than either ignoring touch or probing all channels invasively.

The reference method is `grounding_debt_planner_v5`: a task-conditioned debt estimator with language/vision conflict sensing, value-of-information tactile probe selection, safety-aware probe filtering, calibrated action confidence, fixed-risk abstention, and counterfactual material checks.

## Frozen Experimental Scope

- CPU only.
- RAM-light deterministic tactile-language simulator.
- Seeds: 0-9.
- Main episodes per split/seed: 48.
- Demonstration-free local benchmark; no real tactile-hardware or robot-rollout claim is allowed.
- Main splits:
  - `seen_clean`
  - `language_alias_shift`
  - `visual_counterfactual`
  - `tactile_necessary_ambiguity`
  - `material_novelty_shift`
  - `safety_critical_fragility`
  - `probe_budget_shift`
  - `combined_hard_shift`
  - `adversarial_language_vision_trap`
- Hard aggregate splits:
  - `language_alias_shift`
  - `visual_counterfactual`
  - `tactile_necessary_ambiguity`
  - `material_novelty_shift`
  - `safety_critical_fragility`
  - `probe_budget_shift`
  - `combined_hard_shift`
  - `adversarial_language_vision_trap`

## Main Methods

1. `language_prior_policy`
2. `vision_language_policy`
3. `uncertainty_threshold_policy`
4. `passive_tactile_classifier`
5. `greedy_active_touch`
6. `strong_tactile_then_policy`
7. `risk_aware_touch_policy`
8. `budgeted_information_gain`
9. `calibrated_debt_threshold`
10. `counterfactual_material_filter`
11. `grounding_debt_planner_v5`
12. `oracle_tactile_upper_bound`

The reference must beat strong non-oracle baselines, not merely language-only, weak uncertainty, or intentionally over-invasive tactile policies. The oracle is an upper bound and is not counted as a method to beat.

## Planned Row Counts

- Main rollouts: 9 splits * 10 seeds * 48 episodes * 12 methods = 51,840.
- Dataset/support diagnostics: 9 splits * 10 seeds * 48 episodes = 4,320.
- Raw seed metrics: 9 splits * 10 seeds * 12 methods = 1,080.
- Main aggregate metrics: 9 splits * 12 methods * 11 metrics = 1,188.
- Pairwise stats: 9 splits * 7 comparisons * 7 metrics = 441.
- Aggregate hard-regime seed metrics: 10 seeds * 12 methods = 120.
- Aggregate hard-regime metrics: 12 methods * 11 metrics = 132.
- Aggregate hard-regime pairwise stats: 7 comparisons * 7 metrics = 49.
- Ablation rollouts: 2 splits * 10 seeds * 32 episodes * 10 ablations = 6,400.
- Ablation seed metrics: 2 splits * 10 seeds * 10 ablations = 200.
- Ablation metrics: 2 splits * 10 ablations = 20.
- Stress raw rows: 6 stress axes * 7 stress levels * 10 seeds * 24 episodes * 8 methods = 80,640.
- Stress aggregate rows: 6 stress axes * 7 stress levels * 8 methods = 336.
- Fixed-risk/calibration raw rows: 2 splits * 4 risk budgets * 10 seeds * 32 episodes * 6 methods = 15,360.
- Fixed-risk/calibration seed metrics: 2 splits * 4 budgets * 10 seeds * 6 methods = 480.
- Fixed-risk/calibration metrics: 2 splits * 4 budgets * 6 methods * 4 metrics = 192.
- Fixed-risk/calibration pairwise rows: 2 splits * 4 budgets * 5 comparisons = 40.
- Curated negative cases: 16.

## Metrics

- Task/action success.
- Damage rate.
- Slip/drop rate.
- Action-parameter error.
- Probe count.
- Probe cost.
- Fact accuracy on task-relevant hidden facts.
- Grounding-debt calibration error.
- False-safe confidence rate.
- Abstention rate.
- Tail action-parameter error.
- Paired seed-level differences versus strong baselines.
- Fixed-risk success under bounded false-safe confidence.

## Ablations

- `grounding_debt_v5_full`
- `no_debt_estimator`
- `no_active_probe_selection`
- `no_tactile_belief_update`
- `no_language_vision_conflict_detector`
- `no_safety_gate`
- `no_probe_cost_regularizer`
- `no_calibration`
- `no_counterfactual_material_filter`
- `oracle_handoff`

The mechanism gate fails if practical ablations match or beat the full method on hard splits, especially if removing calibration, counterfactual filtering, or safety gating improves success without increasing damage/false-safe risk.

## Stress And Fixed-Risk Tests

Stress axes: language aliasing, visual counterfactuality, tactile noise, material novelty, safety criticality, and combined shift.

Stress levels: 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50.

Fixed-risk budgets bound false-safe confidence rate: 0.00, 0.02, 0.05, 0.10.

The method must not win by being overconfident on unsafe or ambiguous manipulation episodes. It must retain useful success under fixed false-safe confidence budgets and avoid collapse at maximum combined stress.

## Decision Gates

Assign `STRONG_REVISE` only if all of the following hold:

- `grounding_debt_planner_v5` improves hard-aggregate task success over `passive_tactile_classifier`, `greedy_active_touch`, `strong_tactile_then_policy`, `risk_aware_touch_policy`, `budgeted_information_gain`, `calibrated_debt_threshold`, and `counterfactual_material_filter` by at least 0.05.
- Paired lower confidence bounds against the strongest non-oracle baselines are positive for task success and nonpositive for damage, probe cost, and false-safe confidence.
- The method does not lose maximum-combined-stress success to a non-oracle baseline.
- It clears at least one nonzero fixed-risk budget without being matched by a simpler baseline.
- Ablations show necessity of debt estimation, active probe selection, tactile belief update, safety gating, calibration, and counterfactual material filtering.

If any of these fail, report `KILL_ARCHIVE` or keep `STRONG_REVISE` only if the evidence remains locally strong but externally incomplete. No result may be hidden after the full protocol starts.

## Manuscript And Artifact Rules

- Generate the manuscript only from frozen CSV artifacts.
- Minimum target length: 25 pages, using real theory, tables, appendices, and evidence rather than filler.
- Bright citation boxes are required with `hyperref` border settings.
- Final numbered PDF must be `C:/Users/wangz/Downloads/81.pdf`.
- Do not create or copy `81.pdf` to the visible Desktop.
- Public GitHub repository must be updated after validation.
