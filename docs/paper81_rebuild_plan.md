# Paper 81 Rebuild Plan

## Goal

Rebuild `tactile_language_grounding_debt` from an archive-only synthetic scaffold into the strongest honest ICLR-main-target artifact possible. The scientific question is whether a robot can measure and repay the action-relevant grounding debt left by language and vision before tactile interaction reveals hidden physical facts.

## Evidence Standard

The rebuild must replace scalar probability diagnostics with an implemented benchmark, implemented baselines, multi-seed evaluation, ablations, stress tests, uncertainty summaries, negative cases, figures, and a reproducible paper PDF. Because this workspace does not contain real tactile hardware logs, real robot rollouts, or recognized high-fidelity tactile simulator assets, the ceiling is `STRONG_REVISE`. If the proposed debt mechanism fails to clear strong tactile baselines, the correct terminal decision is `KILL_ARCHIVE`.

## Benchmark

Create a deterministic local tactile-language manipulation benchmark with generated instructions, visual object descriptions, hidden tactile/material facts, tactile probe traces, and robot action targets. Each episode will specify an instruction such as pick, pour, open, hand over, slide, insert, or sort, but the correct action parameters depend on hidden tactile facts such as fragility, fill level, compliance, slip coefficient, latch state, texture, or mass.

Evaluation splits:

1. `seen_clean`: familiar language, visual descriptions, materials, and tactile profiles.
2. `language_alias_shift`: paraphrases and ambiguous instructions that preserve task intent but alter priors.
3. `visual_counterfactual`: vision confidently suggests the wrong material or object state.
4. `tactile_necessary_ambiguity`: language and vision are insufficient; only touch can separate safe actions.
5. `combined_hard_shift`: language shift, visual counterfactuals, tactile noise, novel materials, and safety-sensitive actions combined.

## Methods

Implement all methods in `src/run_experiment.py` with no hidden hand-entered result tables.

Baselines:

1. `language_prior_policy`: chooses actions from instruction priors only.
2. `vision_language_policy`: fuses language and visual object descriptors without touch.
3. `uncertainty_threshold_policy`: abstains or chooses conservative actions when language/vision disagree.
4. `passive_tactile_classifier`: receives a fixed tactile reading before acting.
5. `greedy_active_touch`: probes the most uncertain tactile channel, then acts.
6. `strong_tactile_then_policy`: strong classifier over tactile traces followed by a policy.

Proposed mechanism:

`grounding_debt_planner`: estimates which physical facts remain ungrounded after language and vision, chooses targeted tactile probes to repay that debt, updates the action belief, and selects a safety-aware action with probe cost.

Upper bound:

`oracle_tactile_upper_bound`: uses ground-truth hidden tactile facts and action parameters.

## Metrics

Report seed-level and aggregate metrics:

1. Task success.
2. Damage or unsafe-contact rate.
3. Slip/drop rate.
4. Probe count and probe cost.
5. Hidden tactile-fact accuracy.
6. Grounding-debt calibration error.
7. Paired success and damage differences versus the strongest non-oracle tactile baseline.

## Ablations

Evaluate proposed variants on `combined_hard_shift`:

1. Full grounding-debt planner.
2. Minus debt estimator.
3. Minus active probe selection.
4. Minus tactile belief update.
5. Minus language/vision conflict detector.
6. Minus safety gate.
7. Minus probe-cost regularizer.

## Stress Tests

Sweep independent stress levels for language ambiguity, visual hallucination/counterfactual strength, tactile noise, material novelty, and combined safety-critical shift. Compare the proposed method against strong tactile baselines and the oracle upper bound.

## Paper Update

Rewrite the paper as a compact ICLR-style submission-hardening report:

1. Define tactile-language grounding debt operationally.
2. Describe generated benchmark variables, tactile traces, and action targets.
3. Present main results, ablations, stress tests, and negative cases.
4. Include hostile related-work pressure and limitations.
5. Use the terminal decision supported by evidence, not the desired outcome.

## Delivery Checklist

1. Run all experiments from scratch.
2. Produce CSVs and figures in the repo.
3. Compile the PDF and copy only `81.pdf` to `C:\Users\wangz\Downloads`.
4. Update local status files and root pool reports.
5. Commit changes and push the matching public GitHub repository.

