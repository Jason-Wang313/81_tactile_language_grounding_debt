# Hostile Reviewer Response

Paper: 81 Tactile-Language Grounding Debt

## Strongest Technical Threats
- LINGO-Space: Language-Conditioned Incremental Grounding for Space (2024)
- Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models (2024)
- SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments (2024)
- DRAGON: A Dialogue-Based Robot for Assistive Navigation with Visual Language Grounding (2023)
- When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs (2026)
- What Matters in Building Vision-Language-Action Models for Generalist Robots (2024)
- Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy (2026)
- What Matters in Language Conditioned Robotic Imitation Learning over Unstructured Data (2022)

## ICLR Main Response
A hostile ICLR reviewer would still be correct to reject this as a final main-conference submission because the v4 evidence is generated local tactile-language data, not real tactile hardware or robot validation. However, the v4 rebuild is no longer a template probability scaffold: it implements paper-specific baselines, debt-aware active touch, paired statistics, ablations, stress tests, negative cases, and figures.

## Honest Action
The paper is marked `STRONG_REVISE`. The local result is promising enough to continue, but not enough to submit.

## What Would Be Needed To Submit
- Real tactile-language manipulation experiments or a recognized tactile simulator.
- External tactile-language-action baselines.
- Qualitative real probe traces and rollouts.
- Manual full-paper related-work audit.
- A narrower claim around the components that the ablations actually validate.

## 2026-06-15 Continuation Response

The continuation audit reran the full benchmark and still gives a meaningful local positive result:

- `grounding_debt_planner`: 0.59524 +/- 0.04352 success on `combined_hard_shift`.
- `passive_tactile_classifier`: 0.50595 +/- 0.02917 success.
- `greedy_active_touch`: 0.43155 +/- 0.05615 success.
- `strong_tactile_then_policy`: 0.30357 +/- 0.03435 success, with 0.55059 damage and 0.79000 probe cost.
- Paired success gain over `strong_tactile_then_policy`: 0.29167 +/- 0.03674, better in 7/7 seeds.
- At maximum combined stress, the proposed method reaches 0.40000 +/- 0.04508 success, above all non-oracle stress baselines.

The hostile reviewer still wins on submission readiness because all evidence is generated local tactile-language data. The honest action remains STRONG_REVISE, not ICLR-main-ready.
