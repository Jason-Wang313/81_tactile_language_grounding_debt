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

