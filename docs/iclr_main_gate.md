# ICLR Main Gate

Paper: 81 tactile_language_grounding_debt

Previous v3 decision: KILL_ARCHIVE

Current v4 gate verdict: STRONG_REVISE

Positive evidence:
- Implemented tactile-language manipulation benchmark with hidden tactile/material facts, language/vision evidence, tactile probes, probe cost, and damage risk.
- Implemented baselines: language prior, vision-language, uncertainty threshold, passive tactile, greedy active touch, all-channel tactile, proposed grounding debt planner, and oracle upper bound.
- Seven seeds, 13,440 main rollout rows, 2,352 ablation rollout rows, and 31,500 stress-sweep rows.
- On `combined_hard_shift`, proposed task success is `0.595 +/- 0.044` versus `0.506 +/- 0.029` for passive tactile and `0.432 +/- 0.056` for greedy active touch.
- Proposed damage is `0.173` versus `0.551` for all-channel tactile probing.

Remaining blockers:
- No real tactile hardware traces.
- No robot hardware validation.
- No recognized high-fidelity tactile benchmark.
- No external tactile-language-action baseline.
- Manual related-work synthesis remains thinner than a final ICLR submission requires.

Gate action: continue as `STRONG_REVISE`; do not claim ICLR-main readiness until external tactile validation exists.

