# Claims

- Mechanism claim: tactile-language grounding debt estimates which action-relevant physical facts remain ungrounded after language and vision, then selects targeted tactile probes to repay that debt.
- Evidence claim: the v4 local benchmark shows `grounding_debt_planner` reaches `0.595 +/- 0.044` success on `combined_hard_shift`, outperforming passive tactile, greedy active touch, and all-channel tactile probing on the success/damage/cost tradeoff.
- Safety claim: targeted probing reduces damage to `0.173` versus `0.551` for all-channel tactile probing.
- Ablation claim: debt estimation, active probe selection, tactile belief update, safety gating, and cost regularization matter; language/vision conflict detection is nearly neutral.
- Scope claim: results support a promising local mechanism and justify `STRONG_REVISE`, not ICLR-main submission.
- Unsupported claim explicitly avoided: no claim of SOTA tactile-language robot performance, real-hardware generalization, or deployment readiness.

