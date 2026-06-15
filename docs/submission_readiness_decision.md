# Submission Readiness Decision

Decision: STRONG_REVISE

ICLR main-conference readiness: NO.

Reason: The v4 rebuild contains an implemented local tactile-language benchmark, strong local baselines, seven seeds, paired comparisons, ablations, stress sweeps, negative cases, figures, and a reproducible PDF. On `combined_hard_shift`, `grounding_debt_planner` reaches `0.595 +/- 0.044` task success with damage `0.173`, while all-channel tactile probing reaches `0.304 +/- 0.034` task success with damage `0.551`.

The paper is not ready for ICLR main because the evidence is generated local tactile-language data, not real tactile sensor traces, robot hardware rollouts, or a recognized high-fidelity tactile benchmark.

Honest terminal action: keep as `STRONG_REVISE`; do not submit to ICLR main without external tactile validation.

Revival condition: validate grounding-debt estimation on real tactile-language manipulation data or a recognized tactile simulator, add external baselines, and rewrite as a full empirical submission.

## 2026-06-15 Continuation Decision

Decision: STRONG_REVISE.

ICLR main-conference readiness: NO.

Reason: the full deterministic benchmark was rerun and the positive local evidence reproduces. On `combined_hard_shift`, `grounding_debt_planner` reaches 0.59524 +/- 0.04352 success versus 0.30357 +/- 0.03435 for `strong_tactile_then_policy`, with paired gain 0.29167 +/- 0.03674 and 7/7 better seeds. The method also reduces damage from 0.55059 to 0.17262 and probe cost from 0.79000 to 0.22660 compared with strong tactile probing.

Blocker: the evidence is still generated local tactile-language data with no real tactile sensor traces, robot hardware validation, recognized high-fidelity tactile benchmark, or external tactile-language-action baseline. This blocks ICLR-main readiness even though the local mechanism is promising.
