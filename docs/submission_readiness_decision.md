# Submission Readiness Decision

Decision: STRONG_REVISE

ICLR main-conference readiness: NO.

Reason: The v4 rebuild contains an implemented local tactile-language benchmark, strong local baselines, seven seeds, paired comparisons, ablations, stress sweeps, negative cases, figures, and a reproducible PDF. On `combined_hard_shift`, `grounding_debt_planner` reaches `0.595 +/- 0.044` task success with damage `0.173`, while all-channel tactile probing reaches `0.304 +/- 0.034` task success with damage `0.551`.

The paper is not ready for ICLR main because the evidence is generated local tactile-language data, not real tactile sensor traces, robot hardware rollouts, or a recognized high-fidelity tactile benchmark.

Honest terminal action: keep as `STRONG_REVISE`; do not submit to ICLR main without external tactile validation.

Revival condition: validate grounding-debt estimation on real tactile-language manipulation data or a recognized tactile simulator, add external baselines, and rewrite as a full empirical submission.

