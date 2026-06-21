# Child Status 81

Current stage: 2026-06-21 expanded v5 audit terminal
Last update: 2026-06-21 21:19:15 +08:00
PDF: C:/Users/wangz/Downloads/81.pdf
PDF SHA256: 6447BA3CE2C62A0D36394D07AEF807ADF9F440D68AB40E55319D5D7844CB17C5
PDF pages: 50
GitHub: https://github.com/Jason-Wang313/81_tactile_language_grounding_debt
Submission-hardening version: v5 expanded hostile-review audit
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Evidence summary: the 2026-06-21 plan-first audit rebuilt Paper 81 into a stronger tactile-language benchmark with 51,840 main rollouts, 4,320 support rows, 1,080 seed-metric rows, 1,188 aggregate metric rows, 441 paired comparisons, hard-regime aggregates, 6,400 ablation rollouts, 80,640 stress rollouts, 15,360 fixed-risk rollouts, 16 negative cases, bright boxed citation links, and a 50-page validated PDF in Downloads only. On `combined_hard_shift`, `grounding_debt_planner_v5` reaches 0.43542 +/- 0.02948 action success, versus 0.42083 +/- 0.03209 for `risk_aware_touch_policy`, 0.41458 +/- 0.05145 for `budgeted_information_gain`, 0.38125 +/- 0.05131 for `passive_tactile_classifier`, and 0.21875 +/- 0.04606 for `strong_tactile_then_policy`. The hard-regime aggregate action success is 0.64141 +/- 0.00966. The terminal decision is KILL_ARCHIVE because fixed-risk coverage is nearly zero at the 0.05 false-safe budget, ablations/simple selectors match or beat the full method on parts of the hard split, and no real tactile hardware, robot rollouts, external learned-baseline, or recognized high-fidelity tactile benchmark evidence is present.
