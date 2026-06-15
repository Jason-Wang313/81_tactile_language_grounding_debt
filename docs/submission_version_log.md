# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/81.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats were not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Tactile-Language Grounding Debt Rebuild
- Added `docs/paper81_rebuild_plan.md` before executing changes.
- Replaced the scalar probability scaffold with an implemented tactile-language manipulation benchmark.
- Implemented language-only, vision-language, conservative uncertainty, passive tactile, greedy active touch, all-channel tactile, grounding-debt planning, and oracle methods.
- Ran seven seeds, 13,440 main rollout rows, 2,352 ablation rollout rows, and 31,500 stress-sweep rows.
- Produced figures, paired statistics, ablations, stress sweeps, and negative cases.
- Terminal decision: STRONG_REVISE.

## v4 Continuation Audit - 2026-06-15

- Added `docs/paper81_iclr_submission_execution_plan_20260615.md` before rerunning the evidence gate.
- Recompiled and reran the full deterministic tactile-language benchmark without reducing seeds, baselines, ablations, stress levels, or row counts.
- Rechecked CSV integrity, seed/method/split/ablation/stress coverage, paired statistics, damage/probe-cost tradeoffs, ablations, stress behavior, PDF logs, Downloads-only placement, Desktop exclusion, and public GitHub status.
- Cleaned BibTeX placeholder entries by adding explicit authors and changed fragile `[h]` floats to `[tbp]` before rebuilding the PDF.
- Verified `C:/Users/wangz/Downloads/81.pdf` SHA256 `38A15AD79C700B29B5665633E68741EB534F52C9BF5FF8BC154A0E5135040D3A`.
- Terminal decision remains: STRONG_REVISE.
