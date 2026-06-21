import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(value):
    text = str(value)
    text = text.encode("ascii", "ignore").decode("ascii")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def method_label(name):
    labels = {
        "language_prior_policy": "lang prior",
        "vision_language_policy": "vision-lang",
        "uncertainty_threshold_policy": "uncertain",
        "passive_tactile_classifier": "passive tactile",
        "greedy_active_touch": "greedy touch",
        "strong_tactile_then_policy": "touch-all",
        "risk_aware_touch_policy": "risk-aware",
        "budgeted_information_gain": "budgeted IG",
        "calibrated_debt_threshold": "calib. debt",
        "counterfactual_material_filter": "cf-material",
        "grounding_debt_planner_v5": "GDP-v5",
        "oracle_tactile_upper_bound": "oracle",
    }
    return labels.get(name, name.replace("_", " "))


def short_label(value):
    mapping = {
        "seen_clean": "seen",
        "language_alias_shift": "lang",
        "visual_counterfactual": "vision-cf",
        "tactile_necessary_ambiguity": "tactile",
        "material_novelty_shift": "material",
        "safety_critical_fragility": "safety",
        "probe_budget_shift": "budget",
        "combined_hard_shift": "combined",
        "adversarial_language_vision_trap": "adv-trap",
        "hard_aggregate": "hard-agg",
        "grounding_debt_v5_full": "full",
        "no_debt_estimator": "no-debt",
        "no_active_probe_selection": "no-active",
        "no_tactile_belief_update": "no-update",
        "no_language_vision_conflict_detector": "no-conflict",
        "no_safety_gate": "no-safety",
        "no_probe_cost_regularizer": "no-cost",
        "no_calibration": "no-calib",
        "no_counterfactual_material_filter": "no-cf",
        "oracle_handoff": "oracle",
        "damage_or_probe_damage": "damage",
        "slip_or_drop": "slip",
        "wrong_action_parameter": "param",
        "false_safe_confidence": "false-safe",
        "debt_calibration_error": "debt-calib",
        "tail_param_error": "tail-param",
        "fixed_risk_success": "risk-succ",
        "false_safe_rate": "false-safe",
        "action_success": "success",
        "probe_cost": "cost",
        "probe_count": "probe-n",
        "fact_accuracy": "fact-acc",
        "param_error": "param",
        "abstain_rate": "abstain",
    }
    text = str(value)
    return mapping.get(text, text.replace("_", "-"))


def compact_cell(col, row):
    value = row.get(col, "")
    if col in {"method", "target", "reference", "Method"}:
        return method_label(value)
    if col in {"split", "ablation", "metric", "failure_label", "stress_axis"}:
        return short_label(value)
    if col == "lesson":
        text = str(value)
        return text[:68] + ("..." if len(text) > 68 else "")
    if col in {"object_family", "probes"}:
        text = str(value).replace("_", "-").replace(";", ",")
        if text == "fragile,slippery,heavy,full,soft,locked":
            return "all"
        return text
    return value


def metric_lookup(rows, split, method, metric):
    for row in rows:
        if row.get("split") == split and row.get("method") == method and row.get("metric") == metric:
            return float(row["mean"]), float(row["ci95"])
    return 0.0, 0.0


def fmt_mean_ci(rows, split, method, metric):
    mean, ci = metric_lookup(rows, split, method, metric)
    return f"{mean:.3f} +/- {ci:.3f}"


def bib_key(i):
    return f"prior{i}"


def clean_bib_text(text):
    text = str(text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("\\n", " ").replace("\\", " ")
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = re.sub(r"[{}%#$^]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_bib():
    rows = read_csv(DOCS / "deep_read_250.csv")
    chosen = []
    seen = set()
    for row in rows:
        title = clean_bib_text(row.get("title", "Untitled"))
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        chosen.append(row)
        if len(chosen) >= 24:
            break
    entries = []
    for idx, row in enumerate(chosen, 1):
        key = bib_key(idx)
        title = clean_bib_text(row.get("title", "Untitled"))
        authors = clean_bib_text(row.get("authors", "Local Prior Work Pool")).replace(";", " and ")
        year = clean_bib_text(row.get("year", "2026")) or "2026"
        venue = clean_bib_text(row.get("venue", "preprint")) or "preprint"
        doi = clean_bib_text(row.get("doi", ""))
        arxiv = clean_bib_text(row.get("arxiv_id", ""))
        url = clean_bib_text(row.get("url", ""))
        note_bits = []
        if doi:
            note_bits.append(f"https://doi.org/{doi}")
        if arxiv:
            note_bits.append(f"arXiv:{arxiv}")
        if url and not note_bits:
            note_bits.append(url)
        note = "; ".join(note_bits) if note_bits else "local robotics literature pool"
        entries.append(
            "\n".join(
                [
                    f"@misc{{{key},",
                    f"  author = {{{authors}}},",
                    f"  title = {{{title}}},",
                    f"  year = {{{year}}},",
                    f"  howpublished = {{{venue}}},",
                    f"  note = {{{note}}}",
                    "}",
                ]
            )
        )
    (PAPER / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return [bib_key(i) for i in range(1, len(chosen) + 1)]


def table_rows(rows, columns, limit=None):
    out = []
    selected = rows if limit is None else rows[:limit]
    for row in selected:
        out.append(" & ".join(esc(compact_cell(col, row)) for col in columns) + r" \\")
    return "\n".join(out)


def longtable(title, label, columns, rows, widths=None):
    col_spec = widths or ("l" * len(columns))
    header = " & ".join(esc(c) for c in columns) + r" \\"
    body = table_rows(rows, columns)
    return rf"""
\begin{{center}}
\tiny
\renewcommand{{\arraystretch}}{{0.92}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{{col_spec}}}
\caption{{{esc(title)}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\toprule
{header}
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\renewcommand{{\arraystretch}}{{1.0}}
\end{{center}}
"""


def chunked_longtables(prefix, columns, rows, chunk=42):
    sections = []
    for idx in range(0, len(rows), chunk):
        part = idx // chunk + 1
        sections.append(longtable(f"{prefix} part {part}", f"tab:{prefix.lower().replace(' ', '-')}-{part}", columns, rows[idx : idx + chunk]))
    return "\n".join(sections)


def main_table(metric_rows, split):
    methods = [
        "language_prior_policy",
        "vision_language_policy",
        "uncertainty_threshold_policy",
        "passive_tactile_classifier",
        "greedy_active_touch",
        "strong_tactile_then_policy",
        "risk_aware_touch_policy",
        "budgeted_information_gain",
        "calibrated_debt_threshold",
        "counterfactual_material_filter",
        "grounding_debt_planner_v5",
        "oracle_tactile_upper_bound",
    ]
    rows = []
    for method in methods:
        rows.append(
            {
                "Method": method_label(method),
                "Success": fmt_mean_ci(metric_rows, split, method, "action_success"),
                "Damage": fmt_mean_ci(metric_rows, split, method, "damage"),
                "Cost": fmt_mean_ci(metric_rows, split, method, "probe_cost"),
                "False-safe": fmt_mean_ci(metric_rows, split, method, "false_safe_confidence"),
                "Fact acc.": fmt_mean_ci(metric_rows, split, method, "fact_accuracy"),
            }
        )
    return rows


def tabular(caption, label, columns, rows):
    header = " & ".join(esc(c) for c in columns) + r" \\"
    body = table_rows(rows, columns)
    return rf"""
\begin{{table}}[tbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lccccc}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{{esc(caption)}}}
\label{{{label}}}
\end{{table}}
"""


def generate_tex():
    metric_rows = read_csv(RESULTS / "metrics.csv")
    hard_metrics = read_csv(RESULTS / "aggregate_metrics.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")
    hard_pairwise = read_csv(RESULTS / "aggregate_pairwise_stats.csv")
    ablations = read_csv(RESULTS / "ablation_metrics.csv")
    ablation_seed = read_csv(RESULTS / "ablation_seed_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    fixed_pair = read_csv(RESULTS / "fixed_risk_pairwise.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    raw_seed = read_csv(RESULTS / "raw_seed_metrics.csv")
    dataset = read_csv(RESULTS / "dataset_summary.csv")
    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    decision = "KILL_ARCHIVE" if "Terminal recommendation: KILL_ARCHIVE" in summary else "STRONG_REVISE"
    decision_tex = esc(decision)
    citation_keys = generate_bib()
    cites_a = ",".join(citation_keys[:8])
    cites_b = ",".join(citation_keys[8:16])
    cites_c = ",".join(citation_keys[16:24])
    combined_rows = main_table(metric_rows, "combined_hard_shift")
    hard_rows = main_table(hard_metrics, "hard_aggregate")
    ablation_rows = [
        {
            "Split": r["split"],
            "Ablation": r["ablation"].replace("grounding_debt_v5_full", "full"),
            "Success": f"{float(r['action_success']):.3f} +/- {float(r['ci95_success']):.3f}",
            "Damage": f"{float(r['damage']):.3f}",
            "Cost": f"{float(r['probe_cost']):.3f}",
            "False-safe": f"{float(r['false_safe_confidence']):.3f}",
        }
        for r in ablations
    ]
    fixed_budget_rows = [
        {
            "Method": method_label(r["method"]),
            "Metric": r["metric"],
            "Mean": f"{float(r['mean']):.3f}",
            "CI95": f"{float(r['ci95']):.3f}",
        }
        for r in fixed
        if r["split"] == "combined_hard_shift" and r["risk_budget"] == "0.05"
    ]
    stress_max_rows = [
        {
            "Method": method_label(r["method"]),
            "Success": f"{float(r['action_success']):.3f} +/- {float(r['ci95_success']):.3f}",
            "Damage": f"{float(r['damage']):.3f}",
            "Cost": f"{float(r['probe_cost']):.3f}",
            "False-safe": f"{float(r['false_safe_confidence']):.3f}",
        }
        for r in stress
        if r["stress_axis"] == "combined" and r["stress_level"] == "1.50"
    ]
    tex = rf"""\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=false,pdfborder={{0 0 1.6}},citebordercolor={{0 1 0}},linkbordercolor={{1 0.55 0}},urlbordercolor={{0 0.45 1}}}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{amsmath}}
\title{{Tactile-Language Grounding Debt Under Hidden Physical Facts}}
\author{{Anonymous Authors}}
\begin{{document}}
\maketitle

\begin{{abstract}}
Language-conditioned manipulation can look grounded while action-critical tactile facts remain unknown. This expanded v5 rebuild audits whether a robot should explicitly estimate and repay \emph{{tactile-language grounding debt}} before acting. The protocol contains nine evaluation splits, twelve methods, ten seeds, hard-regime aggregation, component ablations, six-axis stress tests, fixed-risk false-safe calibration, and curated negative cases. The terminal recommendation is \textbf{{{decision_tex}}}. On \texttt{{combined\_hard\_shift}}, \texttt{{grounding\_debt\_planner\_v5}} obtains {fmt_mean_ci(metric_rows, "combined_hard_shift", "grounding_debt_planner_v5", "action_success")} task success, compared with {fmt_mean_ci(metric_rows, "combined_hard_shift", "risk_aware_touch_policy", "action_success")} for risk-aware touch and {fmt_mean_ci(metric_rows, "combined_hard_shift", "budgeted_information_gain", "action_success")} for budgeted information gain. The hard-regime aggregate is {fmt_mean_ci(hard_metrics, "hard_aggregate", "grounding_debt_planner_v5", "action_success")}. The paper is not ICLR-main-ready: fixed-risk coverage is nearly zero at the 0.05 false-safe budget, several ablations or simpler selectors match the full method on hard splits, and the evidence remains generated tactile-language data rather than real tactile hardware or robot rollouts.
\end{{abstract}}

\section{{Decision}}
This document is the expanded v5 hostile-review audit for Paper 81. The frozen runner wrote 51,840 main rollout rows, 6,400 ablation rollout rows, 80,640 stress rows, 15,360 fixed-risk rows, and 16 curated negative cases. The correct terminal decision is \textbf{{{decision_tex}}}. The result should be archived rather than submitted because the strongest local story does not survive the full evidence gate.

The surrounding literature already contains language-conditioned robot imitation, tactile-language-action modeling, tactile commonsense reasoning, and language/vision grounding benchmarks \citep{{{cites_a}}}. That prior-work pressure means a synthetic local benchmark must be treated as mechanism debugging evidence, not as final submission proof. More recent tactile and multimodal robot work makes the external-evidence gap sharper, not softer \citep{{{cites_b}}}.

\section{{Problem Setup}}
Each episode contains a task $g$, language evidence $\ell$, visual evidence $v$, latent tactile facts $z$, optional tactile probes $q$, and a continuous robot action $a$. The hidden facts encode fragility, slipperiness, mass, fill level, compliance, and latch state. The target action is generated by a deterministic physical rule $a^\star=f(g,z)$; the learner acts using beliefs $b(z\mid \ell,v,q)$.

Grounding debt is the task-relevant residual uncertainty
\[
D(g,b)=\frac{{1}}{{|R(g)|}}\sum_{{j\in R(g)}} 2\min(b_j,1-b_j),
\]
augmented by language/vision conflict and safety cost. A useful method should improve task success without merely buying success through invasive probing. Therefore the protocol measures success, damage, slip/drop, action-parameter error, probe count, probe cost, fact accuracy, calibration error, false-safe confidence, abstention, and tail error.

\section{{Method Under Audit}}
\texttt{{grounding\_debt\_planner\_v5}} ranks tactile probes by task relevance, entropy, language/vision conflict, safety criticality, and counterfactual material pressure, then discounts high-risk probes by cost and damage risk. It updates tactile beliefs only for selected channels and uses calibrated confidence to avoid high-risk unsafe action. This is a mechanism-level proposal, not a trained tactile foundation model.

The core hypothesis is falsifiable: if budgeted information gain, risk-aware touch, calibrated debt thresholds, or counterfactual material filtering match the full planner under hard splits, the full mechanism is not yet necessary. The v5 protocol intentionally includes those baselines.

\section{{Frozen Protocol}}
The main splits are \texttt{{seen\_clean}}, \texttt{{language\_alias\_shift}}, \texttt{{visual\_counterfactual}}, \texttt{{tactile\_necessary\_ambiguity}}, \texttt{{material\_novelty\_shift}}, \texttt{{safety\_critical\_fragility}}, \texttt{{probe\_budget\_shift}}, \texttt{{combined\_hard\_shift}}, and \texttt{{adversarial\_language\_vision\_trap}}. The hard aggregate excludes only the clean split. Each main split has ten seeds and 48 episodes per seed.

The baselines are language-only, vision-language, uncertainty thresholding, passive tactile classification, greedy active touch, touch-all tactile probing, risk-aware touch, budgeted information gain, calibrated debt thresholding, counterfactual material filtering, the full v5 planner, and an oracle. The oracle is a sanity upper bound and is not counted as a baseline to beat.

\section{{Main Evidence}}
Table~\ref{{tab:combined-main}} shows the decisive combined hard-shift split. Table~\ref{{tab:hard-main}} shows the aggregate over all hard regimes. The v5 method improves over several tactile baselines and is much safer than touch-all probing, but the margin against risk-aware and budgeted selectors is too small for a main-conference claim.

{tabular("Combined hard-shift main metrics.", "tab:combined-main", ["Method", "Success", "Damage", "Cost", "False-safe", "Fact acc."], combined_rows)}

{tabular("Hard-regime aggregate metrics.", "tab:hard-main", ["Method", "Success", "Damage", "Cost", "False-safe", "Fact acc."], hard_rows)}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.96\linewidth]{{../figures/grounding_debt_success.png}}
\caption{{Combined hard-shift task success for all methods.}}
\end{{figure}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.96\linewidth]{{../figures/grounding_debt_damage_cost.png}}
\caption{{Damage and probe cost. Touch-all obtains high fact accuracy by accepting unsafe probe exposure.}}
\end{{figure}}

\section{{Ablation Evidence}}
The ablations are mixed. Tactile belief update and probe-cost regularization are clearly important. Removing calibration produces similar task success but a large false-safe increase, so calibration matters for safety reporting rather than raw success. However, removing active probe selection or safety gating can match or improve success on \texttt{{combined\_hard\_shift}}, which prevents a clean submission claim.

{tabular("Ablations on the two hostile splits.", "tab:ablation-main", ["Split", "Ablation", "Success", "Damage", "Cost", "False-safe"], ablation_rows)}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.96\linewidth]{{../figures/grounding_debt_ablation.png}}
\caption{{Ablations are reported even when they hurt the method claim.}}
\end{{figure}}

\section{{Stress And Fixed-Risk Evidence}}
At maximum combined stress, the full method remains the strongest non-oracle method, but the absolute success is low. This is useful as a failure diagnostic and weak as a submission result.

{tabular("Maximum combined-stress results.", "tab:stress-main", ["Method", "Success", "Damage", "Cost", "False-safe"], stress_max_rows)}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.96\linewidth]{{../figures/grounding_debt_stress_sweep.png}}
\caption{{Combined language, vision, tactile, material, and safety stress sweep.}}
\end{{figure}}

The fixed-risk result is more damaging. At a 0.05 false-safe budget on \texttt{{combined\_hard\_shift}}, coverage is nearly zero for all methods. A method that cannot act under a safety budget cannot be sold as a robust tactile-language solution.

{tabular("Fixed-risk budget 0.05 on combined hard shift.", "tab:fixed-risk-main", ["Method", "Metric", "Mean", "CI95"], fixed_budget_rows)}

\section{{Related Work Pressure}}
The claim is pressured by language-conditioned imitation learning, touch-language modeling, tactile commonsense reasoning, multimodal grounding, and contact-rich manipulation datasets \citep{{{cites_c}}}. The present work does not train a tactile-language-action model, does not release real tactile trajectories, and does not evaluate on robot hardware. Therefore the honest contribution is an audit harness and negative evidence about selective tactile grounding debt, not a main-ready empirical breakthrough.

\section{{Limitations}}
The benchmark is generated. The tactile channels are stylized probabilities rather than sensor traces. The damage model is hand-specified. The all-channel tactile baseline is intentionally punished for invasive sensing. A real robot might use safer tactile sensors, learned active perception, or compliant probing. The fixed-risk protocol exposes that the confidence model is too conservative to provide useful coverage at tight false-safe budgets.

\section{{Conclusion}}
Paper 81 now has a concrete expanded audit: more baselines, more hard regimes, more seeds, fixed-risk safety analysis, and full appendices. The local mechanism remains interesting, but the expanded evidence supports \textbf{{{decision_tex}}}. The next honest revival path is real tactile hardware or a recognized high-fidelity tactile-language-action benchmark, not another round of synthetic polishing.

\clearpage
\appendix
\section{{Full Main Metrics}}
{chunked_longtables("Main metrics", ["split", "method", "metric", "mean", "ci95", "seeds", "episodes_per_seed"], metric_rows, chunk=45)}
\clearpage
\section{{Full Paired Main Statistics}}
{chunked_longtables("Paired main statistics", ["split", "target", "reference", "metric", "mean_diff", "ci95", "target_better_seeds", "seeds"], pairwise, chunk=45)}
\clearpage
\section{{Hard-Regime Aggregate Tables}}
{longtable("Hard-regime aggregate metrics", "tab:hard-full", ["split", "method", "metric", "mean", "ci95", "seeds", "episodes_per_seed"], hard_metrics)}
{longtable("Hard-regime paired statistics", "tab:hard-pair-full", ["split", "target", "reference", "metric", "mean_diff", "ci95", "target_better_seeds", "seeds"], hard_pairwise)}
\clearpage
\section{{Ablation Seed Tables}}
{longtable("Ablation metrics", "tab:ablation-full", ["split", "ablation", "action_success", "ci95_success", "damage", "probe_cost", "fact_accuracy", "false_safe_confidence", "abstain_rate"], ablations)}
{chunked_longtables("Ablation seed metrics", ["split", "ablation", "seed", "action_success", "damage", "probe_cost", "fact_accuracy", "false_safe_confidence"], ablation_seed, chunk=45)}
\clearpage
\section{{Stress Sweep Tables}}
{chunked_longtables("Stress sweep", ["stress_axis", "stress_level", "method", "action_success", "ci95_success", "damage", "probe_cost", "false_safe_confidence", "rows"], stress, chunk=45)}
\clearpage
\section{{Fixed-Risk Tables}}
{chunked_longtables("Fixed-risk metrics", ["split", "risk_budget", "method", "metric", "mean", "ci95", "seeds", "episodes_per_seed"], fixed, chunk=45)}
{longtable("Fixed-risk paired statistics", "tab:fixed-pair-full", ["split", "risk_budget", "reference", "fixed_risk_success_diff", "fixed_risk_success_ci95", "false_safe_diff", "target_better_seeds", "seeds"], fixed_pair)}
\clearpage
\section{{Negative Cases}}
{longtable("Curated negative cases", "tab:negative-cases", ["split", "seed", "episode_id", "method", "task", "failure_label", "probes", "confidence"], negative)}
\clearpage
\section{{Seed Metric Sample}}
{chunked_longtables("Seed metric sample", ["split", "method", "seed", "action_success", "damage", "probe_cost", "fact_accuracy", "false_safe_confidence"], raw_seed[:180], chunk=45)}
\clearpage
\section{{Dataset Diagnostic Sample}}
{chunked_longtables("Dataset diagnostic sample", ["split", "seed", "episode_id", "task", "language_ambiguity", "visual_counterfactual", "tactile_noise", "material_novelty", "safety_critical"], dataset[:180], chunk=45)}

\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    generate_tex()
