import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 81012026
SEEDS = list(range(7))
TASKS = ["pick", "pour", "open", "handoff", "slide", "insert"]
FACTS = ["fragile", "slippery", "heavy", "full", "soft", "locked"]
METHODS = [
    "language_prior_policy",
    "vision_language_policy",
    "uncertainty_threshold_policy",
    "passive_tactile_classifier",
    "greedy_active_touch",
    "strong_tactile_then_policy",
    "grounding_debt_planner",
    "oracle_tactile_upper_bound",
]
TEST_EPISODES_PER_SPLIT_SEED = 48
STRESS_EPISODES_PER_SEED = 30

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RELEVANT_FACTS = {
    "pick": ["fragile", "slippery", "heavy"],
    "pour": ["full", "slippery", "heavy"],
    "open": ["locked", "soft", "fragile"],
    "handoff": ["fragile", "slippery", "heavy"],
    "slide": ["heavy", "slippery", "soft"],
    "insert": ["soft", "fragile", "slippery"],
}

PROBE_RISK = {
    "fragile": 0.08,
    "slippery": 0.05,
    "heavy": 0.18,
    "full": 0.20,
    "soft": 0.16,
    "locked": 0.22,
}

PROBE_COST = {
    "fragile": 0.10,
    "slippery": 0.08,
    "heavy": 0.16,
    "full": 0.17,
    "soft": 0.13,
    "locked": 0.15,
}


@dataclass
class Episode:
    split: str
    seed: int
    episode_id: int
    task: str
    object_family: str
    facts: dict
    language: dict
    vision: dict
    tactile: dict
    tactile_noise: float
    language_ambiguity: float
    visual_counterfactual: float
    material_novelty: float
    safety_critical: float


def stable_rng(*parts):
    acc = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            for ch in part:
                acc = (acc * 131 + ord(ch)) % (2**32 - 1)
        else:
            acc = (acc * 131 + int(part)) % (2**32 - 1)
    return np.random.default_rng(acc)


def ci95(vals):
    vals = list(vals)
    if len(vals) <= 1:
        return 0.0
    mean = float(np.mean(vals))
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
    return 1.96 * sd / math.sqrt(len(vals))


def clip01(x):
    return float(np.clip(x, 0.02, 0.98))


def logit(p):
    p = clip01(p)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def combine_probs(*items):
    total_weight = sum(w for _, w in items)
    if total_weight <= 0:
        return 0.5
    return clip01(sigmoid(sum(logit(p) * w for p, w in items) / total_weight))


def split_params(split, stress=0.0):
    if split == "seen_clean":
        return 0.08, 0.05, 0.08, 0.03, 0.10
    if split == "language_alias_shift":
        return 0.35 + 0.15 * stress, 0.06, 0.10, 0.06, 0.15
    if split == "visual_counterfactual":
        return 0.12, 0.42 + 0.25 * stress, 0.11, 0.08, 0.22
    if split == "tactile_necessary_ambiguity":
        return 0.30, 0.22, 0.16 + 0.10 * stress, 0.10, 0.35
    if split == "combined_hard_shift":
        return 0.42 + 0.10 * stress, 0.48 + 0.20 * stress, 0.20 + 0.12 * stress, 0.25 + 0.25 * stress, 0.55
    if split == "stress_language":
        return 0.08 + 0.55 * stress, 0.10, 0.10, 0.06, 0.20
    if split == "stress_vision":
        return 0.12, 0.08 + 0.65 * stress, 0.12, 0.08, 0.25
    if split == "stress_tactile":
        return 0.16, 0.18, 0.08 + 0.34 * stress, 0.12, 0.35
    if split == "stress_material":
        return 0.18, 0.22, 0.15 + 0.10 * stress, 0.05 + 0.55 * stress, 0.40
    if split == "stress_combined":
        return 0.12 + 0.42 * stress, 0.14 + 0.48 * stress, 0.10 + 0.28 * stress, 0.05 + 0.50 * stress, 0.30 + 0.40 * stress
    raise ValueError(split)


def make_facts(task, rng):
    priors = {
        "fragile": 0.30,
        "slippery": 0.28,
        "heavy": 0.35,
        "full": 0.30 if task == "pour" else 0.18,
        "soft": 0.32,
        "locked": 0.42 if task == "open" else 0.14,
    }
    if task in {"handoff", "insert"}:
        priors["fragile"] += 0.12
    if task in {"slide", "pick"}:
        priors["heavy"] += 0.10
    facts = {fact: int(rng.random() < priors[fact]) for fact in FACTS}
    if facts["fragile"] and facts["heavy"] and rng.random() < 0.35:
        facts["heavy"] = 0
    return facts


def object_family(task, facts, rng):
    if task == "pour":
        return "opaque_cup" if facts["full"] else "empty_cup"
    if task == "open":
        return "latched_box" if facts["locked"] else "loose_lid_box"
    if task == "insert":
        return "foam_insert" if facts["soft"] else "rigid_insert"
    if facts["fragile"]:
        return str(rng.choice(["glass_part", "thin_shell", "ceramic_tool"]))
    if facts["slippery"]:
        return str(rng.choice(["oily_block", "polished_cylinder"]))
    if facts["heavy"]:
        return "dense_block"
    return "generic_part"


def noisy_prob(truth, signal, noise, rng, invert=False):
    center = 0.5 + (0.5 - noise) * (1 if truth else -1) * signal
    if invert:
        center = 1.0 - center
    return clip01(center + rng.normal(0.0, noise))


def make_episode(split, seed, episode_id, stress=0.0):
    rng = stable_rng("episode", split, seed, episode_id, int(1000 * stress))
    task = TASKS[episode_id % len(TASKS)]
    language_ambiguity, visual_counterfactual, tactile_noise, material_novelty, safety_critical = split_params(split, stress)
    facts = make_facts(task, rng)
    family = object_family(task, facts, rng)
    language = {}
    vision = {}
    tactile = {}
    relevant = set(RELEVANT_FACTS[task])
    for fact in FACTS:
        relevance = 1.0 if fact in relevant else 0.45
        lang_signal = max(0.10, (0.88 - language_ambiguity) * relevance)
        vis_signal = max(0.08, (0.82 - 0.20 * material_novelty) * (0.80 if fact in {"fragile", "soft", "locked"} else 1.0))
        invert = fact in relevant and rng.random() < visual_counterfactual
        language[fact] = noisy_prob(facts[fact], lang_signal, 0.10 + 0.16 * language_ambiguity, rng)
        vision[fact] = noisy_prob(facts[fact], vis_signal, 0.08 + 0.18 * visual_counterfactual + 0.10 * material_novelty, rng, invert=invert)
        tactile_signal = max(0.20, 0.92 - 0.35 * material_novelty)
        tactile[fact] = noisy_prob(facts[fact], tactile_signal, tactile_noise, rng)
    return Episode(
        split=split,
        seed=seed,
        episode_id=episode_id,
        task=task,
        object_family=family,
        facts=facts,
        language=language,
        vision=vision,
        tactile=tactile,
        tactile_noise=tactile_noise,
        language_ambiguity=language_ambiguity,
        visual_counterfactual=visual_counterfactual,
        material_novelty=material_novelty,
        safety_critical=safety_critical,
    )


def base_belief(ep, mode):
    if mode == "language":
        return dict(ep.language)
    if mode == "vision_language":
        return {fact: combine_probs((ep.language[fact], 0.45), (ep.vision[fact], 0.55)) for fact in FACTS}
    raise ValueError(mode)


def entropy(p):
    return 4.0 * p * (1.0 - p)


def conflict(ep, fact):
    return abs(ep.language[fact] - ep.vision[fact])


def debt_score(ep, belief, fact, ablation=None):
    rel = 1.0 if fact in RELEVANT_FACTS[ep.task] else 0.22
    conf = 0.0 if ablation == "minus_language_vision_conflict_detector" else conflict(ep, fact)
    hazard = 0.35 if fact in {"fragile", "slippery", "locked"} else 0.15
    if ablation == "minus_debt_estimator":
        return entropy(belief[fact])
    return rel * (0.65 * entropy(belief[fact]) + 0.55 * conf + hazard * ep.safety_critical)


def choose_probes(ep, belief, method, ablation=None):
    if method == "passive_tactile_classifier":
        return ["slippery", "heavy"]
    if method == "greedy_active_touch":
        return [fact for fact, _ in sorted(((f, entropy(belief[f])) for f in FACTS), key=lambda kv: kv[1], reverse=True)[:2]]
    if method == "strong_tactile_then_policy":
        return list(FACTS)
    if method != "grounding_debt_planner":
        return []
    if ablation == "minus_active_probe_selection":
        return ["slippery", "heavy"]
    if ablation == "minus_probe_cost_regularizer":
        budget = 4
    else:
        budget = 2
    ranked = sorted(((fact, debt_score(ep, belief, fact, ablation=ablation)) for fact in FACTS), key=lambda kv: kv[1], reverse=True)
    probes = []
    for fact, score in ranked:
        if len(probes) >= budget:
            break
        threshold = 0.34 if fact in RELEVANT_FACTS[ep.task] else 0.58
        if score >= threshold:
            probes.append(fact)
    if ablation == "minus_safety_gate":
        return probes
    # Avoid invasive probes when the current belief says the object is probably fragile.
    safe = []
    fragile_belief = belief["fragile"]
    for fact in probes:
        if fragile_belief > 0.70 and PROBE_RISK[fact] > 0.18:
            continue
        safe.append(fact)
    return safe


def apply_tactile(ep, belief, probes, method, ablation=None):
    updated = dict(belief)
    if ablation == "minus_tactile_belief_update":
        return updated
    for fact in probes:
        tactile_weight = 0.72
        if method == "strong_tactile_then_policy":
            tactile_weight = 0.62
        if method == "passive_tactile_classifier":
            tactile_weight = 0.58
        updated[fact] = combine_probs((updated[fact], 1.0 - tactile_weight), (ep.tactile[fact], tactile_weight))
    return updated


def true_action_from_facts(ep, facts):
    task = ep.task
    fragile = facts["fragile"]
    slippery = facts["slippery"]
    heavy = facts["heavy"]
    full = facts["full"]
    soft = facts["soft"]
    locked = facts["locked"]
    grip = 0.42 + 0.22 * slippery + 0.18 * heavy - 0.20 * fragile
    force = 0.45 + 0.22 * heavy + 0.18 * locked - 0.18 * soft - 0.15 * fragile
    speed = 0.58 - 0.18 * fragile - 0.16 * slippery - 0.12 * full - 0.10 * soft
    torque = 0.22 + 0.50 * locked if task == "open" else 0.10 + 0.10 * heavy
    if task == "pour":
        force += 0.12 * full
        speed -= 0.10 * full
    if task == "insert":
        force -= 0.12 * soft
        speed -= 0.10 * soft
    if task == "slide":
        force += 0.18 * heavy
    if task == "handoff":
        speed -= 0.08
    return np.clip(np.array([grip, force, speed, torque], dtype=float), 0.05, 0.95)


def action_from_belief(ep, belief, method, ablation=None):
    soft_facts = {fact: belief[fact] for fact in FACTS}
    action = true_action_from_facts(ep, soft_facts)
    unresolved = np.mean([entropy(belief[f]) for f in RELEVANT_FACTS[ep.task]])
    if method == "uncertainty_threshold_policy" and unresolved > 0.66:
        action[0] *= 0.82
        action[1] *= 0.78
        action[2] *= 0.70
    if method == "grounding_debt_planner" and ablation != "minus_safety_gate":
        fragile_or_slip = max(belief["fragile"], belief["slippery"])
        if fragile_or_slip > 0.68 and unresolved > 0.35:
            action[0] *= 0.90
            action[1] *= 0.88
            action[2] *= 0.84
    return np.clip(action, 0.03, 0.98)


def probe_damage(ep, probes, method, ablation=None):
    risk = sum(PROBE_RISK[p] for p in probes)
    if method == "strong_tactile_then_policy":
        risk *= 0.42
    if method == "grounding_debt_planner" and ablation != "minus_safety_gate":
        risk *= 0.82
    threshold = 0.46 - 0.10 * ep.facts["fragile"] - 0.07 * ep.facts["soft"] - 0.06 * ep.safety_critical
    return int(risk > threshold)


def evaluate_action(ep, action, probes, method, ablation=None):
    target = true_action_from_facts(ep, ep.facts)
    error = float(np.linalg.norm(action - target))
    probe_dmg = probe_damage(ep, probes, method, ablation=ablation)
    damage = int(
        probe_dmg
        or (ep.facts["fragile"] and (action[0] > target[0] + 0.16 or action[1] > target[1] + 0.16))
        or (ep.facts["soft"] and action[1] > target[1] + 0.18)
    )
    slip = int(
        (ep.facts["slippery"] and action[0] < target[0] - 0.14)
        or (ep.facts["heavy"] and action[1] < target[1] - 0.16)
        or (ep.facts["full"] and ep.task == "pour" and action[2] > target[2] + 0.18)
    )
    success = int(error < 0.30 and not damage and not slip)
    if success:
        failure = "success"
    elif damage:
        failure = "damage_or_probe_damage"
    elif slip:
        failure = "slip_or_drop"
    else:
        failure = "wrong_action_parameter"
    return success, damage, slip, error, failure


def run_episode(ep, method, ablation=None):
    if method == "oracle_tactile_upper_bound":
        belief = {fact: 0.98 if ep.facts[fact] else 0.02 for fact in FACTS}
        probes = []
    else:
        mode = "language" if method == "language_prior_policy" else "vision_language"
        belief = base_belief(ep, mode)
        probes = choose_probes(ep, belief, method, ablation=ablation)
        belief = apply_tactile(ep, belief, probes, method, ablation=ablation)
    action = action_from_belief(ep, belief, method, ablation=ablation)
    success, damage, slip, error, failure = evaluate_action(ep, action, probes, method, ablation=ablation)
    relevant = RELEVANT_FACTS[ep.task]
    fact_acc = float(np.mean([int((belief[f] >= 0.5) == bool(ep.facts[f])) for f in relevant]))
    debt = float(np.mean([2.0 * min(belief[f], 1.0 - belief[f]) for f in relevant]))
    wrong = 1.0 - fact_acc
    probe_cost = float(sum(PROBE_COST[p] for p in probes))
    return {
        "split": ep.split,
        "seed": ep.seed,
        "episode_id": ep.episode_id,
        "method": method if ablation is None else ablation,
        "task": ep.task,
        "object_family": ep.object_family,
        "action_success": success,
        "damage": damage,
        "slip_or_drop": slip,
        "param_error": f"{error:.5f}",
        "probe_count": len(probes),
        "probe_cost": f"{probe_cost:.5f}",
        "fact_accuracy": f"{fact_acc:.5f}",
        "debt_score": f"{debt:.5f}",
        "debt_calibration_error": f"{abs(debt - wrong):.5f}",
        "probes": ";".join(probes) if probes else "none",
        "failure_label": failure,
        "language_ambiguity": f"{ep.language_ambiguity:.5f}",
        "visual_counterfactual": f"{ep.visual_counterfactual:.5f}",
        "tactile_noise": f"{ep.tactile_noise:.5f}",
        "material_novelty": f"{ep.material_novelty:.5f}",
        "safety_critical": f"{ep.safety_critical:.5f}",
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_metrics(rows, methods=METHODS):
    out = []
    metrics = ["action_success", "damage", "slip_or_drop", "param_error", "probe_count", "probe_cost", "fact_accuracy", "debt_calibration_error"]
    for split in sorted({r["split"] for r in rows}):
        for method in methods:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                row = {"split": split, "method": method, "seed": seed, "episodes": len(vals)}
                for metric in metrics:
                    row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
                out.append(row)
    return out


def aggregate_metrics(seed_rows, methods=METHODS):
    out = []
    metrics = ["action_success", "damage", "slip_or_drop", "param_error", "probe_count", "probe_cost", "fact_accuracy", "debt_calibration_error"]
    for split in sorted({r["split"] for r in seed_rows}):
        for method in methods:
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            if not vals:
                continue
            for metric in metrics:
                nums = [float(v[metric]) for v in vals]
                out.append({
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "mean": f"{np.mean(nums):.5f}",
                    "ci95": f"{ci95(nums):.5f}",
                    "seeds": len(nums),
                    "episodes_per_seed": vals[0]["episodes"],
                })
    return out


def pairwise_stats(seed_rows):
    refs = ["strong_tactile_then_policy", "greedy_active_touch", "vision_language_policy", "uncertainty_threshold_policy"]
    metrics = ["action_success", "damage", "probe_cost", "fact_accuracy", "debt_calibration_error"]
    rows = []
    for split in sorted({r["split"] for r in seed_rows}):
        for ref in refs:
            for metric in metrics:
                diffs = []
                for seed in SEEDS:
                    tv = [r for r in seed_rows if r["split"] == split and r["method"] == "grounding_debt_planner" and int(r["seed"]) == seed]
                    rv = [r for r in seed_rows if r["split"] == split and r["method"] == ref and int(r["seed"]) == seed]
                    if tv and rv:
                        diffs.append(float(tv[0][metric]) - float(rv[0][metric]))
                higher = metric in {"action_success", "fact_accuracy"}
                rows.append({
                    "split": split,
                    "target": "grounding_debt_planner",
                    "reference": ref,
                    "metric": metric,
                    "mean_diff": f"{np.mean(diffs):.5f}",
                    "ci95": f"{ci95(diffs):.5f}",
                    "target_better_seeds": sum(1 for d in diffs if (d > 0 if higher else d < 0)),
                    "seeds": len(diffs),
                })
    return rows


def metric_value(metric_rows, split, method, metric):
    rows = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    return (float(rows[0]["mean"]), float(rows[0]["ci95"])) if rows else (0.0, 0.0)


def dataset_row(ep):
    row = {
        "split": ep.split,
        "seed": ep.seed,
        "episode_id": ep.episode_id,
        "task": ep.task,
        "object_family": ep.object_family,
        "language_ambiguity": f"{ep.language_ambiguity:.5f}",
        "visual_counterfactual": f"{ep.visual_counterfactual:.5f}",
        "tactile_noise": f"{ep.tactile_noise:.5f}",
        "material_novelty": f"{ep.material_novelty:.5f}",
    }
    for fact in FACTS:
        row[f"true_{fact}"] = ep.facts[fact]
        row[f"language_{fact}"] = f"{ep.language[fact]:.5f}"
        row[f"vision_{fact}"] = f"{ep.vision[fact]:.5f}"
        row[f"tactile_{fact}"] = f"{ep.tactile[fact]:.5f}"
    return row


def run_main():
    rows = []
    dataset = []
    splits = ["seen_clean", "language_alias_shift", "visual_counterfactual", "tactile_necessary_ambiguity", "combined_hard_shift"]
    for split in splits:
        for seed in SEEDS:
            for episode_id in range(TEST_EPISODES_PER_SPLIT_SEED):
                ep = make_episode(split, seed, episode_id)
                dataset.append(dataset_row(ep))
                for method in METHODS:
                    rows.append(run_episode(ep, method))
            print(f"main split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_rows = aggregate_seed_metrics(rows)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "dataset_summary.csv", dataset)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


ABLATIONS = [
    "full_grounding_debt_planner",
    "minus_debt_estimator",
    "minus_active_probe_selection",
    "minus_tactile_belief_update",
    "minus_language_vision_conflict_detector",
    "minus_safety_gate",
    "minus_probe_cost_regularizer",
]


def run_ablation():
    rows = []
    summary = []
    for seed in SEEDS:
        for episode_id in range(TEST_EPISODES_PER_SPLIT_SEED):
            ep = make_episode("combined_hard_shift", seed, episode_id)
            for ablation in ABLATIONS:
                local = None if ablation == "full_grounding_debt_planner" else ablation
                rows.append(run_episode(ep, "grounding_debt_planner", ablation=local) | {"ablation": ablation})
        print(f"ablation seed={seed} rows={len(rows)}", flush=True)
    for ablation in ABLATIONS:
        vals = [r for r in rows if r["ablation"] == ablation]
        seed_success, seed_damage, seed_cost, seed_fact = [], [], [], []
        for seed in SEEDS:
            seed_vals = [r for r in vals if int(r["seed"]) == seed]
            seed_success.append(np.mean([int(v["action_success"]) for v in seed_vals]))
            seed_damage.append(np.mean([int(v["damage"]) for v in seed_vals]))
            seed_cost.append(np.mean([float(v["probe_cost"]) for v in seed_vals]))
            seed_fact.append(np.mean([float(v["fact_accuracy"]) for v in seed_vals]))
        summary.append({
            "split": "combined_hard_shift",
            "ablation": ablation,
            "action_success": f"{np.mean(seed_success):.5f}",
            "ci95_success": f"{ci95(seed_success):.5f}",
            "damage": f"{np.mean(seed_damage):.5f}",
            "probe_cost": f"{np.mean(seed_cost):.5f}",
            "fact_accuracy": f"{np.mean(seed_fact):.5f}",
            "rows": len(vals),
        })
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    axes = {
        "language": "stress_language",
        "vision": "stress_vision",
        "tactile": "stress_tactile",
        "material": "stress_material",
        "combined": "stress_combined",
    }
    methods = ["vision_language_policy", "greedy_active_touch", "strong_tactile_then_policy", "grounding_debt_planner", "oracle_tactile_upper_bound"]
    raw = []
    summary = []
    for axis, split in axes.items():
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for seed in SEEDS:
                for episode_id in range(STRESS_EPISODES_PER_SEED):
                    ep = make_episode(split, seed, episode_id, stress=level)
                    for method in methods:
                        row = run_episode(ep, method)
                        row["stress_axis"] = axis
                        row["stress_level"] = f"{level:.1f}"
                        raw.append(row)
                print(f"stress axis={axis} level={level:.1f} seed={seed} rows={len(raw)}", flush=True)
    for axis in axes:
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for method in methods:
                vals = [r for r in raw if r["stress_axis"] == axis and r["stress_level"] == f"{level:.1f}" and r["method"] == method]
                seed_success, seed_damage, seed_cost = [], [], []
                for seed in SEEDS:
                    seed_vals = [r for r in vals if int(r["seed"]) == seed]
                    seed_success.append(np.mean([int(v["action_success"]) for v in seed_vals]))
                    seed_damage.append(np.mean([int(v["damage"]) for v in seed_vals]))
                    seed_cost.append(np.mean([float(v["probe_cost"]) for v in seed_vals]))
                summary.append({
                    "stress_axis": axis,
                    "stress_level": f"{level:.1f}",
                    "method": method,
                    "action_success": f"{np.mean(seed_success):.5f}",
                    "ci95_success": f"{ci95(seed_success):.5f}",
                    "damage": f"{np.mean(seed_damage):.5f}",
                    "probe_cost": f"{np.mean(seed_cost):.5f}",
                    "rows": len(vals),
                })
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def write_negative_cases(rows):
    failures = [r for r in rows if int(r["action_success"]) == 0]
    lessons = {
        "damage_or_probe_damage": "grounding can fail before action when tactile probing is too invasive",
        "slip_or_drop": "language and vision under-estimated grip or support needs",
        "wrong_action_parameter": "hidden tactile facts were not resolved enough to set action parameters",
    }
    out = []
    seen = set()
    for r in failures:
        key = (r["split"], r["method"], r["failure_label"], r["task"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "split": r["split"],
            "seed": r["seed"],
            "episode_id": r["episode_id"],
            "method": r["method"],
            "task": r["task"],
            "object_family": r["object_family"],
            "failure_label": r["failure_label"],
            "probes": r["probes"],
            "probe_cost": r["probe_cost"],
            "fact_accuracy": r["fact_accuracy"],
            "lesson": lessons.get(r["failure_label"], "negative case retained for audit"),
        })
        if len(out) >= 16:
            break
    write_csv(RESULTS / "negative_cases.csv", out)


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner", "action_success")
    strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "action_success")
    greedy = metric_value(metric_rows, "combined_hard_shift", "greedy_active_touch", "action_success")
    uncertainty = metric_value(metric_rows, "combined_hard_shift", "uncertainty_threshold_policy", "action_success")
    best = max(strong[0], greedy[0], uncertainty[0])
    diff_strong = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "strong_tactile_then_policy" and r["metric"] == "action_success"][0]
    damage_diff = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "strong_tactile_then_policy" and r["metric"] == "damage"][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_grounding_debt_planner"][0]
    no_update = [r for r in ablation_summary if r["ablation"] == "minus_tactile_belief_update"][0]
    no_active = [r for r in ablation_summary if r["ablation"] == "minus_active_probe_selection"][0]
    ablation_drop = float(full["action_success"]) - max(float(no_update["action_success"]), float(no_active["action_success"]))
    if prop[0] >= best + 0.06 and float(diff_strong["mean_diff"]) > 0.04 and float(damage_diff["mean_diff"]) < -0.03 and ablation_drop >= 0.04:
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows, ablation_rows, stress_raw):
    decision = terminal_decision(metric_rows, pair_rows, ablation_summary)
    prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner", "action_success")
    strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "action_success")
    greedy = metric_value(metric_rows, "combined_hard_shift", "greedy_active_touch", "action_success")
    vl = metric_value(metric_rows, "combined_hard_shift", "vision_language_policy", "action_success")
    oracle = metric_value(metric_rows, "combined_hard_shift", "oracle_tactile_upper_bound", "action_success")
    dmg_prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner", "damage")
    dmg_strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "damage")
    cost_prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner", "probe_cost")
    cost_strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "probe_cost")
    diff_strong = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "strong_tactile_then_policy" and r["metric"] == "action_success"][0]
    stress_max = [r for r in stress_summary if r["stress_axis"] == "combined" and r["stress_level"] == "1.0"]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 81 tactile_language_grounding_debt v4 rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write("Reason: local tactile-language benchmark exists, but no real tactile hardware or recognized high-fidelity benchmark is available.\n")
        f.write(f"Main rollout rows: {len(rollout_rows)}\n")
        f.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        f.write(f"Stress rollout rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("\nCombined hard-shift action success:\n")
        f.write(f"grounding_debt_planner={prop[0]:.5f} ci95={prop[1]:.5f}\n")
        f.write(f"strong_tactile_then_policy={strong[0]:.5f} ci95={strong[1]:.5f}\n")
        f.write(f"greedy_active_touch={greedy[0]:.5f} ci95={greedy[1]:.5f}\n")
        f.write(f"vision_language_policy={vl[0]:.5f} ci95={vl[1]:.5f}\n")
        f.write(f"oracle_tactile_upper_bound={oracle[0]:.5f} ci95={oracle[1]:.5f}\n")
        f.write(f"damage proposed={dmg_prop[0]:.5f}, strong_tactile={dmg_strong[0]:.5f}\n")
        f.write(f"probe_cost proposed={cost_prop[0]:.5f}, strong_tactile={cost_strong[0]:.5f}\n")
        f.write(f"paired action-success diff vs strong_tactile={diff_strong['mean_diff']} ci95={diff_strong['ci95']}\n")
        f.write("\nAblation combined_hard_shift:\n")
        for row in ablation_summary:
            f.write(f"{row['ablation']} action_success={row['action_success']} ci95={row['ci95_success']} damage={row['damage']} probe_cost={row['probe_cost']} fact_accuracy={row['fact_accuracy']}\n")
        f.write("\nCombined stress level 1.0:\n")
        for row in stress_max:
            f.write(f"{row['method']} action_success={row['action_success']} ci95={row['ci95_success']} damage={row['damage']} probe_cost={row['probe_cost']}\n")
    write_negative_cases(rollout_rows)
    return decision


def plot_outputs(metric_rows, ablation_summary, stress_summary):
    methods = METHODS
    vals = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[0] for m in methods]
    errs = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[1] for m in methods]
    colors = ["#868e96", "#adb5bd", "#74c0fc", "#4dabf7", "#f08c00", "#2f9e44", "#087f5b", "#095c4a"]
    plt.figure(figsize=(11.5, 4.8))
    plt.bar(range(len(methods)), vals, yerr=errs, color=colors, capsize=3)
    plt.xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=7)
    plt.ylim(0, 1.05)
    plt.ylabel("task success")
    plt.title("Combined hard-shift tactile-language grounding")
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_success.png", dpi=220)
    plt.close()

    damage = [metric_value(metric_rows, "combined_hard_shift", m, "damage")[0] for m in methods]
    cost = [metric_value(metric_rows, "combined_hard_shift", m, "probe_cost")[0] for m in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(11.0, 4.8))
    plt.bar(x - 0.18, damage, width=0.36, label="damage", color="#e8590c")
    plt.bar(x + 0.18, cost, width=0.36, label="probe cost", color="#1971c2")
    plt.xticks(x, [m.replace("_", "\n") for m in methods], fontsize=7)
    plt.ylabel("rate / cost")
    plt.title("Damage and tactile probe cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_damage_cost.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(range(len(ablation_summary)), [float(r["action_success"]) for r in ablation_summary], yerr=[float(r["ci95_success"]) for r in ablation_summary], color="#f08c00", capsize=3)
    plt.xticks(range(len(ablation_summary)), [r["ablation"].replace("_", "\n") for r in ablation_summary], fontsize=7)
    plt.ylim(0, 1.05)
    plt.ylabel("task success")
    plt.title("Grounding-debt ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.2, 5.0))
    for method in ["vision_language_policy", "greedy_active_touch", "strong_tactile_then_policy", "grounding_debt_planner", "oracle_tactile_upper_bound"]:
        rows = sorted([r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(r["stress_level"]) for r in rows]
        y = [float(r["action_success"]) for r in rows]
        e = [float(r["ci95_success"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, label=method)
    plt.xlabel("combined grounding stress")
    plt.ylabel("task success")
    plt.ylim(0, 1.05)
    plt.title("Language/vision/tactile/material stress sweep")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_stress_sweep.png", dpi=220)
    plt.close()


def main():
    rollout_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    decision = write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows, ablation_rows, stress_raw)
    plot_outputs(metric_rows, ablation_summary, stress_summary)
    print(f"terminal={decision}")
    print(f"main_rollouts={len(rollout_rows)} ablation_rollouts={len(ablation_rows)} stress_rollouts={len(stress_raw)}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
