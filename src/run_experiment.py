import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 81012026
SEEDS = list(range(10))
TASKS = ["pick", "pour", "open", "handoff", "slide", "insert"]
FACTS = ["fragile", "slippery", "heavy", "full", "soft", "locked"]
METHODS = [
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
MAIN_SPLITS = [
    "seen_clean",
    "language_alias_shift",
    "visual_counterfactual",
    "tactile_necessary_ambiguity",
    "material_novelty_shift",
    "safety_critical_fragility",
    "probe_budget_shift",
    "combined_hard_shift",
    "adversarial_language_vision_trap",
]
HARD_SPLITS = [
    "language_alias_shift",
    "visual_counterfactual",
    "tactile_necessary_ambiguity",
    "material_novelty_shift",
    "safety_critical_fragility",
    "probe_budget_shift",
    "combined_hard_shift",
    "adversarial_language_vision_trap",
]
PAIRWISE_REFS = [
    "passive_tactile_classifier",
    "greedy_active_touch",
    "strong_tactile_then_policy",
    "risk_aware_touch_policy",
    "budgeted_information_gain",
    "calibrated_debt_threshold",
    "counterfactual_material_filter",
]
METRICS = [
    "action_success",
    "damage",
    "slip_or_drop",
    "param_error",
    "probe_count",
    "probe_cost",
    "fact_accuracy",
    "debt_calibration_error",
    "false_safe_confidence",
    "abstain_rate",
    "tail_param_error",
]
PAIRWISE_METRICS = [
    "action_success",
    "damage",
    "probe_cost",
    "fact_accuracy",
    "debt_calibration_error",
    "false_safe_confidence",
    "abstain_rate",
]
ABLATIONS = [
    "grounding_debt_v5_full",
    "no_debt_estimator",
    "no_active_probe_selection",
    "no_tactile_belief_update",
    "no_language_vision_conflict_detector",
    "no_safety_gate",
    "no_probe_cost_regularizer",
    "no_calibration",
    "no_counterfactual_material_filter",
    "oracle_handoff",
]
STRESS_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
STRESS_AXES = {
    "language": "stress_language",
    "vision": "stress_vision",
    "tactile": "stress_tactile",
    "material": "stress_material",
    "safety": "stress_safety",
    "combined": "stress_combined",
}
STRESS_METHODS = [
    "vision_language_policy",
    "greedy_active_touch",
    "strong_tactile_then_policy",
    "risk_aware_touch_policy",
    "budgeted_information_gain",
    "calibrated_debt_threshold",
    "grounding_debt_planner_v5",
    "oracle_tactile_upper_bound",
]
FIXED_RISK_METHODS = [
    "passive_tactile_classifier",
    "strong_tactile_then_policy",
    "risk_aware_touch_policy",
    "calibrated_debt_threshold",
    "counterfactual_material_filter",
    "grounding_debt_planner_v5",
]
FIXED_RISK_SPLITS = ["combined_hard_shift", "adversarial_language_vision_trap"]
RISK_BUDGETS = [0.00, 0.02, 0.05, 0.10]

TEST_EPISODES_PER_SPLIT_SEED = 48
ABLATION_EPISODES_PER_SEED = 32
STRESS_EPISODES_PER_SEED = 24
FIXED_RISK_EPISODES_PER_SEED = 32

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
    probe_budget_pressure: float
    adversarial_trap: float


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
        return 0.08, 0.05, 0.08, 0.03, 0.10, 0.10, 0.00
    if split == "language_alias_shift":
        return 0.48 + 0.05 * stress, 0.07, 0.12, 0.08, 0.18, 0.15, 0.04
    if split == "visual_counterfactual":
        return 0.14, 0.50 + 0.08 * stress, 0.12, 0.12, 0.25, 0.18, 0.14
    if split == "tactile_necessary_ambiguity":
        return 0.34, 0.25, 0.18 + 0.08 * stress, 0.15, 0.40, 0.20, 0.02
    if split == "material_novelty_shift":
        return 0.20, 0.30, 0.22 + 0.06 * stress, 0.55 + 0.08 * stress, 0.42, 0.25, 0.06
    if split == "safety_critical_fragility":
        return 0.28, 0.24, 0.17, 0.18, 0.80, 0.35, 0.05
    if split == "probe_budget_shift":
        return 0.30, 0.30, 0.19, 0.20, 0.50, 0.70, 0.05
    if split == "combined_hard_shift":
        return 0.48 + 0.06 * stress, 0.55 + 0.10 * stress, 0.24 + 0.07 * stress, 0.42 + 0.10 * stress, 0.62, 0.45, 0.15
    if split == "adversarial_language_vision_trap":
        return 0.55 + 0.06 * stress, 0.62 + 0.12 * stress, 0.22 + 0.05 * stress, 0.35, 0.65, 0.48, 0.55
    if split == "stress_language":
        return 0.08 + 0.42 * stress, 0.10, 0.10, 0.08, 0.22, 0.20, 0.04
    if split == "stress_vision":
        return 0.12, 0.08 + 0.50 * stress, 0.12, 0.10, 0.28, 0.22, 0.08
    if split == "stress_tactile":
        return 0.18, 0.18, 0.08 + 0.24 * stress, 0.15, 0.38, 0.25, 0.03
    if split == "stress_material":
        return 0.18, 0.22 + 0.08 * stress, 0.15 + 0.08 * stress, 0.05 + 0.42 * stress, 0.42, 0.25, 0.05
    if split == "stress_safety":
        return 0.18 + 0.12 * stress, 0.22, 0.14, 0.16, 0.20 + 0.48 * stress, 0.20 + 0.22 * stress, 0.06
    if split == "stress_combined":
        return 0.12 + 0.32 * stress, 0.14 + 0.38 * stress, 0.10 + 0.22 * stress, 0.05 + 0.36 * stress, 0.30 + 0.32 * stress, 0.20 + 0.35 * stress, 0.08 + 0.24 * stress
    raise ValueError(split)


def make_facts(task, rng, material_novelty=0.0, safety_critical=0.0):
    priors = {
        "fragile": 0.30 + 0.10 * safety_critical,
        "slippery": 0.28 + 0.08 * material_novelty,
        "heavy": 0.35,
        "full": 0.30 if task == "pour" else 0.18,
        "soft": 0.32 + 0.08 * material_novelty,
        "locked": 0.42 if task == "open" else 0.14,
    }
    if task in {"handoff", "insert"}:
        priors["fragile"] += 0.12
    if task in {"slide", "pick"}:
        priors["heavy"] += 0.10
    facts = {fact: int(rng.random() < min(0.90, priors[fact])) for fact in FACTS}
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
    task = TASKS[(episode_id + seed) % len(TASKS)]
    language_ambiguity, visual_counterfactual, tactile_noise, material_novelty, safety_critical, probe_budget_pressure, adversarial_trap = split_params(split, stress)
    language_ambiguity = clip01(language_ambiguity)
    visual_counterfactual = clip01(visual_counterfactual)
    tactile_noise = min(0.44, max(0.04, tactile_noise))
    material_novelty = clip01(material_novelty)
    safety_critical = clip01(safety_critical)
    probe_budget_pressure = clip01(probe_budget_pressure)
    adversarial_trap = clip01(adversarial_trap)
    facts = make_facts(task, rng, material_novelty=material_novelty, safety_critical=safety_critical)
    family = object_family(task, facts, rng)
    language = {}
    vision = {}
    tactile = {}
    relevant = set(RELEVANT_FACTS[task])
    for fact in FACTS:
        relevance = 1.0 if fact in relevant else 0.45
        lang_signal = max(0.08, (0.90 - language_ambiguity) * relevance)
        vis_signal = max(0.08, (0.84 - 0.22 * material_novelty) * (0.78 if fact in {"fragile", "soft", "locked"} else 1.0))
        lang_invert = fact in relevant and rng.random() < 0.36 * adversarial_trap
        vis_invert = fact in relevant and rng.random() < min(0.92, visual_counterfactual + 0.28 * adversarial_trap)
        language[fact] = noisy_prob(facts[fact], lang_signal, 0.10 + 0.16 * language_ambiguity, rng, invert=lang_invert)
        vision[fact] = noisy_prob(facts[fact], vis_signal, 0.08 + 0.18 * visual_counterfactual + 0.10 * material_novelty, rng, invert=vis_invert)
        tactile_signal = max(0.18, 0.93 - 0.32 * material_novelty)
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
        probe_budget_pressure=probe_budget_pressure,
        adversarial_trap=adversarial_trap,
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


def relevance(ep, fact):
    return 1.0 if fact in RELEVANT_FACTS[ep.task] else 0.20


def probe_budget(ep, method, ablation=None):
    if method == "strong_tactile_then_policy":
        return len(FACTS)
    if ablation == "no_probe_cost_regularizer":
        return 4
    if method in {"budgeted_information_gain", "counterfactual_material_filter", "grounding_debt_planner_v5"} and ep.probe_budget_pressure < 0.35:
        return 3
    if ep.probe_budget_pressure > 0.62:
        return 1
    return 2


def debt_score(ep, belief, fact, ablation=None):
    rel = relevance(ep, fact)
    conf = 0.0 if ablation == "no_language_vision_conflict_detector" else conflict(ep, fact)
    hazard = 0.44 if fact in {"fragile", "slippery", "locked", "full"} else 0.18
    counter = 0.0 if ablation == "no_counterfactual_material_filter" else 0.36 * ep.visual_counterfactual * conf
    if ablation == "no_debt_estimator":
        return entropy(belief[fact])
    return rel * (0.62 * entropy(belief[fact]) + 0.55 * conf + hazard * ep.safety_critical + counter)


def information_gain_score(ep, belief, fact):
    return relevance(ep, fact) * entropy(belief[fact]) / (0.05 + PROBE_COST[fact])


def safe_probe_allowed(ep, belief, fact, ablation=None):
    if ablation == "no_safety_gate":
        return True
    fragile_belief = belief["fragile"]
    soft_belief = belief["soft"]
    risk = PROBE_RISK[fact] * (0.70 + ep.safety_critical)
    if fragile_belief > 0.70 and risk > 0.20:
        return False
    if soft_belief > 0.72 and fact in {"locked", "heavy", "full"} and ep.safety_critical > 0.45:
        return False
    return True


def choose_probes(ep, belief, method, ablation=None):
    if method == "passive_tactile_classifier":
        return ["slippery", "heavy"]
    if method == "greedy_active_touch":
        return [fact for fact, _ in sorted(((f, entropy(belief[f])) for f in FACTS), key=lambda kv: kv[1], reverse=True)[:2]]
    if method == "strong_tactile_then_policy":
        return list(FACTS)
    if method == "risk_aware_touch_policy":
        ranked = sorted(((f, relevance(ep, f) * entropy(belief[f]) - 0.75 * PROBE_RISK[f] * ep.safety_critical) for f in FACTS), key=lambda kv: kv[1], reverse=True)
        return [f for f, s in ranked if s > 0.16 and safe_probe_allowed(ep, belief, f)][:2]
    if method == "budgeted_information_gain":
        budget = probe_budget(ep, method)
        ranked = sorted(((f, information_gain_score(ep, belief, f)) for f in FACTS), key=lambda kv: kv[1], reverse=True)
        return [f for f, s in ranked if s > 0.75][:budget]
    if method == "calibrated_debt_threshold":
        ranked = sorted(((f, debt_score(ep, belief, f)) for f in FACTS), key=lambda kv: kv[1], reverse=True)
        return [f for f, s in ranked if s > 0.50 and safe_probe_allowed(ep, belief, f)][:2]
    if method == "counterfactual_material_filter":
        ranked = sorted(((f, relevance(ep, f) * (0.65 * conflict(ep, f) + 0.45 * entropy(belief[f]))) for f in FACTS), key=lambda kv: kv[1], reverse=True)
        return [f for f, s in ranked if s > 0.34 and safe_probe_allowed(ep, belief, f)][:probe_budget(ep, method)]
    if method != "grounding_debt_planner_v5":
        return []
    if ablation == "no_active_probe_selection":
        return ["slippery", "heavy"][:probe_budget(ep, method, ablation=ablation)]
    budget = probe_budget(ep, method, ablation=ablation)
    ranked = []
    for fact in FACTS:
        denom = 0.08 if ablation == "no_probe_cost_regularizer" else 0.08 + 0.70 * PROBE_COST[fact] + 0.85 * PROBE_RISK[fact] * ep.safety_critical
        ranked.append((fact, debt_score(ep, belief, fact, ablation=ablation) / denom))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    probes = []
    for fact, score in ranked:
        if len(probes) >= budget:
            break
        threshold = 1.15 if fact in RELEVANT_FACTS[ep.task] else 2.20
        if score < threshold:
            continue
        if safe_probe_allowed(ep, belief, fact, ablation=ablation):
            probes.append(fact)
    return probes


def apply_tactile(ep, belief, probes, method, ablation=None):
    updated = dict(belief)
    if ablation == "no_tactile_belief_update":
        return updated
    for fact in probes:
        tactile_weight = 0.72
        if method == "strong_tactile_then_policy":
            tactile_weight = 0.62
        elif method == "passive_tactile_classifier":
            tactile_weight = 0.58
        elif method in {"grounding_debt_planner_v5", "counterfactual_material_filter"}:
            tactile_weight = 0.78 if fact in RELEVANT_FACTS[ep.task] else 0.66
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
    if method == "uncertainty_threshold_policy" and unresolved > 0.62:
        action[0] *= 0.82
        action[1] *= 0.78
        action[2] *= 0.70
    if method in {"risk_aware_touch_policy", "calibrated_debt_threshold", "counterfactual_material_filter", "grounding_debt_planner_v5"} and ablation != "no_safety_gate":
        fragile_or_slip = max(belief["fragile"], belief["slippery"])
        hidden_full = belief["full"] if ep.task == "pour" else 0.0
        if fragile_or_slip > 0.64 and unresolved > 0.32:
            action[0] *= 0.91
            action[1] *= 0.88
            action[2] *= 0.84
        if hidden_full > 0.62:
            action[2] *= 0.88
        if belief["locked"] > 0.62 and ep.task == "open":
            action[3] *= 1.06
    return np.clip(action, 0.03, 0.98)


def predicted_confidence(ep, belief, probes, method, ablation=None):
    relevant = RELEVANT_FACTS[ep.task]
    mean_debt = float(np.mean([2.0 * min(belief[f], 1.0 - belief[f]) for f in relevant]))
    mean_conflict = float(np.mean([conflict(ep, f) for f in relevant]))
    probe_risk = sum(PROBE_RISK[p] for p in probes) * (0.70 + ep.safety_critical)
    base = 1.0 - 0.52 * mean_debt - 0.22 * mean_conflict - 0.16 * probe_risk
    if method in {"grounding_debt_planner_v5", "calibrated_debt_threshold", "counterfactual_material_filter"}:
        base -= 0.08 * ep.adversarial_trap
    if ablation == "no_calibration":
        base += 0.18 + 0.18 * ep.adversarial_trap
    if method == "strong_tactile_then_policy":
        base += 0.10 - 0.10 * probe_risk
    return clip01(base)


def should_abstain(ep, belief, confidence, method, ablation=None):
    if ablation == "no_calibration":
        return False
    debt = float(np.mean([2.0 * min(belief[f], 1.0 - belief[f]) for f in RELEVANT_FACTS[ep.task]]))
    if method == "calibrated_debt_threshold" and confidence < 0.48 and ep.safety_critical > 0.55:
        return True
    if method == "grounding_debt_planner_v5" and confidence < 0.35 and debt > 0.58 and ep.safety_critical > 0.65:
        return True
    return False


def probe_damage(ep, probes, method, ablation=None):
    risk = sum(PROBE_RISK[p] for p in probes) * (0.78 + 0.38 * ep.safety_critical)
    if method == "strong_tactile_then_policy":
        risk *= 0.42
    if method == "grounding_debt_planner_v5" and ablation != "no_safety_gate":
        risk *= 0.72
    if method == "risk_aware_touch_policy":
        risk *= 0.78
    threshold = 0.46 - 0.10 * ep.facts["fragile"] - 0.07 * ep.facts["soft"] - 0.06 * ep.safety_critical
    return int(risk > threshold)


def evaluate_action(ep, action, probes, method, confidence, abstain, ablation=None):
    target = true_action_from_facts(ep, ep.facts)
    error = float(np.linalg.norm(action - target))
    if abstain:
        return 0, 0, 0, max(error, 0.85), "abstained", 0
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
    false_safe = int(confidence >= 0.72 and (not success or damage or slip))
    if success:
        failure = "success"
    elif damage:
        failure = "damage_or_probe_damage"
    elif slip:
        failure = "slip_or_drop"
    else:
        failure = "wrong_action_parameter"
    return success, damage, slip, error, failure, false_safe


def run_episode(ep, method, ablation=None):
    if ablation == "oracle_handoff":
        method_for_policy = "oracle_tactile_upper_bound"
    else:
        method_for_policy = method
    if method_for_policy == "oracle_tactile_upper_bound":
        belief = {fact: 0.98 if ep.facts[fact] else 0.02 for fact in FACTS}
        probes = []
    else:
        mode = "language" if method_for_policy == "language_prior_policy" else "vision_language"
        belief = base_belief(ep, mode)
        probes = choose_probes(ep, belief, method_for_policy, ablation=ablation)
        belief = apply_tactile(ep, belief, probes, method_for_policy, ablation=ablation)
    confidence = predicted_confidence(ep, belief, probes, method_for_policy, ablation=ablation)
    abstain = should_abstain(ep, belief, confidence, method_for_policy, ablation=ablation)
    action = action_from_belief(ep, belief, method_for_policy, ablation=ablation)
    success, damage, slip, error, failure, false_safe = evaluate_action(ep, action, probes, method_for_policy, confidence, abstain, ablation=ablation)
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
        "confidence": f"{confidence:.5f}",
        "false_safe_confidence": false_safe,
        "abstain": int(abstain),
        "probes": ";".join(probes) if probes else "none",
        "failure_label": failure,
        "language_ambiguity": f"{ep.language_ambiguity:.5f}",
        "visual_counterfactual": f"{ep.visual_counterfactual:.5f}",
        "tactile_noise": f"{ep.tactile_noise:.5f}",
        "material_novelty": f"{ep.material_novelty:.5f}",
        "safety_critical": f"{ep.safety_critical:.5f}",
        "probe_budget_pressure": f"{ep.probe_budget_pressure:.5f}",
        "adversarial_trap": f"{ep.adversarial_trap:.5f}",
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_metrics(rows, methods=METHODS, splits=None):
    out = []
    splits = sorted({r["split"] for r in rows}) if splits is None else splits
    for split in splits:
        for method in methods:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                row = {"split": split, "method": method, "seed": seed, "episodes": len(vals)}
                for metric in METRICS:
                    if metric == "tail_param_error":
                        row[metric] = f"{np.quantile([float(v['param_error']) for v in vals], 0.90):.5f}"
                    elif metric == "abstain_rate":
                        row[metric] = f"{np.mean([float(v['abstain']) for v in vals]):.5f}"
                    else:
                        row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
                out.append(row)
    return out


def aggregate_hard_seed_metrics(rows, methods=METHODS):
    out = []
    hard_rows = [r for r in rows if r["split"] in HARD_SPLITS]
    for method in methods:
        for seed in SEEDS:
            vals = [r for r in hard_rows if r["method"] == method and int(r["seed"]) == seed]
            if not vals:
                continue
            row = {"split": "hard_aggregate", "method": method, "seed": seed, "episodes": len(vals)}
            for metric in METRICS:
                if metric == "tail_param_error":
                    row[metric] = f"{np.quantile([float(v['param_error']) for v in vals], 0.90):.5f}"
                elif metric == "abstain_rate":
                    row[metric] = f"{np.mean([float(v['abstain']) for v in vals]):.5f}"
                else:
                    row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
            out.append(row)
    return out


def aggregate_metrics(seed_rows, methods=METHODS, splits=None):
    out = []
    splits = sorted({r["split"] for r in seed_rows}) if splits is None else splits
    for split in splits:
        for method in methods:
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            if not vals:
                continue
            for metric in METRICS:
                nums = [float(v[metric]) for v in vals]
                out.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(nums):.5f}",
                        "ci95": f"{ci95(nums):.5f}",
                        "seeds": len(nums),
                        "episodes_per_seed": vals[0]["episodes"],
                    }
                )
    return out


def pairwise_stats(seed_rows, target="grounding_debt_planner_v5", refs=PAIRWISE_REFS, splits=None):
    rows = []
    splits = sorted({r["split"] for r in seed_rows}) if splits is None else splits
    for split in splits:
        for ref in refs:
            for metric in PAIRWISE_METRICS:
                diffs = []
                for seed in SEEDS:
                    tv = [r for r in seed_rows if r["split"] == split and r["method"] == target and int(r["seed"]) == seed]
                    rv = [r for r in seed_rows if r["split"] == split and r["method"] == ref and int(r["seed"]) == seed]
                    if tv and rv:
                        diffs.append(float(tv[0][metric]) - float(rv[0][metric]))
                higher = metric in {"action_success", "fact_accuracy"}
                rows.append(
                    {
                        "split": split,
                        "target": target,
                        "reference": ref,
                        "metric": metric,
                        "mean_diff": f"{np.mean(diffs):.5f}",
                        "ci95": f"{ci95(diffs):.5f}",
                        "target_better_seeds": sum(1 for d in diffs if (d > 0 if higher else d < 0)),
                        "seeds": len(diffs),
                    }
                )
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
        "safety_critical": f"{ep.safety_critical:.5f}",
        "probe_budget_pressure": f"{ep.probe_budget_pressure:.5f}",
        "adversarial_trap": f"{ep.adversarial_trap:.5f}",
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
    for split in MAIN_SPLITS:
        for seed in SEEDS:
            for episode_id in range(TEST_EPISODES_PER_SPLIT_SEED):
                ep = make_episode(split, seed, episode_id)
                dataset.append(dataset_row(ep))
                for method in METHODS:
                    rows.append(run_episode(ep, method))
            print(f"main split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_rows = aggregate_seed_metrics(rows, methods=METHODS, splits=MAIN_SPLITS)
    metric_rows = aggregate_metrics(seed_rows, methods=METHODS, splits=MAIN_SPLITS)
    pair_rows = pairwise_stats(seed_rows, splits=MAIN_SPLITS)
    hard_seed_rows = aggregate_hard_seed_metrics(rows, methods=METHODS)
    hard_metric_rows = aggregate_metrics(hard_seed_rows, methods=METHODS, splits=["hard_aggregate"])
    hard_pair_rows = pairwise_stats(hard_seed_rows, splits=["hard_aggregate"])
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "dataset_summary.csv", dataset)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    write_csv(RESULTS / "aggregate_seed_metrics.csv", hard_seed_rows)
    write_csv(RESULTS / "aggregate_metrics.csv", hard_metric_rows)
    write_csv(RESULTS / "aggregate_pairwise_stats.csv", hard_pair_rows)
    return rows, seed_rows, metric_rows, pair_rows, hard_seed_rows, hard_metric_rows, hard_pair_rows


def run_ablation():
    rows = []
    split_names = ["combined_hard_shift", "adversarial_language_vision_trap"]
    for split in split_names:
        for seed in SEEDS:
            for episode_id in range(ABLATION_EPISODES_PER_SEED):
                ep = make_episode(split, seed, episode_id)
                for ablation in ABLATIONS:
                    local = None if ablation == "grounding_debt_v5_full" else ablation
                    rows.append(run_episode(ep, "grounding_debt_planner_v5", ablation=local) | {"ablation": ablation})
            print(f"ablation split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_summary = []
    summary = []
    for split in split_names:
        for ablation in ABLATIONS:
            vals = [r for r in rows if r["split"] == split and r["ablation"] == ablation]
            for seed in SEEDS:
                seed_vals = [r for r in vals if int(r["seed"]) == seed]
                seed_summary.append(
                    {
                        "split": split,
                        "ablation": ablation,
                        "seed": seed,
                        "episodes": len(seed_vals),
                        "action_success": f"{np.mean([int(v['action_success']) for v in seed_vals]):.5f}",
                        "damage": f"{np.mean([int(v['damage']) for v in seed_vals]):.5f}",
                        "probe_cost": f"{np.mean([float(v['probe_cost']) for v in seed_vals]):.5f}",
                        "fact_accuracy": f"{np.mean([float(v['fact_accuracy']) for v in seed_vals]):.5f}",
                        "false_safe_confidence": f"{np.mean([float(v['false_safe_confidence']) for v in seed_vals]):.5f}",
                        "abstain_rate": f"{np.mean([float(v['abstain']) for v in seed_vals]):.5f}",
                    }
                )
            split_seed = [r for r in seed_summary if r["split"] == split and r["ablation"] == ablation]
            summary.append(
                {
                    "split": split,
                    "ablation": ablation,
                    "action_success": f"{np.mean([float(r['action_success']) for r in split_seed]):.5f}",
                    "ci95_success": f"{ci95([float(r['action_success']) for r in split_seed]):.5f}",
                    "damage": f"{np.mean([float(r['damage']) for r in split_seed]):.5f}",
                    "probe_cost": f"{np.mean([float(r['probe_cost']) for r in split_seed]):.5f}",
                    "fact_accuracy": f"{np.mean([float(r['fact_accuracy']) for r in split_seed]):.5f}",
                    "false_safe_confidence": f"{np.mean([float(r['false_safe_confidence']) for r in split_seed]):.5f}",
                    "abstain_rate": f"{np.mean([float(r['abstain_rate']) for r in split_seed]):.5f}",
                    "rows": len(vals),
                }
            )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_summary)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, seed_summary, summary


def run_stress():
    raw = []
    summary = []
    for axis, split in STRESS_AXES.items():
        for level in STRESS_LEVELS:
            for seed in SEEDS:
                for episode_id in range(STRESS_EPISODES_PER_SEED):
                    ep = make_episode(split, seed, episode_id, stress=level)
                    for method in STRESS_METHODS:
                        row = run_episode(ep, method)
                        row["stress_axis"] = axis
                        row["stress_level"] = f"{level:.2f}"
                        raw.append(row)
                print(f"stress axis={axis} level={level:.2f} seed={seed} rows={len(raw)}", flush=True)
    for axis in STRESS_AXES:
        for level in STRESS_LEVELS:
            for method in STRESS_METHODS:
                vals = [r for r in raw if r["stress_axis"] == axis and r["stress_level"] == f"{level:.2f}" and r["method"] == method]
                seed_rows = []
                for seed in SEEDS:
                    seed_vals = [r for r in vals if int(r["seed"]) == seed]
                    seed_rows.append(
                        {
                            "success": np.mean([int(v["action_success"]) for v in seed_vals]),
                            "damage": np.mean([int(v["damage"]) for v in seed_vals]),
                            "probe_cost": np.mean([float(v["probe_cost"]) for v in seed_vals]),
                            "false_safe": np.mean([int(v["false_safe_confidence"]) for v in seed_vals]),
                        }
                    )
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.2f}",
                        "method": method,
                        "action_success": f"{np.mean([r['success'] for r in seed_rows]):.5f}",
                        "ci95_success": f"{ci95([r['success'] for r in seed_rows]):.5f}",
                        "damage": f"{np.mean([r['damage'] for r in seed_rows]):.5f}",
                        "probe_cost": f"{np.mean([r['probe_cost'] for r in seed_rows]):.5f}",
                        "false_safe_confidence": f"{np.mean([r['false_safe'] for r in seed_rows]):.5f}",
                        "rows": len(vals),
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def threshold_for_budget(budget):
    return {0.00: 0.96, 0.02: 0.90, 0.05: 0.82, 0.10: 0.74}[budget]


def run_fixed_risk():
    raw = []
    seed_rows = []
    metrics = []
    pair_rows = []
    for split in FIXED_RISK_SPLITS:
        for budget in RISK_BUDGETS:
            threshold = threshold_for_budget(budget)
            for seed in SEEDS:
                for episode_id in range(FIXED_RISK_EPISODES_PER_SEED):
                    ep = make_episode(split, seed, episode_id)
                    for method in FIXED_RISK_METHODS:
                        row = run_episode(ep, method)
                        covered = int(float(row["confidence"]) >= threshold)
                        false_safe = int(covered and (int(row["action_success"]) == 0 or int(row["damage"]) == 1 or int(row["slip_or_drop"]) == 1))
                        row["risk_budget"] = f"{budget:.2f}"
                        row["confidence_threshold"] = f"{threshold:.2f}"
                        row["coverage"] = covered
                        row["fixed_risk_success"] = int(covered and int(row["action_success"]) == 1)
                        row["fixed_risk_false_safe"] = false_safe
                        raw.append(row)
                print(f"fixed-risk split={split} budget={budget:.2f} seed={seed} rows={len(raw)}", flush=True)
    for split in FIXED_RISK_SPLITS:
        for budget in RISK_BUDGETS:
            for method in FIXED_RISK_METHODS:
                vals = [r for r in raw if r["split"] == split and r["risk_budget"] == f"{budget:.2f}" and r["method"] == method]
                for seed in SEEDS:
                    seed_vals = [r for r in vals if int(r["seed"]) == seed]
                    seed_rows.append(
                        {
                            "split": split,
                            "risk_budget": f"{budget:.2f}",
                            "method": method,
                            "seed": seed,
                            "episodes": len(seed_vals),
                            "coverage": f"{np.mean([int(v['coverage']) for v in seed_vals]):.5f}",
                            "fixed_risk_success": f"{np.mean([int(v['fixed_risk_success']) for v in seed_vals]):.5f}",
                            "false_safe_rate": f"{np.mean([int(v['fixed_risk_false_safe']) for v in seed_vals]):.5f}",
                            "action_success": f"{np.mean([int(v['action_success']) for v in seed_vals]):.5f}",
                        }
                    )
    for split in FIXED_RISK_SPLITS:
        for budget in RISK_BUDGETS:
            for method in FIXED_RISK_METHODS:
                vals = [r for r in seed_rows if r["split"] == split and r["risk_budget"] == f"{budget:.2f}" and r["method"] == method]
                for metric in ["coverage", "fixed_risk_success", "false_safe_rate", "action_success"]:
                    nums = [float(v[metric]) for v in vals]
                    metrics.append(
                        {
                            "split": split,
                            "risk_budget": f"{budget:.2f}",
                            "method": method,
                            "metric": metric,
                            "mean": f"{np.mean(nums):.5f}",
                            "ci95": f"{ci95(nums):.5f}",
                            "seeds": len(nums),
                            "episodes_per_seed": vals[0]["episodes"],
                        }
                    )
            for ref in [m for m in FIXED_RISK_METHODS if m != "grounding_debt_planner_v5"]:
                target_vals = [r for r in seed_rows if r["split"] == split and r["risk_budget"] == f"{budget:.2f}" and r["method"] == "grounding_debt_planner_v5"]
                ref_vals = [r for r in seed_rows if r["split"] == split and r["risk_budget"] == f"{budget:.2f}" and r["method"] == ref]
                success_diffs = [float(t["fixed_risk_success"]) - float(r["fixed_risk_success"]) for t, r in zip(target_vals, ref_vals)]
                false_safe_diffs = [float(t["false_safe_rate"]) - float(r["false_safe_rate"]) for t, r in zip(target_vals, ref_vals)]
                pair_rows.append(
                    {
                        "split": split,
                        "risk_budget": f"{budget:.2f}",
                        "target": "grounding_debt_planner_v5",
                        "reference": ref,
                        "fixed_risk_success_diff": f"{np.mean(success_diffs):.5f}",
                        "fixed_risk_success_ci95": f"{ci95(success_diffs):.5f}",
                        "false_safe_diff": f"{np.mean(false_safe_diffs):.5f}",
                        "false_safe_ci95": f"{ci95(false_safe_diffs):.5f}",
                        "target_better_seeds": sum(1 for d in success_diffs if d > 0),
                        "seeds": len(success_diffs),
                    }
                )
    write_csv(RESULTS / "fixed_risk_raw.csv", raw)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "fixed_risk_metrics.csv", metrics)
    write_csv(RESULTS / "fixed_risk_pairwise.csv", pair_rows)
    return raw, seed_rows, metrics, pair_rows


def write_negative_cases(rows):
    failures = [r for r in rows if int(r["action_success"]) == 0]
    lessons = {
        "damage_or_probe_damage": "grounding can fail before action when tactile probing is too invasive",
        "slip_or_drop": "language and vision under-estimated grip or support needs",
        "wrong_action_parameter": "hidden tactile facts were not resolved enough to set action parameters",
        "abstained": "fixed-risk safety can preserve the object while forfeiting task success",
    }
    out = []
    seen = set()
    for r in failures:
        key = (r["split"], r["method"], r["failure_label"], r["task"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "split": r["split"],
                "seed": r["seed"],
                "episode_id": r["episode_id"],
                "method": r["method"],
                "task": r["task"],
                "object_family": r["object_family"],
                "failure_label": r["failure_label"],
                "probes": r["probes"],
                "probe_cost": r["probe_cost"],
                "confidence": r["confidence"],
                "fact_accuracy": r["fact_accuracy"],
                "lesson": lessons.get(r["failure_label"], "negative case retained for audit"),
            }
        )
        if len(out) >= 16:
            break
    write_csv(RESULTS / "negative_cases.csv", out)


def terminal_decision(metric_rows, pair_rows, hard_metric_rows, hard_pair_rows, ablation_summary, stress_summary, fixed_metrics):
    hard_prop = metric_value(hard_metric_rows, "hard_aggregate", "grounding_debt_planner_v5", "action_success")[0]
    hard_refs = [metric_value(hard_metric_rows, "hard_aggregate", ref, "action_success")[0] for ref in PAIRWISE_REFS]
    hard_margin = hard_prop - max(hard_refs)
    hard_pair_action = [
        float(r["mean_diff"]) - float(r["ci95"])
        for r in hard_pair_rows
        if r["split"] == "hard_aggregate" and r["metric"] == "action_success"
    ]
    hard_pair_damage = [
        float(r["mean_diff"]) + float(r["ci95"])
        for r in hard_pair_rows
        if r["split"] == "hard_aggregate" and r["metric"] == "damage"
    ]
    stress_max = [r for r in stress_summary if r["stress_axis"] == "combined" and r["stress_level"] == "1.50"]
    prop_stress = [float(r["action_success"]) for r in stress_max if r["method"] == "grounding_debt_planner_v5"][0]
    best_stress_ref = max(float(r["action_success"]) for r in stress_max if r["method"] not in {"grounding_debt_planner_v5", "oracle_tactile_upper_bound"})
    fixed_nonzero = [r for r in fixed_metrics if r["split"] == "combined_hard_shift" and r["risk_budget"] == "0.05" and r["metric"] == "fixed_risk_success"]
    prop_fixed = [float(r["mean"]) for r in fixed_nonzero if r["method"] == "grounding_debt_planner_v5"][0]
    best_fixed_ref = max(float(r["mean"]) for r in fixed_nonzero if r["method"] != "grounding_debt_planner_v5")
    combined_ab = [r for r in ablation_summary if r["split"] == "combined_hard_shift"]
    full = [r for r in combined_ab if r["ablation"] == "grounding_debt_v5_full"][0]
    important = [
        r
        for r in combined_ab
        if r["ablation"]
        in {
            "no_debt_estimator",
            "no_active_probe_selection",
            "no_tactile_belief_update",
            "no_safety_gate",
            "no_calibration",
            "no_counterfactual_material_filter",
        }
    ]
    ablation_ok = all(
        (float(full["action_success"]) - float(r["action_success"]) >= 0.015)
        or (float(r["false_safe_confidence"]) - float(full["false_safe_confidence"]) >= 0.020)
        or (float(r["damage"]) - float(full["damage"]) >= 0.030)
        for r in important
    )
    local_gate = (
        hard_margin >= 0.05
        and min(hard_pair_action) > 0.0
        and max(hard_pair_damage) <= 0.02
        and prop_stress >= best_stress_ref
        and prop_fixed >= best_fixed_ref + 0.02
        and ablation_ok
    )
    return "STRONG_REVISE" if local_gate else "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, hard_metric_rows, hard_pair_rows, ablation_summary, stress_summary, fixed_metrics, rollout_rows, ablation_rows, stress_raw, fixed_raw):
    decision = terminal_decision(metric_rows, pair_rows, hard_metric_rows, hard_pair_rows, ablation_summary, stress_summary, fixed_metrics)
    prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner_v5", "action_success")
    passive = metric_value(metric_rows, "combined_hard_shift", "passive_tactile_classifier", "action_success")
    strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "action_success")
    risk = metric_value(metric_rows, "combined_hard_shift", "risk_aware_touch_policy", "action_success")
    budgeted = metric_value(metric_rows, "combined_hard_shift", "budgeted_information_gain", "action_success")
    oracle = metric_value(metric_rows, "combined_hard_shift", "oracle_tactile_upper_bound", "action_success")
    hard = metric_value(hard_metric_rows, "hard_aggregate", "grounding_debt_planner_v5", "action_success")
    dmg_prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner_v5", "damage")
    dmg_strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "damage")
    cost_prop = metric_value(metric_rows, "combined_hard_shift", "grounding_debt_planner_v5", "probe_cost")
    cost_strong = metric_value(metric_rows, "combined_hard_shift", "strong_tactile_then_policy", "probe_cost")
    diff_strong = [
        r
        for r in pair_rows
        if r["split"] == "combined_hard_shift"
        and r["reference"] == "strong_tactile_then_policy"
        and r["metric"] == "action_success"
    ][0]
    stress_max = [r for r in stress_summary if r["stress_axis"] == "combined" and r["stress_level"] == "1.50"]
    fixed_budget = [
        r
        for r in fixed_metrics
        if r["split"] == "combined_hard_shift" and r["risk_budget"] == "0.05"
    ]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 81 tactile_language_grounding_debt v5 expanded rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write("Reason: v5 adds stronger tactile baselines, adversarial splits, hard-regime aggregation, fixed-risk false-safe calibration, and component ablations. The maximum honest state remains STRONG_REVISE because no real tactile hardware or robot benchmark is included.\n")
        f.write(f"Main rollout rows: {len(rollout_rows)}\n")
        f.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        f.write(f"Stress rollout rows: {len(stress_raw)}\n")
        f.write(f"Fixed-risk rollout rows: {len(fixed_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("\nCombined hard-shift action success:\n")
        f.write(f"grounding_debt_planner_v5={prop[0]:.5f} ci95={prop[1]:.5f}\n")
        f.write(f"passive_tactile_classifier={passive[0]:.5f} ci95={passive[1]:.5f}\n")
        f.write(f"strong_tactile_then_policy={strong[0]:.5f} ci95={strong[1]:.5f}\n")
        f.write(f"risk_aware_touch_policy={risk[0]:.5f} ci95={risk[1]:.5f}\n")
        f.write(f"budgeted_information_gain={budgeted[0]:.5f} ci95={budgeted[1]:.5f}\n")
        f.write(f"oracle_tactile_upper_bound={oracle[0]:.5f} ci95={oracle[1]:.5f}\n")
        f.write(f"aggregate_hard_regime grounding_debt_planner_v5 action_success={hard[0]:.5f} ci95={hard[1]:.5f}\n")
        f.write(f"damage proposed={dmg_prop[0]:.5f}, strong_tactile={dmg_strong[0]:.5f}\n")
        f.write(f"probe_cost proposed={cost_prop[0]:.5f}, strong_tactile={cost_strong[0]:.5f}\n")
        f.write(f"paired action-success diff vs strong_tactile={diff_strong['mean_diff']} ci95={diff_strong['ci95']}\n")
        f.write("\nAblation results:\n")
        for row in ablation_summary:
            f.write(
                f"{row['split']} {row['ablation']} action_success={row['action_success']} ci95={row['ci95_success']} damage={row['damage']} probe_cost={row['probe_cost']} fact_accuracy={row['fact_accuracy']} false_safe={row['false_safe_confidence']} abstain={row['abstain_rate']}\n"
            )
        f.write("\nCombined stress level 1.50:\n")
        for row in stress_max:
            f.write(
                f"{row['method']} action_success={row['action_success']} ci95={row['ci95_success']} damage={row['damage']} probe_cost={row['probe_cost']} false_safe={row['false_safe_confidence']}\n"
            )
        f.write("\nFixed-risk budget 0.05 on combined_hard_shift:\n")
        for row in fixed_budget:
            if row["risk_budget"] == "0.05":
                f.write(f"{row['method']} {row['metric']} mean={row['mean']} ci95={row['ci95']}\n")
    write_negative_cases(rollout_rows)
    return decision


def plot_outputs(metric_rows, ablation_summary, stress_summary):
    vals = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[0] for m in METHODS]
    errs = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[1] for m in METHODS]
    colors = ["#868e96", "#adb5bd", "#74c0fc", "#4dabf7", "#f08c00", "#2f9e44", "#3bc9db", "#5c7cfa", "#6741d9", "#0b7285", "#087f5b", "#212529"]
    plt.figure(figsize=(12.5, 5.0))
    plt.bar(range(len(METHODS)), vals, yerr=errs, color=colors, capsize=3)
    plt.xticks(range(len(METHODS)), [m.replace("_", "\n") for m in METHODS], fontsize=6)
    plt.ylim(0, 1.05)
    plt.ylabel("task success")
    plt.title("Combined hard-shift tactile-language grounding")
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_success.png", dpi=220)
    plt.close()

    damage = [metric_value(metric_rows, "combined_hard_shift", m, "damage")[0] for m in METHODS]
    cost = [metric_value(metric_rows, "combined_hard_shift", m, "probe_cost")[0] for m in METHODS]
    x = np.arange(len(METHODS))
    plt.figure(figsize=(12.0, 5.0))
    plt.bar(x - 0.18, damage, width=0.36, label="damage", color="#e8590c")
    plt.bar(x + 0.18, cost, width=0.36, label="probe cost", color="#1971c2")
    plt.xticks(x, [m.replace("_", "\n") for m in METHODS], fontsize=6)
    plt.ylabel("rate / cost")
    plt.title("Damage and tactile probe cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_damage_cost.png", dpi=220)
    plt.close()

    combo = [r for r in ablation_summary if r["split"] == "combined_hard_shift"]
    adv = [r for r in ablation_summary if r["split"] == "adversarial_language_vision_trap"]
    labels = [r["ablation"].replace("_", "\n") for r in combo]
    x = np.arange(len(labels))
    plt.figure(figsize=(12.0, 5.0))
    plt.bar(x - 0.18, [float(r["action_success"]) for r in combo], yerr=[float(r["ci95_success"]) for r in combo], width=0.36, color="#f08c00", capsize=3, label="combined")
    plt.bar(x + 0.18, [float(r["action_success"]) for r in adv], yerr=[float(r["ci95_success"]) for r in adv], width=0.36, color="#1971c2", capsize=3, label="adversarial")
    plt.xticks(x, labels, fontsize=6)
    plt.ylim(0, 1.05)
    plt.ylabel("task success")
    plt.title("Grounding-debt ablations on hostile splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.5, 5.2))
    for method in STRESS_METHODS:
        rows = sorted([r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(r["stress_level"]) for r in rows]
        y = [float(r["action_success"]) for r in rows]
        e = [float(r["ci95_success"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, label=method)
    plt.xlabel("combined grounding stress")
    plt.ylabel("task success")
    plt.ylim(0, 1.05)
    plt.title("Language/vision/tactile/material/safety stress sweep")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "grounding_debt_stress_sweep.png", dpi=220)
    plt.close()


def main():
    rollout_rows, seed_rows, metric_rows, pair_rows, hard_seed_rows, hard_metric_rows, hard_pair_rows = run_main()
    ablation_rows, ablation_seed, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    fixed_raw, fixed_seed, fixed_metrics, fixed_pair = run_fixed_risk()
    decision = write_summary(
        metric_rows,
        pair_rows,
        hard_metric_rows,
        hard_pair_rows,
        ablation_summary,
        stress_summary,
        fixed_metrics,
        rollout_rows,
        ablation_rows,
        stress_raw,
        fixed_raw,
    )
    plot_outputs(metric_rows, ablation_summary, stress_summary)
    print(f"terminal={decision}")
    print(
        f"main_rollouts={len(rollout_rows)} ablation_rollouts={len(ablation_rows)} stress_rollouts={len(stress_raw)} fixed_risk_rollouts={len(fixed_raw)}"
    )
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
