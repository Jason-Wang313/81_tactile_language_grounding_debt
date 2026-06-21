import csv
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOWNLOADS_PDF = Path.home() / "Downloads" / "81.pdf"
DESKTOP_PDF = Path.home() / "Desktop" / "81.pdf"


EXPECTED_COUNTS = {
    "rollouts.csv": 51840,
    "dataset_summary.csv": 4320,
    "raw_seed_metrics.csv": 1080,
    "metrics.csv": 1188,
    "pairwise_stats.csv": 441,
    "aggregate_seed_metrics.csv": 120,
    "aggregate_metrics.csv": 132,
    "aggregate_pairwise_stats.csv": 49,
    "ablation_rollouts.csv": 6400,
    "ablation_seed_metrics.csv": 200,
    "ablation_metrics.csv": 20,
    "stress_sweep_raw.csv": 80640,
    "stress_sweep.csv": 336,
    "fixed_risk_raw.csv": 15360,
    "fixed_risk_seed_metrics.csv": 480,
    "fixed_risk_metrics.csv": 192,
    "fixed_risk_pairwise.csv": 40,
    "negative_cases.csv": 16,
}


def count_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(message):
    raise SystemExit(message)


def main():
    for name, expected in EXPECTED_COUNTS.items():
        actual = count_rows(RESULTS / name)
        if actual != expected:
            fail(f"{name}: expected {expected}, got {actual}")
    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    for token in [
        "Paper 81 tactile_language_grounding_debt v5 expanded rebuild",
        "Terminal recommendation: KILL_ARCHIVE",
        "Fixed-risk budget 0.05",
        "grounding_debt_planner_v5",
    ]:
        if token not in summary:
            fail(f"summary missing {token}")
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    for token in [
        r"colorlinks=false",
        r"citebordercolor={0 1 0}",
        r"linkbordercolor={1 0.55 0}",
        r"urlbordercolor={0 0.45 1}",
        r"\appendix",
    ]:
        if token not in tex:
            fail(f"main.tex missing {token}")
    log_path = PAPER / "main.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="ignore")
        bad_patterns = [
            r"LaTeX Warning: Citation .* undefined",
            r"LaTeX Warning: Reference .* undefined",
            r"There were undefined references",
            r"Package natbib Warning",
        ]
        for pattern in bad_patterns:
            if re.search(pattern, log):
                fail(f"LaTeX log has unresolved issue: {pattern}")
    if not DOWNLOADS_PDF.exists():
        fail(f"missing Downloads PDF: {DOWNLOADS_PDF}")
    if DESKTOP_PDF.exists():
        fail(f"Desktop PDF should not exist: {DESKTOP_PDF}")
    reader = PdfReader(str(DOWNLOADS_PDF))
    pages = len(reader.pages)
    if pages < 25:
        fail(f"expected at least 25 pages, got {pages}")
    digest = sha256(DOWNLOADS_PDF)
    print(f"validated Paper 81 artifacts: pages={pages}, sha256={digest}")


if __name__ == "__main__":
    main()
