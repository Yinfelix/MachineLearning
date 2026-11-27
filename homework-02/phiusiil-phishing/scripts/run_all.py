"""Run the full experiment pipeline in order (roughly 60–90 minutes)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/01_audit_eda.py",
    "scripts/02_tune.py",
    "scripts/03_final_eval.py",
    "scripts/04_optimize.py",
    "scripts/05_robustness.py",
    "scripts/06_feature_engineering.py",
    "scripts/07_report_assets.py",
    "scripts/08_posthoc.py",
    "report/build_report.py",
    "report/build_ppt.py",
    "scripts/09_verify.py",
]


def main():
    for rel in SCRIPTS:
        print(f"\n===== {rel} =====", flush=True)
        subprocess.run([sys.executable, "-u", "-I", str(ROOT / rel)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
