"""Final sanity check: expected artefacts and headline numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C


def main():
    required = [
        "outputs/metrics/results_all.csv",
        "outputs/metrics/tuning_summary.json",
        "outputs/metrics/robustness_learning_curve.csv",
        "outputs/metrics/feature_selection_k.csv",
        "outputs/metrics/posthoc_analysis.json",
        "outputs/figures/fig01_dataset_overview.png",
        "outputs/figures/fig04_ablation_lines.png",
        "outputs/figures/fig06_roc.png",
        "outputs/figures/fig09_dt_pruning.png",
        "outputs/figures/fig16_permutation_importance.png",
        "outputs/figures/fig17_shap.png",
        "实验2报告_钓鱼网站检测.docx",
        "实验2汇报_钓鱼网站检测.pptx",
    ]
    missing = [p for p in required if not (C.ROOT / p).exists()]
    if missing:
        print("MISSING:")
        for p in missing:
            print(" -", p)
        raise SystemExit(1)

    res = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    assert len(res) == 6 * 8, len(res)
    assert res["test_f1"].notna().all()
    print(f"results: {res.shape[0]} rows, {res.shape[1]} columns")
    print(res.pivot(index="model", columns="scheme", values="test_f1")[
        ["A", "B", "C", "D", "S", "U", "T", "V"]
    ].round(4).to_string())
    print("\nAll required artefacts present.")


if __name__ == "__main__":
    main()
