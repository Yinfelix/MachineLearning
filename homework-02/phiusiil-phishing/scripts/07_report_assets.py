"""Step 7: consolidate figures used by the report and slides."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import plots
from src.evaluate import save_json


def main() -> None:
    # ------------------------------------------------ two-rule shortcut baseline
    from src import data
    from src.evaluate import metrics_dict

    df = data.load_raw()
    splits = data.load_splits()
    te = splits["test_idx"]
    y_te = (df["label"].iloc[te] == 0).astype(int)
    sim = df["URLSimilarityIndex"].iloc[te]
    https = df["IsHTTPS"].iloc[te]
    rule1 = sim < 100
    rule2 = (~rule1) & (https == 0)
    pred = (rule1 | rule2).astype(int)
    m = metrics_dict(y_te, pred)
    train_df = df.iloc[splits["train_idx"]].copy()
    plots.plot_shortcut_panel(train_df, "fig23_shortcut.png")

    save_json(
        {
            "rule": "URLSimilarityIndex < 100 -> phishing; else IsHTTPS == 0 -> phishing; else legitimate",
            "test_metrics": m,
            "rule1_flag_rate": float(rule1.mean()),
            "rule2_flag_rate": float(rule2.mean()),
            "phishing_caught_by_rule1": int((rule1 & (y_te == 1)).sum()),
            "phishing_caught_by_rule2": int((rule2 & (y_te == 1)).sum()),
            "phishing_total": int(y_te.sum()),
        },
        C.METRIC_DIR / "shortcut_rule.json",
    )

    res = pd.read_csv(C.METRIC_DIR / "results_all.csv")
    res_main = pd.read_csv(C.METRIC_DIR / "results_main.csv")
    scores = np.load(C.METRIC_DIR / "test_scores.npz")
    cms = json.loads((C.METRIC_DIR / "confusion_matrices.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------ main heatmap A-D
    plots.plot_ablation_heatmap(res_main, C.SCHEME_ORDER, "fig04_scheme_heatmap.png",
                                "模型 × 特征方案：测试集 F1（方案 A–D）")

    # ------------------------------------------- supplementary strict ablation
    ladder = res[res["scheme"].isin(["A", "B", "S", "T"])]
    plots.plot_ablation_heatmap(ladder, ["A", "B", "S", "T"], "fig04b_ablation.png",
                                "逐步去除捷径：A → B → S → T（测试集 F1）")
    url_ladder = res[res["scheme"].isin(["C", "U", "V"])]
    plots.plot_ablation_heatmap(url_ladder, ["C", "U", "V"], "fig04c_url_ablation.png",
                                "仅 URL 特征：C → U → V（测试集 F1）")
    plots.plot_ablation_lines(res, "fig04_ablation_lines.png")

    # ------------------------------------------------------------ metric bars
    d = res_main[res_main["scheme"] == "B"].copy()
    d = d.rename(columns={"cv_f1_mean": "f1_mean", "cv_f1_std": "f1_std"})
    d["model"] = pd.Categorical(d["model"], categories=C.MODEL_ORDER, ordered=True)
    plots.plot_metric_bars(d, metric="f1", scheme="B", path_name="fig05_metric_bars.png")

    # ------------------------------------------------------------- ROC / PR
    curves = {}
    for name in C.MODEL_ORDER:
        curves[name] = (scores["y_test"], scores[f"{name}|V|score"])
    plots.plot_roc_curves(curves, "fig06_roc.png", title="ROC 曲线（特征方案 V，测试集）")
    plots.plot_pr_curves(curves, "fig07_pr.png", title="PR 曲线（特征方案 V，测试集）")

    # ------------------------------------------------ RF optimisation & threshold
    opt = pd.read_csv(C.METRIC_DIR / "optimization_rf.csv")
    plots.plot_rf_optimization(opt, "fig08_rf_optimization.png")

    # ------------------------------------------------------- confusion matrices
    plots.plot_confusion_pair(
        {"Logistic Regression|V": cms["Logistic Regression|V"], "Decision Tree|V": cms["Decision Tree|V"]},
        {"Logistic Regression|V": "(a) 逻辑回归（方案 V）", "Decision Tree|V": "(b) 决策树（方案 V）"},
        "fig20_confusion.png",
    )

    # -------------------------------------------------------- probability hist
    y_test = scores["y_test"]
    plots.plot_probability_hist(scores["Logistic Regression|V|score"], y_test,
                                "逻辑回归（方案 V）的预测概率分布", "fig21_prob_hist.png")

    # ----------------------------------------------------- feature selection
    k_df = pd.read_csv(C.METRIC_DIR / "feature_selection_k.csv")
    pca_df = pd.read_csv(C.METRIC_DIR / "pca_results.csv")
    plots.plot_feature_selection(k_df, pca_df, "fig15_feature_selection.png")

    # ----------------------------------------------------- error counts on B
    d = res[res["scheme"] == "V"].set_index("model").reindex(C.MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(d))
    ax.bar(x, d["test_fp"], color=C.PALETTE["blue"], width=0.55, label="误报 FP（合法→钓鱼）")
    ax.bar(x, d["test_fn"], bottom=d["test_fp"], color=C.PALETTE["orange"], width=0.55, label="漏报 FN（钓鱼→合法）")
    for i, (fp, fn) in enumerate(zip(d["test_fp"], d["test_fn"])):
        ax.text(i, fp + fn + max(d["test_fp"].max(), 1) * 0.02, f"{int(fp)} / {int(fn)}", ha="center", fontsize=8, color=C.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n", 1) for m in d.index], fontsize=8.3)
    ax.set_ylabel("测试集错误数")
    ax.set_title("方案 V 的测试集错误构成")
    ax.legend()
    plots._style_axes(ax)
    fig.tight_layout()
    plots.savefig(fig, "fig22_error_counts.png")

    # ------------------------------------------------------ key numbers table
    save_json(
        {
            "best_test_f1": float(res_main["test_f1"].max()),
            "best_row": res_main.loc[res_main["test_f1"].idxmax(), ["model", "scheme", "test_f1", "test_accuracy", "test_fpr"]].to_dict(),
            "worst_test_f1": float(res_main["test_f1"].min()),
            "scheme_mean_f1": res_main.groupby("scheme")["test_f1"].mean().to_dict(),
            "model_mean_f1": res_main.groupby("model")["test_f1"].mean().to_dict(),
        },
        C.METRIC_DIR / "report_key_numbers.json",
    )
    print("report assets complete", flush=True)


if __name__ == "__main__":
    main()
