"""Static report figures with a restrained, colour-blind-safe palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import auc as auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from . import config as C

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": C.SURFACE,
        "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": C.GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "text.color": C.INK,
        "axes.labelcolor": C.INK_2,
        "xtick.color": C.MUTED,
        "ytick.color": C.MUTED,
        "xtick.labelcolor": C.INK_2,
        "ytick.labelcolor": C.INK_2,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    }
)


def savefig(fig, name: str) -> Path:
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _style_axes(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=C.GRID, linewidth=0.6)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_dataset_overview(df: pd.DataFrame, train_df: pd.DataFrame, path_name="fig01_dataset_overview.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    ax = axes[0, 0]
    counts = train_df["label"].map({1: "合法", 0: "钓鱼"}).value_counts()
    colors = [C.PALETTE["blue"], C.PALETTE["orange"]]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:,}\n({v/len(train_df):.1%})",
                ha="center", va="bottom", fontsize=8.5, color=C.INK_2)
    ax.set_title("(a) 训练集类别分布")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, counts.max() * 1.18)
    _style_axes(ax)

    ax = axes[0, 1]
    bins = np.linspace(0, 100, 51)
    for lab, name, color in [("合法", "legitimate", C.PALETTE["blue"]), ("钓鱼", "phishing", C.PALETTE["orange"])]:
        vals = train_df.loc[train_df["label"] == (1 if name == "legitimate" else 0), "URLSimilarityIndex"]
        ax.hist(vals, bins=bins, alpha=0.65, label=lab, color=color, density=True)
    ax.set_yscale("log")
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("密度（对数轴）")
    ax.set_title("(b) 捷径特征的类条件分布")
    ax.legend(loc="upper center")
    _style_axes(ax)

    ax = axes[1, 0]
    dom = train_df["Domain"].value_counts()
    ax.hist(dom.values, bins=np.logspace(0, np.log10(max(dom.max(), 2)), 35), color=C.PALETTE["violet"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("单个 Domain 的样本数")
    ax.set_ylabel("Domain 数量（对数轴）")
    ax.set_title(f"(c) Domain 重复情况：{len(dom):,} 个域名")
    _style_axes(ax)

    ax = axes[1, 1]
    tld = df["TLD"].value_counts().head(12)
    ax.barh(tld.index[::-1], tld.values[::-1], color=C.PALETTE["aqua"], height=0.62)
    for i, v in enumerate(tld.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=C.INK_2)
    ax.set_xlabel("样本数")
    ax.set_title("(d) 最常见的 12 个顶级域名")
    _style_axes(ax, grid_axis="x")

    fig.tight_layout()
    return savefig(fig, path_name)


def plot_shortcut_panel(train_df: pd.DataFrame, path_name="fig23_shortcut.png"):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    bins = np.linspace(0, 100, 41)
    for lab, val, color in [("合法", 1, C.PALETTE["blue"]), ("钓鱼", 0, C.PALETTE["orange"])]:
        ax.hist(train_df.loc[train_df["label"] == val, "URLSimilarityIndex"], bins=bins,
                alpha=0.65, color=color, label=lab, density=True)
    ax.set_yscale("log")
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("密度（对数轴）")
    ax.set_title("(a) 合法样本全部等于 100")
    ax.legend(loc="upper center")
    _style_axes(ax)

    ax = axes[1]
    d = train_df.groupby("label")["IsHTTPS"].mean().reindex([1, 0])
    bars = ax.bar(["合法", "钓鱼"], d.values, color=[C.PALETTE["blue"], C.PALETTE["orange"]], width=0.5)
    for b, v in zip(bars, d.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9, color=C.INK_2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("IsHTTPS = 1 的比例")
    ax.set_title("(b) 合法样本全部使用 HTTPS")
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_univariate_auc(auc_df: pd.DataFrame, path_name="fig02_univariate_auc.png", top_n=18):
    d = auc_df.sort_values("auc_oriented", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    group_colors = {"URL/域名": C.PALETTE["blue"], "页面内容": C.PALETTE["orange"], "捷径特征": C.PALETTE["red"]}
    colors = [group_colors.get(g, C.MUTED) for g in d["group"]]
    bars = ax.barh(d["feature"], d["auc_oriented"], color=colors, height=0.62)
    for b, v, raw, direction in zip(bars, d["auc_oriented"], d["auc"], d["direction"]):
        arrow = "↓" if direction.startswith("lower") else "↑"
        ax.text(min(v + 0.012, 1.005), b.get_y() + b.get_height() / 2, f"{v:.3f}{arrow}",
                va="center", fontsize=7.8, color=C.INK_2)
    ax.axvline(0.5, color=C.MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlim(0.45, 1.06)
    ax.set_xlabel("单变量 ROC-AUC（方向校正后，训练集）")
    ax.set_title(f"单变量判别力最强的 {top_n} 个特征")
    handles = [Patch(color=v, label=k) for k, v in group_colors.items()]
    ax.legend(handles=handles, loc="lower right")
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_correlation(corr: pd.DataFrame, path_name="fig03_correlation.png", title="特征 Spearman 相关性（训练集）"):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.3, linecolor=C.SURFACE,
        cbar_kws={"shrink": 0.72, "label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.tick_params(axis="both", labelsize=6.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_scheme_model_heatmap(results: pd.DataFrame, metric="f1", path_name="fig04_scheme_heatmap.png"):
    pivot = results.pivot(index="model", columns="scheme", values=f"{metric}_test")
    pivot = pivot.reindex(index=C.MODEL_ORDER, columns=C.SCHEME_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="Blues", vmin=max(0, pivot.min().min() - 0.02), vmax=1.0,
        linewidths=0.8, linecolor=C.SURFACE, cbar_kws={"label": f"测试集 {metric.upper()}"}, ax=ax,
    )
    ax.set_xlabel("特征方案")
    ax.set_ylabel("")
    ax.set_title(f"模型 × 特征方案：测试集 {metric.upper()}")
    ax.tick_params(axis="both", labelsize=8.5)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_ablation_lines(results: pd.DataFrame, path_name, metric="test_f1"):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for ax, schemes, title in [
        (axes[0], ["A", "B", "S", "T"], "(a) 全特征路径：A → B → S → T"),
        (axes[1], ["C", "U", "V"], "(b) 仅 URL 路径：C → U → V"),
    ]:
        for model in C.MODEL_ORDER:
            vals = []
            for s in schemes:
                d = results[(results["model"].astype(str) == model) & (results["scheme"] == s)]
                vals.append(float(d[metric].iloc[0]) if len(d) else np.nan)
            ax.plot(range(len(schemes)), vals, color=C.MODEL_COLORS[model], linewidth=2.0,
                    marker="o", markersize=5, label=model)
            # Exact values live in the table; the figure only carries the trend.
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([f"{s}\n{C.SCHEMES[s]['label'].split(' ', 1)[1]}" for s in schemes], fontsize=8.5)
        ax.set_ylabel("测试集 F1")
        ax.set_ylim(0.45, 1.015)
        ax.set_xlim(-0.15, len(schemes) - 0.35)
        ax.set_title(title)
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return savefig(fig, path_name)


def plot_ablation_heatmap(results: pd.DataFrame, schemes, path_name, title, metric="test_f1"):
    pivot = results.pivot(index="model", columns="scheme", values=metric)
    pivot = pivot.reindex(index=C.MODEL_ORDER, columns=schemes)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="Blues", vmin=max(0, pivot.min().min() - 0.02), vmax=1.0,
        linewidths=0.8, linecolor=C.SURFACE, cbar_kws={"label": metric.replace("test_", "测试集 ").upper()}, ax=ax,
    )
    ax.set_xlabel("特征方案")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8.5)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_metric_bars(cv_results: pd.DataFrame, metric="f1", scheme="B", path_name="fig05_metric_bars.png"):
    d = cv_results[(cv_results["scheme"] == scheme)].set_index("model").reindex(C.MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(d))
    vals = d[f"{metric}_mean"].values
    errs = d[f"{metric}_std"].values
    bars = ax.bar(x, vals, yerr=errs, color=[C.MODEL_COLORS[m] for m in d.index], width=0.6,
                  capsize=3, error_kw={"elinewidth": 1, "ecolor": C.INK_2})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, color=C.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n", 1) for m in d.index], fontsize=8.5)
    ax.set_ylabel(f"{metric.upper()}（5 折均值 ± 标准差）")
    ax.set_title(f"特征方案 {scheme} 下的模型对比")
    lo = max(0, np.nanmin(vals - errs) - 0.04)
    ax.set_ylim(lo, min(1.02, np.nanmax(vals + errs) + 0.035))
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_roc_curves(curves: dict, path_name="fig06_roc.png", title="ROC 曲线（特征方案 B，测试集）"):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot([0, 1], [0, 1], color=C.MUTED, linestyle="--", linewidth=1, label="随机猜测")
    for name in C.MODEL_ORDER:
        if name not in curves:
            continue
        y, s = curves[name]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=C.MODEL_COLORS[name], linewidth=1.9, label=f"{name} (AUC={auc_score(fpr, tpr):.4f})")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", fontsize=7.8)
    _style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_pr_curves(curves: dict, path_name="fig07_pr.png", title="PR 曲线（特征方案 B，测试集）"):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for name in C.MODEL_ORDER:
        if name not in curves:
            continue
        y, s = curves[name]
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(r, p, color=C.MODEL_COLORS[name], linewidth=1.9, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=7.8)
    _style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_confusion(cm: np.ndarray, title: str, path_name: str):
    fig, ax = plt.subplots(figsize=(4.1, 3.6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, square=True,
                xticklabels=["合法", "钓鱼"], yticklabels=["合法", "钓鱼"], ax=ax)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_probability_hist(scores: np.ndarray, y: np.ndarray, title: str, path_name: str):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bins = np.linspace(0, 1, 51)
    ax.hist(scores[y == 0], bins=bins, alpha=0.62, label="真实合法", color=C.PALETTE["blue"], density=True)
    ax.hist(scores[y == 1], bins=bins, alpha=0.62, label="真实钓鱼", color=C.PALETTE["orange"], density=True)
    ax.set_yscale("log")
    ax.set_xlabel("预测为钓鱼的概率 / 归一化分数")
    ax.set_ylabel("密度（对数轴）")
    ax.set_title(title)
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_threshold_curve(thresholds, recall, fpr, chosen, title, path_name):
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(thresholds, recall, color=C.PALETTE["blue"], linewidth=1.9, label="召回率（钓鱼）")
    ax.plot(thresholds, 1 - np.asarray(fpr), color=C.PALETTE["orange"], linewidth=1.9, label="合法样本通过率 (1-FPR)")
    ax.axvline(chosen, color=C.PALETTE["red"], linestyle="--", linewidth=1.2, label=f"选定阈值 {chosen:.3f}")
    ax.set_xlabel("分类阈值")
    ax.set_ylabel("比例")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    _style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_curve(x, series: dict, xlabel, ylabel, title, path_name, marker=True, logx=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, color=None, linewidth=2.0, marker="o" if marker else None, markersize=5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_horizontal_importance(df: pd.DataFrame, path_name, title, top_n=20, color=None):
    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    vals = d["importance"].values
    ax.barh(d["feature"], vals, color=color or C.PALETTE["blue"], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7.6, color=C.INK_2)
    ax.set_xlabel("重要性")
    ax.set_title(title)
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_pruning(depth_df: pd.DataFrame, ccp_df: pd.DataFrame, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    ax.plot(depth_df["max_depth"], depth_df["train_f1"], color=C.PALETTE["blue"], linewidth=2.0, marker="o", markersize=5, label="训练 F1")
    ax.plot(depth_df["max_depth"], depth_df["val_f1"], color=C.PALETTE["orange"], linewidth=2.0, marker="o", markersize=5, label="验证 F1")
    ax.fill_between(depth_df["max_depth"], depth_df["val_f1"] - depth_df["val_f1_std"], depth_df["val_f1"] + depth_df["val_f1_std"], color=C.PALETTE["orange"], alpha=0.15, linewidth=0)
    ax.set_xlabel("max_depth（99 表示不限制）")
    ax.set_ylabel("F1")
    ax.set_title("(a) 树深度的过拟合曲线（方案 B）")
    ax.legend(loc="lower right")
    _style_axes(ax)

    ax = axes[1]
    d = ccp_df[ccp_df["ccp_alpha"] > 0]
    ax.plot(d["ccp_alpha"], d["train_f1"], color=C.PALETTE["blue"], linewidth=2.0, marker="o", markersize=4.5, label="训练 F1")
    ax.plot(d["ccp_alpha"], d["val_f1"], color=C.PALETTE["orange"], linewidth=2.0, marker="o", markersize=4.5, label="验证 F1")
    ax.fill_between(d["ccp_alpha"], d["val_f1"] - d["val_f1_std"], d["val_f1"] + d["val_f1_std"], color=C.PALETTE["orange"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("ccp_alpha（对数轴）")
    ax.set_ylabel("F1")
    ax.set_title("(b) 代价复杂度剪枝曲线（方案 B）")
    ax.legend(loc="lower left")
    _style_axes(ax)

    fig.tight_layout()
    return savefig(fig, path_name)


def plot_rf_optimization(df: pd.DataFrame, path_name):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["cv_f1_mean"], width=w, color=C.PALETTE["blue"], label="5 折验证 F1",
           yerr=df["cv_f1_std"], capsize=3, error_kw={"elinewidth": 1, "ecolor": C.INK_2})
    ax.bar(x + w / 2, df["test_f1"], width=w, color=C.PALETTE["orange"], label="测试集 F1")
    for i, (a, b) in enumerate(zip(df["cv_f1_mean"], df["test_f1"])):
        ax.text(i - w / 2, a + 0.004, f"{a:.4f}", ha="center", fontsize=7.8, color=C.INK_2)
        ax.text(i + w / 2, b + 0.004, f"{b:.4f}", ha="center", fontsize=7.8, color=C.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], fontsize=8.5)
    ax.set_ylabel("F1")
    ax.set_ylim(max(0, min(df["cv_f1_mean"].min(), df["test_f1"].min()) - 0.05), 1.01)
    ax.set_title("随机森林调参前后的性能（方案 B）")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_confusion_pair(cms: dict, titles: dict, path_name):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.7))
    if len(cms) == 1:
        axes = [axes]
    for ax, (key, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm)
        row = cm.sum(axis=1, keepdims=True)
        pct = cm / np.maximum(row, 1)
        annot = np.array([[f"{v:,}\n({p:.2%})" for v, p in zip(rv, rp)] for rv, rp in zip(cm, pct)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, square=True,
                    xticklabels=["合法", "钓鱼"], yticklabels=["合法", "钓鱼"], ax=ax, annot_kws={"fontsize": 9})
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(titles.get(key, key))
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_feature_selection(k_df: pd.DataFrame, pca_df: pd.DataFrame, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax = axes[0]
    for model in k_df["model"].unique():
        d = k_df[k_df["model"] == model].copy()
        d["x"] = d["k"].replace({"all": 45}).astype(int)
        ax.plot(d["x"], d["cv_f1_mean"], color=C.MODEL_COLORS.get(model, C.MUTED), linewidth=2.0,
                marker="o", markersize=5, label=model)
        ax.fill_between(d["x"], d["cv_f1_mean"] - d["cv_f1_std"], d["cv_f1_mean"] + d["cv_f1_std"],
                        color=C.MODEL_COLORS.get(model, C.MUTED), alpha=0.12, linewidth=0)
    ax.set_xlabel("SelectKBest 保留的特征数（45 = 全部）")
    ax.set_ylabel("5 折验证 F1")
    ax.set_title("(a) 过滤法特征选择（方案 B）")
    ax.legend()
    _style_axes(ax)

    ax = axes[1]
    for model in pca_df["model"].unique():
        d = pca_df[pca_df["model"] == model]
        ax.plot(d["variance"] * 100, d["cv_f1_mean"], color=C.MODEL_COLORS.get(model, C.MUTED),
                linewidth=2.0, marker="o", markersize=5, label=model)
    ax.set_xlabel("PCA 保留方差（%）")
    ax.set_ylabel("5 折验证 F1")
    ax.set_title("(b) PCA 降维（方案 B）")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_learning_curve(df: pd.DataFrame, path_name):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in df["model"].unique():
        d = df[df["model"] == model].groupby("fraction").agg(f1=("f1", "mean"), sd=("f1", "std")).reset_index()
        ax.plot(d["fraction"] * 100, d["f1"], color=C.MODEL_COLORS.get(model, C.MUTED), linewidth=2.0,
                marker="o", markersize=5, label=model)
        ax.fill_between(d["fraction"] * 100, d["f1"] - d["sd"].fillna(0), d["f1"] + d["sd"].fillna(0),
                        color=C.MODEL_COLORS.get(model, C.MUTED), alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("训练集使用比例（对数轴）")
    ax.set_ylabel("验证集 F1")
    ax.set_title("学习曲线（方案 V，3 个随机种子）")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_noise_panels(label_df: pd.DataFrame, feature_df: pd.DataFrame, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, df, xcol, title in [
        (axes[0], label_df, "noise_rate", "(a) 标签噪声"),
        (axes[1], feature_df, "noise_sd", "(b) 测试特征高斯噪声"),
    ]:
        for model in df["model"].unique():
            d = df[df["model"] == model]
            ax.plot(d[xcol], d["f1"], color=C.MODEL_COLORS.get(model, C.MUTED), linewidth=2.0,
                    marker="o", markersize=5, label=model)
        ax.set_xlabel("噪声比例" if xcol == "noise_rate" else "噪声强度（训练集标准差倍数）")
        ax.set_ylabel("验证集 F1")
        ax.set_title(title)
        ax.legend()
        _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_overfitting(df: pd.DataFrame, path_name):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["train_f1"], width=w, color=C.PALETTE["blue"], label="训练 F1")
    ax.bar(x + w / 2, df["val_f1"], width=w, color=C.PALETTE["orange"], label="验证 F1")
    for i, (tr, va) in enumerate(zip(df["train_f1"], df["val_f1"])):
        ax.text(i, max(tr, va) + 0.004, f"差 {tr-va:+.4f}", ha="center", fontsize=7.8, color=C.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=8.3)
    ax.set_ylabel("F1（5 折均值）")
    ax.set_ylim(max(0, df[["train_f1", "val_f1"]].min().min() - 0.05), 1.02)
    ax.set_title("训练集与验证集 F1 的差距（方案 V）")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_tld_transfer(df: pd.DataFrame, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for ax, metric, title in [(axes[0], "f1", "(a) F1"), (axes[1], "roc_auc", "(b) ROC-AUC")]:
        models_ = df["model"].unique()
        x = np.arange(len(models_))
        w = 0.36
        a = df[df["split"] == "随机划分"].set_index("model").reindex(models_)[metric]
        b = df[df["split"] == "罕见 TLD"].set_index("model").reindex(models_)[metric]
        ax.bar(x - w / 2, a.values, width=w, color=C.PALETTE["blue"], label="同分布随机划分")
        ax.bar(x + w / 2, b.values, width=w, color=C.PALETTE["orange"], label="罕见 TLD 迁移")
        for i, v in enumerate(a.values):
            ax.text(i - w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5, color=C.INK_2)
        for i, v in enumerate(b.values):
            ax.text(i + w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5, color=C.INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels(models_)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_ylim(max(0, min(a.min(), b.min()) - 0.08), 1.02)
        ax.legend()
        _style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path_name)


def plot_pca_scatter(emb: np.ndarray, y: np.ndarray, pred: np.ndarray, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(len(emb), size=min(12000, len(emb)), replace=False)
    for ax, color_by, title in [(axes[0], y, "(a) 按真实类别着色"), (axes[1], (pred != y).astype(int), "(b) 按预测对错着色")]:
        vals = color_by[idx]
        if title.startswith("(a)"):
            for cls, name, color in [(0, "合法", C.PALETTE["blue"]), (1, "钓鱼", C.PALETTE["orange"])]:
                m = vals == cls
                ax.scatter(emb[idx][m, 0], emb[idx][m, 1], s=4, alpha=0.35, color=color, label=name, linewidths=0)
            ax.legend(markerscale=3)
        else:
            m0 = vals == 0
            ax.scatter(emb[idx][m0, 0], emb[idx][m0, 1], s=4, alpha=0.25, color=C.MUTED, label="预测正确", linewidths=0)
            ax.scatter(emb[idx][~m0, 0], emb[idx][~m0, 1], s=10, alpha=0.8, color=C.PALETTE["red"], label="预测错误", linewidths=0)
            ax.legend(markerscale=2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        _style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return savefig(fig, path_name)
