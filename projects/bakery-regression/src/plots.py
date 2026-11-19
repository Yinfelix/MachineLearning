"""从实验输出绘制分布、误差比较与实际值对预测值图。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AUDIT_OUTPUT_PATH, FIGURE_DIRECTORY


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY / name, dpi=185, bbox_inches="tight")
    plt.close(fig)


def draw_figures(train, results, pred, winners):
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                                             "WenQuanYi Micro Hei", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    })
    qty, bill = "Quantity", "Total_Bill"
    teal, blue = "#126e71", "#43618b"

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 2.9))
    counts = train[qty].value_counts().sort_index()
    left.bar(counts.index, 100 * counts / len(train), color=teal)
    left.set(xlabel="购买件数", ylabel="训练明细占比（%）", xticks=[1, 2, 4, 6, 10, 15, 20])
    left.set_title("（a）每条明细的购买件数", loc="left")
    right.hist(train[bill], bins=np.geomspace(10, 2000, 34), color=blue)
    right.set_xscale("log")
    ticks = [15, 40, 100, 300, 1000]
    right.set(xticks=ticks, xticklabels=ticks, xlabel="金额（对数刻度）", ylabel="训练明细数")
    right.set_title("（b）每条明细的结账金额", loc="left")
    save(fig, "03_training_targets.png")

    names = {"median": "总体中位数", "product_median": "商品中位数", "ridge": "岭回归",
             "tree": "决策树", "forest": "随机森林", "price_x_quantity": "按商品估计"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, target, title, color in zip(axes, [qty, bill], ["（a）购买件数", "（b）明细金额"], [teal, blue]):
        part = results[results.target == target].sort_values("MAE")
        labels = [names.get(name, name.replace("_", " ")) for name in part.model]
        ax.barh(labels, part.MAE, color=["#c6cdd1" if name != winners[target] else color for name in part.model])
        ax.invert_yaxis()
        ax.set(xlabel="测试集 MAE", title=title, xlim=(0, 1.25 * part.MAE.max()))
        for i, value in enumerate(part.MAE):
            label = f" {value:.4f}" if target == qty else f" {value:.2f}"
            ax.text(value, i, label, va="center", fontsize=9)
    save(fig, "04_test_mae.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, target, title in zip(axes, [qty, bill], ["（a）购买件数", "（b）明细金额"]):
        actual, fitted = pred[target], pred[f"{target}_{winners[target]}"]
        high = max(actual.max(), fitted.max()) * 1.03
        cells = ax.hexbin(actual, fitted, gridsize=30, mincnt=1, bins="log", cmap="viridis",
                          extent=(0, high, 0, high))
        ax.plot([0, high], [0, high], "--", color="#a34d30", lw=1.2)
        ax.set(xlabel="实际值", ylabel="预测值", title=title, xlim=(0, high), ylim=(0, high), aspect="equal")
        fig.colorbar(cells, ax=ax, shrink=0.75, label="明细数")
    save(fig, "05_actual_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for ax, target, title in zip(axes, [qty, bill], ["（a）购买件数", "（b）明细金额"]):
        error = pred[f"{target}_{winners[target]}"] - pred[target]
        bins = np.arange(error.min() - 0.5, error.max() + 1.5) if target == qty else 45
        ax.hist(error, bins=bins, color=teal if target == qty else blue)
        ax.axvline(0, color="#a34d30", ls="--", lw=1.2)
        ax.set(xlabel="预测值减去实际值", ylabel="明细数（对数刻度）", yscale="log", title=title)
    save(fig, "06_residuals.png")

    groups = pd.read_csv(AUDIT_OUTPUT_PATH.parent / "groups.csv").sort_values("removed_delta")
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    pos = np.arange(len(groups))
    ax.barh(pos - 0.17, groups.removed_delta, 0.32, color=teal, label="移除后重训")
    ax.barh(pos + 0.17, groups.shuffled_delta, 0.32, color=blue, label="验证集内置换")
    ax.axvline(0, color="#555555", lw=0.8)
    labels = {"customer": "顾客", "calendar": "日历", "price": "价格与促销", "weather": "天气", "product": "商品"}
    ax.set(yticks=pos, yticklabels=groups.group.map(labels), xlabel="验证集 MAE 变化（件）")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "07_feature_groups.png")
