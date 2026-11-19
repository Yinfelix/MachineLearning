"""从实验输出绘图。python -m src.plots 单独更新实际值与预测值图。"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .config import AUDIT_OUTPUT_PATH, FIGURE_DIRECTORY
from .models import metrics


FONTS = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
         "WenQuanYi Micro Hei", "DejaVu Sans"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY / name, dpi=185, bbox_inches="tight")
    plt.close(fig)


def draw_predictions(pred, winners):
    """按每条明细的原始坐标绘图，并放大低值密集区。"""
    style = {
        "font.sans-serif": FONTS, "font.size": 12, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
    }
    specs = [
        ("Quantity", "（a）购买件数", "件", 21, [0, 5, 10, 15, 20], 8),
        ("Total_Bill", "（b）明细金额", "数据集金额单位", 1750, [0, 500, 1000, 1500], 500),
    ]
    names = {"median": "总体中位数", "product_median": "商品中位数",
             "ridge": "岭回归", "tree": "决策树", "forest": "随机森林",
             "price_x_quantity": "折后单价 × 预测件数"}
    blue, orange = "#226793", "#b95532"
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.4))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.81, wspace=0.28)
        fig.suptitle(f"独立测试集实际值与预测值   n = {len(pred):,}", fontsize=16, y=0.98)
        for ax, (target, title, unit, limit, ticks, zoom) in zip(axes, specs):
            actual = pred[target]
            fitted = pred[f"{target}_{winners[target]}"]
            limit = max(limit, 1.05 * actual.max(), 1.05 * fitted.max())
            ax.scatter(actual, fitted, s=16, alpha=0.28, color=blue, edgecolors="none", zorder=3)
            ax.plot([0, limit], [0, limit], "--", color=orange, lw=1.3, zorder=2)
            ax.set(xlim=(0, limit), ylim=(0, limit), xticks=ticks, yticks=ticks,
                   xlabel=f"实际值（{unit}）", ylabel=f"预测值（{unit}）", aspect="equal")
            ax.set_title(f"{title}\n{names[winners[target]]}", fontsize=13, pad=12)
            ax.grid(color="#dfe4e8", linewidth=0.6, zorder=0)

            inset = ax.inset_axes([0.12, 0.54, 0.40, 0.40])
            inset.set_facecolor("#f7f9fb")
            inset.scatter(actual, fitted, s=10, alpha=0.28, color=blue, edgecolors="none", zorder=3)
            inset.plot([0, zoom], [0, zoom], "--", color=orange, lw=1, zorder=2)
            inset.set(xlim=(0, zoom), ylim=(0, zoom), xticks=[0, zoom / 2, zoom],
                      yticks=[0, zoom / 2, zoom], aspect="equal")
            inset.set_title("局部放大", fontsize=10, pad=4)
            inset.tick_params(labelsize=9, length=3, pad=2)
            inset.grid(color="#e3e8ec", linewidth=0.5, zorder=0)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#9aa5ae")
            ax.add_patch(Rectangle((0, 0), zoom, zoom, fill=False, edgecolor="#9aa5ae", lw=0.8))

            score = metrics(actual, fitted)
            digits = 4 if target == "Quantity" else 2
            ax.text(0.5, -0.23, f"MAE = {score['MAE']:.{digits}f}    RMSE = {score['RMSE']:.{digits}f}\n"
                    f"R² = {score['R2']:.4f}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=11, linespacing=1.5)

        legend = [
            Line2D([], [], marker="o", linestyle="none", color=blue, markersize=5, label="测试明细"),
            Line2D([], [], linestyle="--", color=orange, lw=1.3, label="理想预测  y = x"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIRECTORY / "05_actual_predicted.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def draw_figures(train, results, pred, winners):
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "font.sans-serif": FONTS,
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

    draw_predictions(pred, winners)

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


if __name__ == "__main__":
    directory = AUDIT_OUTPUT_PATH.parent
    predictions = pd.read_csv(directory / "predictions.csv")
    selected = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    draw_predictions(predictions, selected)
