"""从实验输出绘图。python -m src.plots 单独更新实际值与预测值图。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .config import AUDIT_OUTPUT_PATH, FEATURE_COLUMNS, FIGURE_DIRECTORY, NUMERIC_FEATURES
from .dataset import load_dataset, split_by_date
from .models import pipeline


FONTS = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
         "WenQuanYi Micro Hei", "DejaVu Sans"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY / name, dpi=185, bbox_inches="tight")
    plt.close(fig)


def draw_predictions(pred):
    """以面包的标价为横轴，对比实际值、商品中位数预测和岭回归直线。"""
    pred = pred.reset_index(drop=True)
    data, _ = load_dataset()
    parts = split_by_date(data)
    history = pd.concat([parts.train, parts.validation])
    test = parts.test.reset_index(drop=True)
    group = history[history.Product == "Bread"]
    mask = test.Product == "Bread"
    actual = pred.loc[mask]
    price = test.loc[mask, "Unit_Price"]

    # 固定其他输入，观察已选定的多变量岭回归模型随标价变化的预测。
    reference = group[FEATURE_COLUMNS].mode(dropna=False).iloc[[0]].copy()
    reference[NUMERIC_FEATURES] = group[NUMERIC_FEATURES].median().to_numpy()
    grid = pd.concat([reference, reference], ignore_index=True)
    grid["Unit_Price"] = [price.min(), price.max()]
    cv = pd.read_csv(AUDIT_OUTPUT_PATH.parent / "cv.csv")

    style = {
        "font.sans-serif": FONTS, "font.size": 13, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
    }
    specs = [
        ("Quantity", "Quantity_product_median", "（a）购买件数", "件"),
        ("Total_Bill", "Total_Bill_price_x_quantity", "（b）明细金额", "数据集金额单位"),
    ]
    gray, red, blue = "#92979d", "#ce3434", "#2463a6"
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.32, top=0.80, wspace=0.33)
        fig.suptitle(f"面包（Bread）的实际值与预测值   测试明细 n = {len(actual)}",
                     fontsize=16, y=0.96)
        for ax, (target, column, title, unit) in zip(axes, specs):
            alpha = cv.loc[(cv.target == target) & (cv.model == "ridge") & cv.chosen, "alpha"].iloc[0]
            ridge = pipeline("ridge").set_params(model__alpha=alpha)
            ridge.fit(history[FEATURE_COLUMNS], history[target])
            line = ridge.predict(grid)

            ax.scatter(price, actual[target], s=42, color=gray, alpha=0.65,
                       edgecolors="white", linewidths=0.4, zorder=2)
            ax.scatter(price, actual[column], s=15, color=red, alpha=0.85,
                       edgecolors="none", zorder=4)
            ax.plot(grid.Unit_Price, line, color=blue, lw=2.1, zorder=3)
            ax.set(xlabel="自变量 x：商品标价\n（数据集金额单位）",
                   ylabel=f"因变量 y：{title[3:]}\n（{unit}）", title=title,
                   xlim=(38, 42), ylim=(0, 1.08 * max(actual[target].max(), line.max())),
                   xticks=[38, 39, 40, 41, 42])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(color="#e1e5e8", linewidth=0.6, zorder=0)

        legend = [
            Line2D([], [], marker="o", linestyle="none", color=gray, markersize=6, label="实际值"),
            Line2D([], [], marker="o", linestyle="none", color=red, markersize=5, label="商品中位数方案预测值"),
            Line2D([], [], color=blue, lw=2.1, label="岭回归预测直线"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.10))
        fig.text(0.5, 0.045, "蓝线固定其余数值输入为历史中位数、类别输入为众数；红点为逐条测试预测。",
                 ha="center", fontsize=11, color="#444444")
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

    draw_predictions(pred)

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
    draw_predictions(predictions)
