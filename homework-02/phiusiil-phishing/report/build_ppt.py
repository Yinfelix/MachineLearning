"""Build the 16:9 presentation deck from the same metrics as the report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from load_metrics import FIG, load_all  # noqa: E402
from team import TEAM  # noqa: E402
from ppt_helpers import (  # noqa: E402
    AQUA, BLUE, GRID, INK, INK_2, LIGHT_BLUE, MAGENTA, MUTED, ORANGE, RED, SURFACE,
    VIOLET, YELLOW, blank, bullets, callout, footer, image_fit, new_deck, rect,
    stat_tile, table, textbox, title,
)

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_ORANGE = RGBColor(0xFF, 0xF4, 0xEE)
GREEN = RGBColor(0x00, 0x83, 0x00)
from build_report import MODEL_CN, MAIN, f4, pct, row  # noqa: E402

TOTAL = 17


def slide_cover(prs, R):
    s = blank(prs)
    rect(s, 0, 0, 13.333, 7.5, fill=LIGHT_BLUE)
    rect(s, 0, 0, 0.18, 7.5, fill=BLUE)
    textbox(s, "机器学习课程设计 · 实验二", 0.95, 0.85, 8, 0.4, size=14, bold=True, color=BLUE)
    textbox(s, "PhiUSIIL 钓鱼网站检测", 0.92, 1.65, 11.5, 0.95, size=40, bold=True, color=INK)
    textbox(s, "捷径特征、模型比较与鲁棒性分析", 0.95, 2.65, 11.2, 0.65, size=22, color=INK_2)
    rect(s, 0.98, 3.55, 1.8, 0.07, fill=ORANGE)
    textbox(s, "组员：（待填）　（待填）　（待填）　（待填）", 0.95, 3.95, 10.5, 0.45, size=14, color=INK_2)
    textbox(s, "姓名 / 学号按要求留空　｜　2026 年 11 月", 0.95, 4.45, 9.5, 0.4, size=11, color=MUTED)
    callout(s, "一句话：先证明这个数据集“太容易”，再回答去掉捷径后模型还剩多少真实能力。",
            0.95, 5.45, 8.9, 0.85, accent=ORANGE, size=13)
    footer(s, 1, TOTAL)


def slide_questions(prs, R):
    s = blank(prs)
    title(s, "01　研究问题", "一个“太容易”的数据集，值得认真做吗？", 2, TOTAL)
    items = [
        ("01  哪些特征构成了捷径？", "URLSimilarityIndex、IsHTTPS、页面规模特征分别贡献了多少？"),
        ("02  去掉捷径后，谁更强？", "逻辑回归、朴素贝叶斯、决策树、随机森林、SVM 在同一套划分下比较。"),
        ("03  模型什么时候会失效？", "样本不足、标签噪声、特征噪声、异常值、跨 TLD 迁移。"),
    ]
    y = 1.75
    for i, (head, body) in enumerate(items):
        rect(s, 0.75, y, 7.4, 1.35, fill=LIGHT_BLUE, line=None, radius=True)
        textbox(s, head, 0.98, y + 0.14, 6.9, 0.42, size=17, bold=True, color=INK)
        textbox(s, body, 0.98, y + 0.6, 6.9, 0.6, size=11.5, color=INK_2)
        y += 1.55
    # right stat tiles
    a = R["audit"]
    stat_tile(s, f"{a['n_rows']:,}", "样本数", 8.65, 1.75, 1.95, 1.25, BLUE)
    stat_tile(s, f"{a['n_features_excluding_text']}", "可直接入模特征", 10.8, 1.75, 1.95, 1.25, ORANGE)
    stat_tile(s, pct(a["phishing_rate"]), "钓鱼样本占比", 8.65, 3.2, 1.95, 1.25, VIOLET)
    stat_tile(s, f"{a['unique_domains']:,}", "不同 Domain", 10.8, 3.2, 1.95, 1.25, AQUA)
    callout(s, "我们的判断：高分不等于能力。先做数据体检，再谈模型。", 8.65, 4.85, 4.1, 0.95, accent=ORANGE, size=12)
    footer(s, 2, TOTAL)


def slide_data(prs, R):
    s = blank(prs)
    title(s, "02　数据", "数据体检：类别、域名与 TLD", 3, TOTAL)
    image_fit(s, FIG / "fig01_dataset_overview.png", 0.55, 1.55, 7.7, 5.15)
    a = R["audit"]
    bullets(s, [
        (f"合法 {a['label_counts']['1']:,} / 钓鱼 {a['label_counts']['0']:,}，轻度不平衡", False),
        (f"{a['unique_domains']:,} 个域名；{a['rows_in_repeated_domains']:,} 行落在重复域名中，最大域名 {a['max_domain_group']:,} 条", False),
        (f"54 个域名同时包含两类样本；随机划分会造成同域名泄漏", False),
        (f"{a['unique_tld']} 个 TLD，com 占 {pct(a['top_tld']['com']/a['n_rows'])}；长尾很长", False),
    ], 8.45, 1.75, 4.3, 3.2, size=12.5, spacing=10)
    callout(s, "划分策略：StratifiedGroupKFold(5)，先按 Domain 分组切出 20% 测试集，训练/测试域名重叠 = 0。",
            8.45, 5.3, 4.25, 1.15, accent=BLUE, size=11.5)
    footer(s, 3, TOTAL)


def slide_shortcut(prs, R):
    s = blank(prs)
    title(s, "03　发现", "URLSimilarityIndex：合法样本恒为 100", 4, TOTAL)
    a = R["audit"]
    stat_tile(s, "0.996", "单变量 ROC-AUC", 0.75, 1.65, 2.3, 1.25, RED, "方向校正后")
    stat_tile(s, pct(a["url_similarity_by_label"]["0"]["pct_at_100"]), "钓鱼样本恰好 = 100", 3.2, 1.65, 2.3, 1.25, ORANGE)
    stat_tile(s, f"{a['phishing_with_similarity_100']:,}", "等于 100 的钓鱼样本", 5.65, 1.65, 2.3, 1.25, BLUE)
    image_fit(s, FIG / "fig23_shortcut.png", 8.15, 1.55, 4.65, 3.7)
    s_ = R["shortcut"]
    textbox(s, "两条规则，零训练：", 0.8, 3.25, 3.2, 0.4, size=13, bold=True, color=INK)
    bullets(s, [
        "URLSimilarityIndex < 100 → 判钓鱼",
        "否则 IsHTTPS = 0 → 判钓鱼",
        "其余 → 判合法",
    ], 0.8, 3.62, 6.7, 1.5, size=13, spacing=4)
    callout(s, f"测试集：F1 = {f4(s_['test_metrics']['f1'])}　召回 = {f4(s_['test_metrics']['recall'])}　FPR = 0　只漏 {s_['test_metrics']['fn']} 条",
            0.8, 5.35, 7.1, 0.85, accent=RED, size=12.5)
    textbox(s, "合法样本 100% 等于 100；钓鱼样本均值 49.6，只有 0.78% 恰好等于 100。",
            8.25, 5.4, 4.5, 0.8, size=10.5, color=INK_2)
    footer(s, 4, TOTAL)


def slide_shortcuts2(prs, R):
    s = blank(prs)
    title(s, "03　发现", "不止一个捷径：页面规模特征也有 0.96–0.99 的 AUC", 5, TOTAL)
    image_fit(s, FIG / "fig02_univariate_auc.png", 0.55, 1.5, 7.3, 5.2)
    bullets(s, [
        ("官方衍生特征只是第一层：", True),
        "URLSimilarityIndex 0.996",
        "CharContinuationRate 0.742",
        "URLCharProb 0.768",
        ("页面规模是第二层：", True),
        "LineOfCode 0.991、NoOfExternalRef 0.988、NoOfImage 0.979",
        ("IsHTTPS 是第三层：", True),
        "合法 100% = 1，钓鱼仅 49.2% = 1",
    ], 8.05, 1.65, 4.65, 4.6, size=11.3, spacing=5)
    callout(s, "只去掉 URLSimilarityIndex 还不够，必须做多级消融。", 8.05, 5.95, 4.65, 0.85, accent=ORANGE, size=11)
    footer(s, 5, TOTAL)


def slide_design(prs, R):
    s = blank(prs)
    title(s, "04　设计", "防泄漏划分 + 多级特征消融", 6, TOTAL)
    # left split diagram
    rect(s, 0.75, 1.65, 2.75, 1.15, fill=LIGHT_BLUE, line=None, radius=True)
    textbox(s, "训练集 188,801", 0.9, 1.78, 2.45, 0.4, size=15, bold=True, color=BLUE)
    textbox(s, "钓鱼 80,886（42.8%）", 0.9, 2.2, 2.45, 0.35, size=10, color=INK_2)
    rect(s, 3.65, 1.65, 2.7, 1.15, fill=LIGHT_ORANGE, line=None, radius=True)
    textbox(s, "独立测试集 46,994", 3.8, 1.78, 2.4, 0.4, size=13.5, bold=True, color=ORANGE)
    textbox(s, "钓鱼 20,059（42.7%）", 3.8, 2.2, 2.4, 0.35, size=10, color=INK_2)
    textbox(s, "按 Domain 分组，重叠 = 0", 0.8, 2.95, 5.5, 0.35, size=11.5, color=INK_2)
    bullets(s, [
        "训练集内部 5 折 StratifiedGroupKFold",
        "缩放 / 频次编码 / 特征选择 / PCA 全部在训练折内拟合",
        "测试集只在最终评估时使用一次",
        "超参数、阈值都不使用测试集",
    ], 0.75, 3.35, 5.8, 2.3, size=11.5, spacing=6)
    # right scheme ladder
    textbox(s, "特征方案（A–D 主实验，S–V 严格消融）", 7.0, 1.55, 5.5, 0.4, size=12.5, bold=True, color=INK)
    schemes = [
        ("A 全特征", "51", "基线", BLUE),
        ("B 去官方捷径", "45", "-6 衍生特征", ORANGE),
        ("S 严格去捷径", "39", "-6 页面规模", AQUA),
        ("T 去除全部单向捷径", "38", "-IsHTTPS", RED),
        ("C 仅 URL", "23", "22 数值 + TLD", VIOLET),
        ("D 仅页面", "28", "28 页面数值", YELLOW),
        ("U 仅 URL 基础词法", "19", "-4 URL 捷径", MAGENTA),
        ("V 仅 URL 基础词法（去 HTTPS）", "18", "-IsHTTPS", GREEN),
    ]
    y = 2.0
    for name, n, note, color in schemes:
        rect(s, 7.0, y, 5.55, 0.46, fill=SURFACE, line=GRID, radius=True)
        rect(s, 7.0, y, 0.07, 0.46, fill=color)
        textbox(s, name, 7.15, y + 0.05, 3.2, 0.32, size=10.5, bold=True, color=INK)
        textbox(s, note, 10.35, y + 0.06, 1.65, 0.3, size=9.0, color=MUTED)
        textbox(s, n, 12.0, y + 0.03, 0.45, 0.32, size=11.2, bold=True, color=color, align=PP_ALIGN.RIGHT)
        y += 0.54
    footer(s, 6, TOTAL)


def slide_models(prs, R):
    s = blank(prs)
    title(s, "05　模型", "五类模型，一套公平比较协议", 7, TOTAL)
    rows = [
        ["逻辑回归", "线性对数几率 + Sigmoid", "C、L2 正则", "可解释的线性基线"],
        ["高斯朴素贝叶斯", "条件独立 + 正态假设", "var_smoothing", "检验独立假设"],
        ["决策树", "基尼不纯度递归划分", "max_depth、min_samples_leaf", "规则直观，易过拟合"],
        ["随机森林", "Bagging + 随机特征子空间", "树数、深度、max_features", "表格数据强基线"],
        ["RBF SVM", "最大间隔 + 核函数", "C、gamma", "非线性边界；20k 子样本"],
    ]
    table(s, ["模型", "基本原理", "关键超参数", "选择理由"], rows, 0.7, 1.65, 11.95, 3.5,
          col_widths=[1.7, 3.3, 3.1, 3.4], font_size=11)
    callout(s, "调参：在方案 A 上用完整网格 + 5 折分组 CV 选一次，随后固定用于所有方案，保证消融差异只来自特征。",
            0.7, 5.55, 11.95, 0.8, accent=BLUE, size=12)
    textbox(s, "RBF SVM 在 20,000 条分层子样本上训练；LinearSVC 在全量训练集上作为对照。",
            0.7, 6.5, 11.95, 0.35, size=10.5, color=MUTED)
    footer(s, 7, TOTAL)


def slide_results_main(prs, R):
    s = blank(prs)
    title(s, "06　结果", "全特征方案：四个模型接近 100%，朴素贝叶斯掉队", 8, TOTAL)
    image_fit(s, FIG / "fig04_scheme_heatmap.png", 0.5, 1.55, 7.1, 4.9)
    res = R["res"][R["res"]["model"].isin(MAIN)]
    a = res[res.scheme == "A"]
    a4 = a[a.model != "Gaussian NB"]
    c = res[res.scheme == "C"]
    bullets(s, [
        (f"方案 A：除 NB 外四个模型 F1 {f4(a4.test_f1.min())}–{f4(a4.test_f1.max())}", False),
        (f"朴素贝叶斯 {f4(row(res,'Gaussian NB','A')['test_f1'])}：独立假设拖累概率估计", False),
        (f"方案 C（仅 URL）：F1 {f4(c.test_f1.min())}–{f4(c.test_f1.max())}", False),
        ("A/C 的高分主要来自 URLSimilarityIndex，不能作为模型优劣的证据", False),
    ], 7.85, 1.75, 4.85, 2.8, size=11.2, spacing=8)
    callout(s, "结论：先审计捷径，再比较模型。", 7.85, 5.0, 4.85, 0.9, accent=ORANGE, size=13)
    footer(s, 8, TOTAL)


def slide_ablation(prs, R):
    s = blank(prs)
    title(s, "06　结果", "逐步去掉捷径：B/S/T 仍高，V 才见真章", 9, TOTAL)
    image_fit(s, FIG / "fig04_ablation_lines.png", 0.4, 1.5, 7.6, 4.95)
    res = R["res"][R["res"]["model"].isin(MAIN)]
    v = res[res.scheme == "V"].sort_values("test_f1", ascending=False)
    bullets(s, [
        ("B/S/T：", True), "随机森林与逻辑回归仍 > 0.996",
        "官方衍生特征只是第一层捷径；页面标志与 URL 词法仍携带大量信息",
        ("V：", True), f"逻辑回归 {f4(row(res,'Logistic Regression','V')['test_f1'])}，随机森林 {f4(row(res,'Random Forest','V')['test_f1'])}",
        f"决策树降到 {f4(row(res,'Decision Tree','V')['test_f1'])}，朴素贝叶斯仅 {f4(row(res,'Gaussian NB','V')['test_f1'])}",
    ], 8.2, 1.7, 4.6, 3.4, size=11.0, spacing=6)
    best = v.iloc[0]
    callout(s, f"V 方案最佳：{MODEL_CN[best.model]}，F1 {f4(best.test_f1)}", 8.2, 5.2, 4.55, 0.7, accent=BLUE, size=11.5)
    callout(s, "算法优势取决于特征类型，而不是排名本身。", 8.2, 6.0, 4.55, 0.65, accent=ORANGE, size=11)
    footer(s, 9, TOTAL)


def slide_model_diff(prs, R):
    s = blank(prs)
    title(s, "06　结果", "去掉捷径后：没有单一赢家", 10, TOTAL)
    res = R["res"][R["res"]["model"].isin(MAIN)]
    rows = []
    for m in MAIN:
        rows.append([MODEL_CN[m]] + [f4(row(res, m, sc)["test_f1"]) for sc in ["B", "D", "T", "V"]])
    table(s, ["模型", "B", "D", "T", "V"], rows, 0.6, 1.6, 6.1, 2.9,
          col_widths=[1.55, 1.1, 1.1, 1.1, 1.1], font_size=10.2)
    bullets(s, [
        ("随机森林：", True), "页面特征主导的 B/D 最好（Bagging 降方差）",
        ("逻辑回归：", True), "URL 词法主导的 T/V 最好，V 上 FPR 仅 0.015%",
        ("决策树：", True), "A 上调到 depth=3，V 上召回仅 67.6%；V 重选深度后 F1 0.994",
        ("朴素贝叶斯：", True), "S/T/V 上 F1 0.49–0.59；阈值降到 6e-5 也仅 0.692（概率失准）",
    ], 0.65, 4.75, 6.0, 1.8, size=10.4, spacing=3)
    image_fit(s, FIG / "fig06_roc.png", 6.9, 1.55, 5.9, 5.1)
    footer(s, 10, TOTAL)


def slide_optimize(prs, R):
    s = blank(prs)
    title(s, "07　优化", "调参收益有限，阈值可以换误报", 11, TOTAL)
    image_fit(s, FIG / "fig08_rf_optimization.png", 0.5, 1.55, 5.2, 3.2)
    image_fit(s, FIG / "fig10_threshold.png", 6.4, 1.55, 5.2, 3.2)
    opt = R["rfopt"]
    d0, d2 = opt.iloc[0], opt.iloc[-1]
    th = R["threshold"]
    dd = R["dt_depth"]
    bestd = dd.loc[dd["val_f1"].idxmax()]
    bullets(s, [
        (f"随机森林：默认 F1 {f4(d0['test_f1'])} → 重新调参 {f4(d2['test_f1'])}，收益很小", False),
        (f"决策树：最佳深度 {int(bestd['max_depth']) if bestd['max_depth'] < 99 else '不限'}，验证 F1 {f4(bestd['val_f1'])}；不限制时训练 {f4(dd.iloc[-1]['train_f1'])} / 验证 {f4(dd.iloc[-1]['val_f1'])}", False),
        (f"阈值：0.5 → {th['chosen_threshold']:.3f}，召回 {f4(th['default_0.5']['recall'])} → {f4(th['chosen']['recall'])}，FPR {pct(th['default_0.5']['fpr'],3)} → {pct(th['chosen']['fpr'],3)}", False),
        ("但逻辑回归在 V 上默认 FPR 仅 0.015%——换模型比调阈值更有效", False),
    ], 0.7, 4.95, 12.0, 1.45, size=10.6, spacing=4)
    callout(s, "当数据接近可分时，模型容量的边际收益很低；真正可控的是操作点。", 0.7, 6.25, 11.95, 0.6, accent=BLUE, size=11.5)
    footer(s, 11, TOTAL)


def slide_robust1(prs, R):
    s = blank(prs)
    title(s, "08　拓展", "样本越少、噪声越大，模型还稳吗？", 12, TOTAL)
    image_fit(s, FIG / "fig11_learning_curve.png", 0.45, 1.55, 6.2, 3.3)
    image_fit(s, FIG / "fig12_noise.png", 6.8, 1.55, 6.1, 3.3)
    lc = R["lc"].groupby(["model", "fraction"])["f1"].mean().reset_index()
    p1 = lc[lc.fraction == 0.01].set_index("model")["f1"]
    p100 = lc[lc.fraction == 1.0].set_index("model")["f1"]
    ln = R["label_noise"]
    ln0 = ln[ln.noise_rate == 0].set_index("model")["f1"]
    ln20 = ln[ln.noise_rate == 0.2].set_index("model")["f1"]
    drops = (ln0 - ln20).sort_values(ascending=False)
    best1 = p1.idxmax()
    bullets(s, [
        (f"1% 样本时最强的是 {MODEL_CN[best1]}（F1 {f4(p1.get(best1, np.nan))}）；100% 时 RF F1 {f4(p100.get('Random Forest', np.nan))}", False),
        (f"20% 标签噪声：{MODEL_CN[drops.index[0]]} 下降最多（{drops.iloc[0]:.4f}），{MODEL_CN[drops.index[-1]]} 最少（{drops.iloc[-1]:.4f}）", False),
        ("RF 的 Bagging 最稳；LR 的 C=100 正则化弱，对标签噪声最敏感", False),
        ("特征噪声 0.1sd 即让 F1 降到 0.62–0.69：模型依赖精确的字符比例", False),
    ], 0.7, 4.9, 12.0, 1.65, size=10.4, spacing=4)
    footer(s, 12, TOTAL)


def slide_robust2(prs, R):
    s = blank(prs)
    title(s, "08　拓展", "跨 TLD 迁移与过拟合：分布外才是真考验", 13, TOTAL)
    image_fit(s, FIG / "fig13_tld_transfer.png", 0.4, 1.55, 5.7, 3.1)
    image_fit(s, FIG / "fig14_overfitting.png", 6.25, 1.55, 6.7, 3.1)
    tld = R["tld"]
    base = tld[tld.split == "随机划分"].set_index("model")
    rare = tld[tld.split == "罕见 TLD"].set_index("model")
    gaps = (base["f1"] - rare["f1"]).sort_values(ascending=False)
    of = R["overfit"].sort_values("gap", ascending=False)
    bullets(s, [
        (f"罕见 TLD：F1 下降 {gaps.iloc[0]:.4f}（{MODEL_CN[gaps.index[0]]}）到 {gaps.iloc[-1]:.4f}（{MODEL_CN[gaps.index[-1]]}）", False),
        ("常见 TLD 占绝对多数，模型会用“TLD 是否常见”当捷径", False),
        (f"过拟合差距最大：{MODEL_CN[of.iloc[0]['model']]}，训练–验证 F1 差 {of.iloc[0]['gap']:.4f}", False),
        ("RF 差距 0.0034，LR/NB/SVM 均 <0.001：过拟合不是主要矛盾", False),
    ], 0.7, 4.8, 12.0, 1.5, size=11.2, spacing=5)
    footer(s, 13, TOTAL)


def slide_features(prs, R):
    s = blank(prs)
    title(s, "08　拓展", "模型到底在看什么？置换重要性与 SHAP", 14, TOTAL)
    image_fit(s, FIG / "fig16_permutation_importance.png", 0.35, 1.5, 5.9, 4.4)
    image_fit(s, FIG / "fig17_shap.png", 6.35, 1.5, 6.5, 4.4)
    perm = R["perm"]
    top = "、".join(perm.head(5)["feature"].tolist())
    bullets(s, [
        (f"置换重要性前五：{top}", False),
        ("去掉 URLSimilarityIndex 后，模型仍依赖页面规模与 HTTPS；URL 词法直接贡献较小", False),
        ("SHAP 排序一致并给出方向；约 20 个特征可保留大部分性能", False),
    ], 0.7, 5.9, 12.0, 1.0, size=10.2, spacing=3)
    footer(s, 14, TOTAL)


def slide_conclusions(prs, R):
    s = blank(prs)
    title(s, "09　结论", "五条结论", 15, TOTAL)
    s_ = R["shortcut"]
    res = R["res"][R["res"]["model"].isin(MAIN)]
    t = res[res.scheme == "T"].sort_values("test_f1", ascending=False)
    best = t.iloc[0]
    items = [
        f"数据存在多组单向捷径：两条规则即达 F1 {f4(s_['test_metrics']['f1'])}、FPR = 0。",
        "A/C 接近天花板，B/S/T 仍高；只有 V（仅 URL 基础词法）拉开差距。",
        f"没有单一赢家：页面特征主导时 RF 最好，URL 词法主导时 LR 最好（V 上 {f4(row(res,'Logistic Regression','V')['test_f1'])}）；DT/NB 崩塌。",
        "标签噪声、特征噪声与跨 TLD 迁移是主要风险；LR 对样本量最不敏感。",
        "部署前先做捷径审计与分布外验证，再谈准确率。",
    ]
    y = 1.65
    for i, text in enumerate(items, start=1):
        rect(s, 0.75, y, 0.55, 0.55, fill=BLUE, radius=True)
        textbox(s, str(i), 0.75, y + 0.05, 0.55, 0.45, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        textbox(s, text, 1.5, y + 0.02, 11.1, 0.55, size=14.5, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        y += 0.82
    callout(s, "模型为什么有效、何时失效，比分数本身更重要。", 0.75, 5.95, 11.8, 0.7, accent=ORANGE, size=13)
    footer(s, 15, TOTAL)


def slide_team(prs, R):
    s = blank(prs)
    title(s, "10　团队", "队伍贡献", 16, TOTAL)
    textbox(s, "姓名、学号按要求留空，由小组后续填写。", 0.75, 1.55, 8.0, 0.4, size=11.5, color=MUTED)
    positions = [(0.7, 2.1), (6.75, 2.1), (0.7, 4.25), (6.75, 4.25)]
    for (x, y), m in zip(positions, TEAM):
        rect(s, x, y, 5.8, 1.85, fill=LIGHT_BLUE, line=None, radius=True)
        textbox(s, m["member"], x + 0.2, y + 0.12, 1.2, 0.35, size=13, bold=True, color=BLUE)
        textbox(s, m["name_id"], x + 1.35, y + 0.12, 2.0, 0.35, size=12, color=MUTED)
        textbox(s, m["share"], x + 4.85, y + 0.08, 0.8, 0.4, size=16, bold=True, color=ORANGE, align=PP_ALIGN.RIGHT)
        textbox(s, m["tasks"], x + 0.2, y + 0.55, 5.4, 1.2, size=9.6, color=INK_2)
    callout(s, "四人按“数据—线性/概率—树/核—评估与汇报”分工，各 25%，每人均参与全部实验的讨论与复核。",
            0.7, 6.35, 11.9, 0.55, accent=BLUE, size=10.8)
    footer(s, 16, TOTAL)


def slide_thanks(prs, R):
    s = blank(prs)
    rect(s, 0, 0, 13.333, 7.5, fill=LIGHT_BLUE)
    rect(s, 0, 0, 0.18, 7.5, fill=BLUE)
    textbox(s, "谢谢，欢迎提问", 0.95, 2.0, 10.5, 0.9, size=38, bold=True, color=INK)
    textbox(s, "先审计捷径，再比较模型；先做分布外验证，再谈准确率。", 0.98, 3.2, 10.8, 0.6, size=17, color=INK_2)
    rect(s, 0.98, 4.0, 1.8, 0.07, fill=ORANGE)
    textbox(s, "代码、图表与实验记录：phiusiil-phishing/", 0.98, 4.35, 8.0, 0.4, size=11, color=MUTED)
    footer(s, 17, TOTAL)


WHITE = __import__("pptx").dml.color.RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_ORANGE = __import__("pptx").dml.color.RGBColor(0xFF, 0xF4, 0xEE)
SURFACE = __import__("pptx").dml.color.RGBColor(0xFC, 0xFC, 0xFB)
GRID = __import__("pptx").dml.color.RGBColor(0xE1, 0xE0, 0xD9)


def main():
    R = load_all()
    prs = new_deck()
    slide_cover(prs, R)
    slide_questions(prs, R)
    slide_data(prs, R)
    slide_shortcut(prs, R)
    slide_shortcuts2(prs, R)
    slide_design(prs, R)
    slide_models(prs, R)
    slide_results_main(prs, R)
    slide_ablation(prs, R)
    slide_model_diff(prs, R)
    slide_optimize(prs, R)
    slide_robust1(prs, R)
    slide_robust2(prs, R)
    slide_features(prs, R)
    slide_conclusions(prs, R)
    slide_team(prs, R)
    slide_thanks(prs, R)
    out = ROOT / "实验2汇报_钓鱼网站检测.pptx"
    prs.save(out)
    print(f"slides saved: {out}")


if __name__ == "__main__":
    main()
