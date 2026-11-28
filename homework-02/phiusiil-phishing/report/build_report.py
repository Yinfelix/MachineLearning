"""Build the Word report from experiment artefacts.

Run after scripts/01…07:
    python report/build_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from docx_helpers import (  # noqa: E402
    add_bullet, add_caption, add_equation, add_figure, add_h1, add_h2, add_h3, add_paragraph,
    add_rich_paragraph, add_table, mfrac, mr, msub, msup, page_break, set_run_font,
    setup_document, add_page_number_footer,
)
from load_metrics import FIG, load_all  # noqa: E402
from team import TEAM, TEAM_NOTE  # noqa: E402

MODEL_CN = {
    "Logistic Regression": "逻辑回归",
    "Gaussian NB": "高斯朴素贝叶斯",
    "Decision Tree": "决策树",
    "Random Forest": "随机森林",
    "SVM (RBF)": "RBF 支持向量机",
    "Linear SVM": "线性 SVM",
}
MAIN = ["Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "SVM (RBF)"]
SCHEME_CN = {
    "A": "A 全特征",
    "B": "B 去官方捷径",
    "C": "C 仅 URL",
    "D": "D 仅页面",
    "S": "S 严格去捷径",
    "T": "T 去掉 HTTPS",
    "U": "U 仅 URL 基础词法",
    "V": "V 仅 URL 基础词法（去 HTTPS）",
}


def f4(x):
    return f"{x:.4f}"


def pct(x, d=2):
    return f"{x*100:.{d}f}%"


def row(res, model, scheme):
    d = res[(res["model"].astype(str) == model) & (res["scheme"] == scheme)]
    return d.iloc[0] if len(d) else None


def build_header(doc):
    add_paragraph(doc, "机器学习课程设计　实验二", size=11, align=WD_ALIGN_PARAGRAPH.CENTER,
                  first_indent=False, line=1.0, before=6, after=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run("PhiUSIIL 钓鱼网站检测：捷径特征、模型比较与鲁棒性分析")
    set_run_font(r, 18, bold=True)

    add_caption(doc, "表 1　小组分工与贡献")
    rows = [[m["name_id"], f"{m['member']}：{m['tasks']}（{m['share']}）"] for m in TEAM]
    add_table(doc, ["姓名 / 学号", "具体任务及贡献"], rows, widths_in=[1.45, 4.3],
              font_size=8.8, header_fill="EDF2F9")


def build_section1(doc, R):
    a = R["audit"]
    add_h1(doc, "一、课题综述")
    add_h2(doc, "1.1　课题简介")
    add_paragraph(doc, (
        "钓鱼网站通过伪造银行、支付、云服务等页面诱导用户输入账号或验证码，是网络诈骗的常见入口。"
        "与传统黑名单相比，机器学习检测器可以利用网址词法、域名结构和页面内容，对未见过的网址给出风险评分。"
        "本项目要解决的问题是：给定一条网址及其页面信息，判断它是钓鱼网站还是合法网站。"
    ))
    add_paragraph(doc, (
        "我们使用 UCI 公开的 PhiUSIIL Phishing URL Dataset [1]。该数据集由 Prasad 与 Chandra 在 2024 年发布 [2]，"
        f"包含 {a['n_rows']:,} 条样本、{a['n_features_excluding_text']} 个可直接入模的特征，涵盖 URL 长度、特殊字符比例、"
        "域名与顶级域名、HTML 结构、表单与链接、关键词等。原始标签中 1 表示合法、0 表示钓鱼；"
        "我们把钓鱼重编码为正类（y=1），这样精确率、召回率都直接对应“抓钓鱼网站”这一目标。"
    ))
    add_paragraph(doc, '两条简单规则在留出的 46,994 条记录中只漏掉 5 个钓鱼网址。我们据此追问：如果删去相似度分数，模型还能利用哪些信息识别钓鱼样本？')

    add_h2(doc, "1.2　课题目标")
    add_paragraph(doc, '我们围绕三个问题组织实验。')
    add_paragraph(doc, '1. 单个特征能区分多少样本？模型移除这一列后损失多少性能？')
    add_paragraph(doc, "2. 五类算法在同一套严格划分下的表现有何差异，差异能否用算法原理和数据特点解释？", first_indent=False)
    add_paragraph(doc, "3. 当训练样本减少、标签被污染、特征被加噪或遇到罕见 TLD 时，模型是否仍然可靠？", first_indent=False)

    add_h2(doc, "1.3　课题数据集")
    add_paragraph(doc, (
        f"PhiUSIIL 数据集共 {a['n_rows']:,} 行、56 列，其中标签 1 列、文件名 1 列，其余 54 列为特征；"
        f"URL、Domain、Title 是文本列，FILENAME 是样本编号，均不直接入模，因此模型实际使用 {a['n_features_excluding_text']} 个特征。"
        f"数据没有缺失值，也没有完全重复的行；网址基本唯一（{a['unique_url']:,} 个不同 URL），"
        f"但 Domain 有 {a['unique_domains']:,} 个，其中 {a['rows_in_repeated_domains']:,} 行落在重复域名中，最大单域名包含 {a['max_domain_group']:,} 条样本。"
        "这说明如果随机划分，同一域名的样本可能同时出现在训练和测试中，评估可能高估对新域名的识别效果。"
    ))
    add_paragraph(doc, '全量数据含 134,850 条合法记录和 100,945 条钓鱼记录。原始标签 0 表示钓鱼；建模时将钓鱼记为正类 1。')
    add_caption(doc, "表 2　数据集概况与最终划分")
    tr, te = a["train_test"]["train"], a["train_test"]["test"]
    rows = [
        ["样本数 / 原始列数", f"{a['n_rows']:,} / {a['n_columns']}"],
        ["可直接入模特征", f"{a['n_features_excluding_text']}（含 1 个类别特征 TLD）"],
        ["类别分布（合法 : 钓鱼）", f"{a['label_counts']['1']:,} : {a['label_counts']['0']:,}（{pct(a['label_counts']['0']/a['n_rows'])} 钓鱼）"],
        ["缺失值 / 完全重复行", f"{a['missing_total']} / {a['duplicate_rows']}"],
        ["Domain 数 / 重复域名行数", f"{a['unique_domains']:,} / {a['rows_in_repeated_domains']:,}"],
        ["训练集", f"{tr['n']:,}（钓鱼 {tr['phishing']:,}，{pct(tr['phishing_rate'])}）"],
        ["独立测试集", f"{te['n']:,}（钓鱼 {te['phishing']:,}，{pct(te['phishing_rate'])}）"],
        ["训练/测试 Domain 重叠", f"{a['train_test']['group_overlap']} 个"],
    ]
    add_table(doc, ["项目", "数值"], rows, widths_in=[2.25, 3.5], font_size=9)



def build_section2(doc, R):
    a = R["audit"]
    uni = R["uni"]
    add_h1(doc, "二、数据处理")

    add_h2(doc, "2.1　数据审计与分布")
    add_figure(doc, FIG / "fig01_dataset_overview.png", width_in=4.4,
               caption="图 1　训练集类别分布、URLSimilarityIndex 分布、域名重复情况与常见 TLD")

    add_paragraph(doc, '训练集有 107,915 条合法记录和 80,886 条钓鱼记录，其中 44 个域名同时出现两类标签。划分时，同一 Domain 的全部记录放在同一侧。')
    add_h2(doc, "2.2　捷径特征的发现：这个数据集有多容易？")
    add_paragraph(doc, '在 80,886 条钓鱼训练记录中，636 条的 URLSimilarityIndex 等于 100。合法训练记录的这一列均为 100。以相似度由低到高排列，单列识别钓鱼样本的 ROC-AUC 为 0.9961。')
    add_paragraph(doc, (
        f"第二个单向捷径是 IsHTTPS：全部合法样本都使用 HTTPS，而钓鱼样本中只有 {pct(a['is_https_by_label']['0'])} 使用 HTTPS。"
        "按方向校正后的单变量 AUC 显示：除了 URLSimilarityIndex，"
        "LineOfCode、NoOfExternalRef、NoOfImage、NoOfSelfRef、NoOfJS、NoOfCSS 等页面规模特征的 AUC 也在 0.96–0.99 之间。"
        "它们不是官方标注的“衍生特征”，但同样把两类样本推得很开。"
    ))
    s = R["shortcut"]
    add_paragraph(doc, '将相似度小于 100 或 IsHTTPS 等于 0 的网址判为钓鱼，在留出集中得到 0 次误报和 5 次漏报，F1 为 0.99988。这说明数据中存在很强的判别信号。判断是否存在标签泄漏，还需要核实相似度的计算过程和参考库来源。')
    add_caption(doc, "表 3　两个单向捷径特征的统计")
    rows = [
        ["URLSimilarityIndex", "合法恒为 100；钓鱼训练均值 49.53，0.79% 等于 100", "0.9961", "小于 100 判钓鱼"],
        ["IsHTTPS", "合法 100% 为 1；钓鱼训练样本中 49.15% 为 1", "0.7542", "等于 0 判钓鱼"],
        ["两条规则合并", "测试集仅漏报 5 条，误报 0 条", "—", "F1 0.99988"],
    ]
    add_table(doc, ["特征 / 规则", "取值特点", "方向校正 AUC", "使用方式"], rows,
              widths_in=[1.3, 2.4, 0.95, 1.1], font_size=8.8)

    add_h2(doc, "2.3　预处理")
    add_paragraph(doc, '主实验保留长度、数量和比例列的原始尺度。数值缺失用训练折中位数填补，TLD 按训练折出现次数编码，未见类别记为 0。逻辑回归与 SVM 另做标准化，树模型和 GaussianNB 使用未缩放输入。')
    add_paragraph(doc, '原始 URL、Domain、Title 与文件名不直接入模；Domain 用于分组。')

    add_h2(doc, "2.4　数据划分与防泄漏")
    add_paragraph(doc, '外层按 Domain 分组留出 46,994 条样本，训练和留出两侧的域名交集为零。其余 188,801 条样本按域名做五折交叉验证。后续树深度和贝叶斯阈值的事后诊断也查看过同一留出集，这部分结果作为探索性比较。')
    add_paragraph(doc, '主表中的 RBF SVM 使用训练集内固定的 20,000 条分层子样本，其余模型使用完整训练集。第 4.4 节另用同一子样本比较两种 SVM。')

    add_h2(doc, "2.5　八套特征方案")
    add_paragraph(doc, '表 4 描述两条删列路径。A→B→S→T 依次去除衍生特征、六项页面规模特征和 HTTPS；C→U→V 从 URL 输入中去除四项衍生特征和 HTTPS。D 单独考察页面信息。')
    add_caption(doc, "表 4　特征方案与特征数")
    rows = [
        ["A 全特征", "51", "全部数值特征 + TLD", "基线，预计接近 100%"],
        ["B 去官方捷径", "45", "A 去掉 6 个官方衍生特征", "检验官方衍生特征的贡献"],
        ["C 仅 URL", "23", "22 个 URL 数值特征 + TLD", "只看网址能做到多好"],
        ["D 仅页面", "28", "28 个页面内容特征", "只看页面能做到多好"],
        ["S 严格去捷径", "39", "B 去掉 6 个页面规模特征", "去掉第二组捷径"],
        ["T 去掉 HTTPS", "38", "S 再去掉 IsHTTPS", "最严格的全特征消融"],
        ["U 仅 URL 基础词法", "19", "C 去掉 4 个 URL 衍生特征", "严格的 URL-only 对照"],
        ["V 仅 URL 基础词法（去 HTTPS）", "18", "U 再去掉 IsHTTPS", "最严格的 URL-only 对照"],
    ]
    add_table(doc, ["方案", "特征数", "构成", "用途"], rows, widths_in=[1.55, 0.6, 2.45, 1.15], font_size=8.5)


def tr_rate(R):
    return R["audit"]["train_test"]["train"]["phishing_rate"]


def te_rate(R):
    return R["audit"]["train_test"]["test"]["phishing_rate"]


def build_section3(doc, R):
    add_h1(doc, "三、模型构建")
    add_paragraph(doc, '实验比较逻辑回归、高斯朴素贝叶斯、决策树、随机森林和 RBF 支持向量机，另列线性支持向量机作对照。各模型通过 scikit-learn Pipeline 训练 [5]。')

    add_h2(doc, "3.1　逻辑回归")
    add_paragraph(doc, (
        "逻辑回归假设正类的对数几率是特征的线性组合，再用 Sigmoid 函数把线性输出映射到 (0,1)："
    ))
    add_equation(doc, mr("p(y=1 | x) = ") + mfrac(mr("1"), mr("1 + exp(−(") + msup(mr("w"), mr("T")) + mr("x + b))")))
    add_paragraph(doc, (
        "训练时最小化带正则化的交叉熵。L2 正则化把系数整体压小，L1 正则化可以把部分系数压到 0，起到特征选择的作用。"
        "关键超参数是正则化强度的倒数 C：C 越小，正则化越强。我们选择逻辑回归，是因为它训练快、系数可解释，"
        "并且能告诉我们“在这个特征空间里，类别是否已经接近线性可分”。"
    ))

    add_h2(doc, "3.2　高斯朴素贝叶斯")
    add_paragraph(doc, (
        "朴素贝叶斯用贝叶斯公式计算后验概率，并假设各特征在给定类别下条件独立 [7]。高斯版本进一步假设每个特征在每个类别下服从正态分布，"
        "均值与方差从训练数据估计。关键超参数 var_smoothing 在方差上加一个很小的平滑项，避免某个特征方差为 0 时概率计算失败。"
        "选择它，是为了检验“特征独立假设”在一组高度相关的特征上会被破坏到什么程度；它也是训练成本最低的生成式基线。"
    ))

    add_h2(doc, "3.3　决策树")
    add_paragraph(doc, (
        "决策树递归地选择使子节点不纯度下降最多的特征与阈值，把样本不断划分到叶节点，再用叶节点中多数类作为预测。"
        "我们使用基尼不纯度："
    ))
    add_equation(doc, mr("Gini(t) = 1 − ") + msub(mr("Σ"), mr("k")) + msup(msub(mr("p"), mr("tk")), mr("2")))
    add_paragraph(doc, (
        "关键超参数是最大深度 max_depth 与叶节点最小样本数 min_samples_leaf。树越深、叶节点越小，模型越容易记住训练集中的个别样本，"
        "产生过拟合。选择决策树，是因为它的规则直观、容易可视化，也方便展示剪枝前后训练/验证曲线的变化。"
    ))

    add_h2(doc, "3.4　随机森林")
    add_paragraph(doc, (
        "随机森林由 Breiman 提出 [3]，训练多棵决策树，每棵树使用自助采样（Bootstrap）得到的训练子集，并在每次划分时只考虑随机抽取的一部分特征；"
        "预测时对所有树的输出取平均（分类任务取多数投票）："
    ))
    add_paragraph(doc, (
        "这种“样本随机 + 特征随机”的双重扰动降低了树与树之间的相关性，从而降低方差。关键超参数包括树的数量 n_estimators、"
        "最大深度 max_depth、每次划分考虑的特征比例 max_features 和叶节点最小样本数 min_samples_leaf。"
        "随机森林通常是表格数据上最稳的强基线，因此我们把它作为重点优化对象。"
    ))

    add_h2(doc, "3.5　支持向量机")
    add_paragraph(doc, (
        "支持向量机由 Cortes 与 Vapnik 提出 [4]，寻找一个使两类间隔最大的决策边界。对于线性不可分的数据，RBF 核把样本隐式映射到高维空间："
    ))
    add_paragraph(doc, (
        "关键超参数是惩罚系数 C 与核宽度 γ。C 越大，模型越不允许训练样本越过间隔，容易过拟合；γ 越大，单个样本的影响范围越小，"
        "决策边界越弯曲。选择 SVM，是为了与线性模型和树模型形成对照：它依赖距离与间隔，对特征尺度敏感，"
        "理论上能学到非线性边界，但在二十多万样本上训练成本较高，因此只在 20,000 条分层子样本上训练。"
    ))

    add_h2(doc, "3.6　超参数搜索")
    add_paragraph(doc, (
        "所有模型先在方案 A 上用完整网格和训练集内部的 5 折分组交叉验证选择超参数，评分指标为钓鱼类的 F1；"
        "选定后固定用于 A–D、S–V 全部特征方案。这样做的目的，是让消融实验的差异只来自特征集合，"
        "而不是每个方案各自调参带来的额外自由度。RBF SVM 在 20,000 条子样本上做同样的 5 折搜索；"
        "LinearSVC 在全量训练集上搜索 C。表 5 汇总了搜索空间与最终选中的参数。"
    ))
    add_caption(doc, "表 5　超参数搜索空间与最佳参数")
    t = R["tuning"]
    rows = []
    spec_text = {
        "Logistic Regression": "C ∈ {0.01,0.1,1,10,100}；L2 正则（L1 另做路径分析）",
        "Gaussian NB": "var_smoothing ∈ {1e-9,…,1e-5}",
        "Decision Tree": "max_depth ∈ {3,5,8,12,None}；min_samples_leaf ∈ {1,5,20,50}",
        "Random Forest": "n_estimators ∈ {100,300,500}；max_depth ∈ {None,20}；max_features ∈ {sqrt,0.3,0.5}；min_samples_leaf ∈ {1,5}",
        "SVM (RBF)": "C ∈ {0.1,1,10}；gamma ∈ {scale,0.01,0.1}（20k 子样本）",
        "Linear SVM": "C ∈ {0.01,0.1,1,10}（全量）",
    }
    for name in ["Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "SVM (RBF)", "Linear SVM"]:
        if name not in t:
            continue
        best = t[name]["best_params"]
        best_str = ", ".join(f"{k}={v}" for k, v in best.items())
        rows.append([MODEL_CN[name], spec_text[name], best_str, f4(t[name]["best_cv_f1"])])
    add_table(doc, ["模型", "搜索空间", "最佳参数", "CV F1"], rows,
              widths_in=[1.05, 2.15, 1.75, 0.65], font_size=7.8)
    add_paragraph(doc, "注：RBF SVM 的 CV F1 在 20,000 条分层子样本上计算，其余模型使用全量训练集。",
                  size=9, first_indent=False, line=1.1, before=2, after=4, color="52514E")


def build_section4(doc, R):
    res = R["res"][R["res"]["model"].isin(MAIN)]
    add_h1(doc, "四、模型训练与评估")

    add_h2(doc, "4.1　评价指标与比较协议")
    add_paragraph(doc, (
        "本项目把钓鱼样本作为正类。混淆矩阵的四个格子分别为：真正例 TP（钓鱼判钓鱼）、假正例 FP（合法判钓鱼，即误报）、"
        "假反例 FN（钓鱼判合法，即漏报）、真反例 TN（合法判合法）。我们使用以下指标："
    ))
    add_equation(doc, mr("P = ") + mfrac(mr("TP"), mr("TP + FP")) + mr("，  R = ") + mfrac(mr("TP"), mr("TP + FN")))
    add_equation(doc, mr("F1 = ") + mfrac(mr("2PR"), mr("P + R")) + mr("，  FPR = ") + mfrac(mr("FP"), mr("FP + TN")))
    add_paragraph(doc, '精确率表示报警中钓鱼样本的比例，召回率表示实际钓鱼样本被识别的比例。调参以钓鱼类 F1 为准，另报合法网址误报率、ROC-AUC 和平均精确率 AP。下文分别标注交叉验证、固定验证折及外层留出的结果。')

    add_h2(doc, "4.2　测试集结果：全特征方案没有区分度")
    a = res[res.scheme == "A"]
    a4 = a[a.model != "Gaussian NB"]
    nb_a = a[a.model == "Gaussian NB"].iloc[0]
    add_paragraph(doc, '表 6 列出五个主模型的留出集 F1。方案 A 的决策树无错分；逻辑回归有 1 次误报和 6 次漏报。即使两者四舍五入后都接近满分，错误数仍可直接比较。')
    add_caption(doc, "表 6　测试集 F1（正类 = 钓鱼）")
    schemes = ["A", "B", "C", "D", "S", "T", "V"]
    rows = []
    for model in ["Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "SVM (RBF)"]:
        vals = []
        for s in schemes:
            r = row(res, model, s)
            vals.append(f4(r["test_f1"]) if r is not None else "—")
        rows.append([MODEL_CN[model]] + vals)
    add_table(doc, ["模型", "A", "B", "C", "D", "S", "T", "V"], rows,
              widths_in=[1.0, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66], font_size=8.4)
    add_h2(doc, "4.3　消融：捷径不止一层")
    add_paragraph(doc, '在训练内第 1 折，仅移除 URLSimilarityIndex，逻辑回归 F1 从 0.999938 降至 0.999659。相似度单独使用时区分度很高，同时模型还能利用其他字段补偿其缺失。图 2 分别展示保留页面和仅用 URL 的两条消融路径。')
    add_figure(doc, FIG / "fig04_ablation_lines.png", width_in=5.55,
               caption="图 2　两条消融路径上的测试集 F1：(a) A → B → S → T；(b) C → U → V")
    add_paragraph(doc, 'S 和 T 中，高斯朴素贝叶斯的 F1 已降至约 0.493。进一步限制到 V 的 18 项 URL 特征后，各模型的误报和漏报也呈现不同分布。')

    add_h2(doc, "4.4　去掉捷径后，哪种模型更好？")
    def best_on(scheme):
        return res[res.scheme == scheme].sort_values("test_f1", ascending=False).iloc[0]

    bB, bD, bT, bV = best_on("B"), best_on("D"), best_on("T"), best_on("V")
    add_paragraph(doc, '在仅页面方案 D，随机森林在五个主模型中 F1 最高（0.9970）。在 V，逻辑回归为 0.996022，随机森林为 0.995108。另列的全量 LinearSVC 为 0.996071，与逻辑回归只差约 0.00005。')
    add_paragraph(doc, '在 V 上，高斯朴素贝叶斯默认阈值的召回率为 0.4191。我们在训练集内比较三种输入表示，各自在其余四折选择平滑参数。第 1 折用 TLD 原始计数时 F1 为 0.6189，改用相对频率后为 0.6815，Brier 分数也从 0.2315 降至 0.2060。输入尺度影响了本次结果。')
    ph = R["posthoc"]
    add_paragraph(doc, '决策树沿用 A 上选出的深度 3 时，V 的留出 F1 为 0.7835。用训练内验证重新为 V 选择深度后，不限深度的树在原留出集上达到 0.9939。前一个数字衡量的是固定参数迁移的表现。')
    add_paragraph(doc, '为对照训练规模，我们还在同一 20,000 条子样本内，用完全相同的 16,053 条训练记录和 3,947 条验证记录比较两种 SVM。V 上 RBF 与线性核对应的验证 F1 分别为 0.9922 和 0.9913。')


def build_section5(doc, R):
    res = R["res"][R["res"]["model"].isin(MAIN)]
    add_h1(doc, "五、结果可视化与模型优化")

    add_h2(doc, "5.1　混淆矩阵与预测概率")
    lr_v = row(res, "Logistic Regression", "V")
    dt_v = row(res, "Decision Tree", "V")
    add_paragraph(doc, (
        "图 3 对比了逻辑回归与决策树在 V 方案上的混淆矩阵（括号内为按真实类别归一化的比例）。"
        f"逻辑回归只误报 {int(lr_v['test_fp'])} 条、漏报 {int(lr_v['test_fn'])} 条，钓鱼类召回率 {pct(lr_v['test_recall'])}；"
        f"决策树误报 {int(dt_v['test_fp'])} 条、漏报 {int(dt_v['test_fn'])} 条，召回率只有 {pct(dt_v['test_recall'])}。"
        "两者面对的是同一组 URL 基础词法特征，差别来自模型容量与边界形状：线性模型可以把多个连续特征平滑地组合，"
        "而深度 3 的树只能做很少几次硬切分。"
    ))
    add_figure(doc, FIG / "fig20_confusion.png", width_in=5.55,
               caption="图 3　方案 V 上逻辑回归与决策树的混淆矩阵")
    add_h2(doc, "5.2　ROC 与 PR 曲线")
    auc_rank = res[res.scheme == "V"].sort_values("test_roc_auc", ascending=False)
    add_paragraph(doc, 'ROC-AUC 衡量模型将钓鱼网址排在合法网址前面的能力。V 上浅树 F1 为 0.7835，GaussianNB 为 0.5870；阈值下的 F1 还取决于最终将多少样本判为钓鱼。逐模型 ROC-AUC 与 AP 见 results_all.csv。')

    add_h2(doc, "5.3　随机森林调参：收益有多大？")
    add_paragraph(doc, (
        "我们比较了三种随机森林配置：默认参数（100 棵树，其余取默认值）、在方案 A 上调参得到的最佳参数、以及专门在方案 B 上重新网格搜索得到的最佳参数。"
        "三者都在方案 B 上做 5 折交叉验证并评估测试集，结果见表 7。"
    ))
    add_caption(doc, "表 7　随机森林调参前后的性能（方案 B）")
    opt = R["rfopt"]
    rows = []
    for _, r in opt.iterrows():
        rows.append([r["config"], f4(r["cv_f1_mean"]), f4(r["test_f1"]),
                     f4(r["test_accuracy"]), pct(r["test_fpr"], 3), int(r["test_fp"] + r["test_fn"])])
    add_table(doc, ["配置", "CV F1", "测试 F1", "准确率", "FPR", "错误数"], rows,
              widths_in=[1.7, 0.72, 0.75, 0.75, 0.7, 0.6], font_size=8.4)
    d0 = opt.iloc[0]
    d2 = opt.iloc[-1]
    delta = d2["test_f1"] - d0["test_f1"]
    add_paragraph(doc, '在 B 上，默认森林误报 1 次、漏报 5 次；重新调参后误报 2 次、漏报仍为 5 次。增加搜索成本没有减少这批留出样本上的错误。')

    add_h2(doc, "5.4　决策树剪枝：从记住样本到学习规律")
    dd = R["dt_depth"]
    best_depth = dd.loc[dd["val_f1"].idxmax()]
    best_depth_label = int(best_depth["max_depth"]) if best_depth["max_depth"] < 99 else "不限制"
    add_paragraph(doc, (
        "图 4(a) 展示了决策树随最大深度变化的训练与验证 F1（方案 B，5 折均值）。"
        "深度为 1–2 时明显欠拟合；深度增加到 10–20 时，验证 F1 升到 0.9987 左右；"
        "不限制深度时训练 F1 达到 1.0、验证 F1 为 0.99865，过拟合迹象存在，但幅度不大。"
        f"验证 F1 在深度 {best_depth_label} 附近达到最高，继续加深不再提高。"
        "图 4(b) 的代价复杂度剪枝曲线给出类似的结论：ccp_alpha 从 0 增大时，训练 F1 下降，验证 F1 先小幅上升后下降，"
        "说明适当剪枝可以去掉只对个别样本有效的分裂，但在 B 上收益有限。"
    ))
    add_figure(doc, FIG / "fig09_dt_pruning.png", width_in=5.55,
               caption="图 4　决策树的深度曲线与代价复杂度剪枝曲线（方案 B）")
    add_paragraph(doc, '深度 20 的五折验证 F1 为 0.998686，不限深度为 0.998649。两者相差不到 0.00004，现有验证结果不足以区分这两个设置。')

    add_h2(doc, "5.5　阈值优化：召回率与误报的权衡")
    th = R["threshold"]
    add_paragraph(doc, '对方案 V 的随机森林，用训练集五折的折外预测（OOF）选择阈值。每条概率由训练时未见该行的模型给出。我们选择满足折外召回率至少 99% 的最高候选阈值，再到留出集对照默认阈值 0.5。SVM 的类别预测使用其决策函数。')
    add_caption(doc, "表 8　随机森林（方案 V）在默认阈值与优化阈值下的测试集表现（阈值在 OOF 上选择）")
    d05, dch = th["default_0.5"], th["chosen"]
    rows = [
        ["0.500（默认）", f4(d05["recall"]), f4(d05["precision"]), f4(d05["f1"]), pct(d05["fpr"], 3), f"{d05['fp']} / {d05['fn']}"],
        [f"{th['chosen_threshold']:.3f}（优化）", f4(dch["recall"]), f4(dch["precision"]), f4(dch["f1"]), pct(dch["fpr"], 3), f"{dch['fp']} / {dch['fn']}"],
    ]
    add_table(doc, ["阈值", "召回率", "精确率", "F1", "FPR", "FP / FN"], rows,
              widths_in=[1.25, 0.8, 0.8, 0.75, 0.75, 0.8], font_size=8.6)
    add_paragraph(doc, '阈值从 0.5 提到 0.735，误报从 71 次减至 22 次，漏报从 125 次增至 227 次。留出集召回率为 98.87%，低于折外选择时设定的 99%。是否采用这一阈值，取决于误封正常站点与放过钓鱼站点的相对代价。')


def build_section6(doc, R):
    add_h1(doc, "六、拓展任务")

    add_h2(doc, "6.1　泛化与鲁棒性")
    add_paragraph(doc, '鲁棒性实验使用方案 V 和训练内的域名分组。决策树用折 2–4 训练、折 0 选深度；折 1 的 37,796 条记录用于学习曲线、扰动和 TLD 对照。')

    add_h3(doc, "（1）训练样本量：谁最“省样本”？")
    lc = R["lc"]
    add_paragraph(doc, '从其余四折按类别比例抽取 1%、5%、10%、25% 和 50% 的记录，每个比例使用三个种子。100% 直接使用完整训练池，只训练一次。')
    add_figure(doc, FIG / "fig11_learning_curve.png", width_in=5.0,
               caption="图 5　训练样本比例对验证 F1 的影响（方案 V，子样本重复三次）")
    lc_sum = lc.groupby(["model", "fraction"])["f1"].mean().reset_index()
    p1 = lc_sum[lc_sum.fraction == 0.01].set_index("model")["f1"]
    p100 = lc_sum[lc_sum.fraction == 1.0].set_index("model")["f1"]
    best1, best100 = p1.idxmax(), p100.idxmax()
    add_paragraph(doc, '使用 1% 样本时，逻辑回归、决策树和随机森林的平均验证 F1 分别为 0.9754、0.9375 和 0.9510。使用全量训练池时分别为 0.9963、0.9942 和 0.9956；三者的差距随样本量增加而缩小。')

    add_h3(doc, "（2）标签噪声与特征噪声")
    ln, fn = R["label_noise"], R["feat_noise"]
    add_paragraph(doc, (
        "标签噪声实验把训练集中 0%、5%、10%、20% 的标签随机翻转，验证集标签保持干净；"
        "特征噪声实验在验证集的连续特征上加入 0、0.1、0.5、1.0 倍训练集标准差的高斯噪声，模型本身不重新训练。"
        "两组实验分别模拟“标注被污染”和“线上特征有测量误差”的情况。"
    ))
    add_figure(doc, FIG / "fig12_noise.png", width_in=5.55,
               caption="图 6　标签噪声与特征噪声对验证 F1 的影响（方案 V）")
    ln0 = ln[ln.noise_rate == 0].set_index("model")["f1"]
    ln20 = ln[ln.noise_rate == 0.2].set_index("model")["f1"]
    drops = (ln0 - ln20).sort_values(ascending=False)
    add_paragraph(doc, '翻转 20% 的训练标签后，逻辑回归 F1 从 0.9963 降至 0.8167，随机森林从 0.9956 降至 0.9670。图 6 的特征扰动按各列训练标准差生成，可能打破计数、比例之间的约束。这是一次数值压力测试。')

    add_h3(doc, "（3）异常值与缩放方式")
    out = R["outliers"]
    add_paragraph(doc, '在 5% 的验证样本中将三个计数特征改为各自的训练均值加 15 倍标准差。标准缩放和稳健缩放的 F1 分别从 0.9963、0.9971 降至 0.9715、0.9723。模型已固定，扰动改变了样本相对决策边界的位置。')

    add_h3(doc, "（4）跨 TLD 迁移")
    tld = R["tld"]
    add_paragraph(doc, '用训练池最常见的 20 种顶级域名（TLD）训练，在同一验证折比较常见组 30,473 条与罕见组 7,323 条。两组与训练数据共享域名数均为零。钓鱼占比分别为 39.36% 和 56.32%。')
    base = tld[tld.split == "随机划分"].set_index("model")
    rare = tld[tld.split == "罕见 TLD"].set_index("model")
    gaps = (base["f1"] - rare["f1"]).sort_values(ascending=False)
    add_paragraph(doc, '随机森林从常见到罕见组的 F1 由 0.995698 降至 0.988416；逻辑回归从 0.995814 微升至 0.996360，但其合法网址误报率从 0.0108% 升至 0.3751%。删除 TLD 编码后，森林两组的 F1 仍有差距（0.995698、0.989904）。')

    add_h3(doc, "（5）过拟合分析")
    of = R["overfit"]
    worst_gap = of.sort_values("gap", ascending=False).iloc[0]
    add_paragraph(doc, '在训练池内比较训练与验证 F1，决策树的差距约 0.0054，随机森林约 0.0035。这个差距衡量同一数据来源内的表现，跨来源效果仍待检验。')

    add_h2(doc, "6.2　特征工程与可解释性")
    add_paragraph(doc, (
        "这一节回答“模型到底在看什么，多余特征能不能去掉”。所有特征选择与降维都放在 Pipeline 内，只在训练折上拟合，"
        "避免用验证集或测试集挑选特征。"
    ))

    add_h3(doc, "（1）相关性与单变量筛选")
    cs = R["corr_summary"]
    hc = R["high_corr"]
    add_paragraph(doc, (
        f"在训练集的 {cs.get('n_numeric_features', 50)} 个数值特征中，有 {cs.get('n_pairs_abs_rho_ge_095', 0)} 对特征的 "
        f"|Spearman ρ| ≥ 0.95，{cs.get('n_pairs_abs_rho_ge_090', 0)} 对 ≥ 0.90。"
        "最极端的是 HasObfuscation、NoOfObfuscatedChar、ObfuscationRatio 三个变量，两两相关系数等于 1.0；"
        "NoOfDegitsInURL 与 DegitRatioInURL 的相关系数为 0.997；URLLength 与 NoOfLettersInURL 为 0.972。"
        "这些高度相关的特征会放大朴素贝叶斯的“重复证据”问题，也让线性模型的系数难以单独解释。"
    ))
    add_paragraph(doc, '单变量 AUC 按能区分两类的方向校正后排序；互信息另在方案 B 的数值列上计算，两份名单的候选列不同。')

    add_h3(doc, "（2）过滤法、嵌入法与降维")
    kdf, pca_df = R["kbest"], R["pca"]
    add_paragraph(doc, (
        "我们用 SelectKBest（f_classif）在 Pipeline 内比较 K = 5、10、20 与全部特征；"
        "用 PCA 比较保留 90%、95%、99% 方差；用 L1 逻辑回归的正则化路径观察非零系数个数；"
        "再用随机森林的 impurity importance 做嵌入法排序。"
    ))
    bestk = kdf.sort_values("cv_f1_mean", ascending=False).iloc[0]
    add_paragraph(doc, '方案 B 下，随机森林保留 20 列时五折 F1 为 0.999350，使用全部 45 列时为 0.999858。另一组 PCA 对照中，逻辑回归保留 95% 方差需要 34 个主成分，F1 为 0.999708。')
    l1 = R["l1"]
    add_paragraph(doc, '在测过的 C=0.001 至 1 范围内，L1 逻辑回归的非零系数从 13 个增至 45 个。该表给出训练样本上的系数路径。')

    add_h3(doc, "（3）特征升维：线性模型能否追上非线性模型？")
    poly = R["poly"]
    add_paragraph(doc, '每个训练折用随机森林在数值列中选出前 10 项。线性和二阶逻辑回归共用该折选出的列，并在相同验证行上比较。')
    lin = poly[poly["mode"] == "linear"].iloc[0]
    quad = poly[poly["mode"] == "quadratic"].iloc[0]
    add_paragraph(doc, '五折平均 F1 从 0.996275 增至 0.996819，增加 0.000544。二阶展开把 10 列扩为 65 列，本次观察到的额外收益较小。')

    add_h3(doc, "（4）置换重要性与 SHAP")
    perm = R["perm"]
    add_paragraph(doc, '用内折 1–4 训练方案 B 的森林，留出折 0 做解释。从折 0 抽取 20,000 行逐列置换，以 F1 下降量衡量该模型在这一折上对各列的依赖。')
    top_perm = perm.head(5)["feature"].tolist()
    add_paragraph(doc, '置换 LineOfCode、IsHTTPS、NoOfSubDomain 后，F1 平均下降 0.1420、0.1307、0.0365。相关特征可能相互替代，因此单列的下降量需要结合分组消融解读。')
    add_paragraph(doc, '图 7 展示同一留折中 2,000 个样本的 SHAP 值 [6]。每个点表示一个样本；横轴表示该特征对钓鱼类模型输出的贡献，颜色表示特征值的高低。')
    add_figure(doc, FIG / "fig17_shap.png", width_in=4.9,
               caption="图 7　随机森林（方案 B）的 SHAP 摘要图（验证集 2,000 条）")

    add_h3(doc, "（5）浅层决策树规则")
    add_paragraph(doc, '在方案 B 上另拟合深度 3 的树，根节点按外部引用数 NoOfExternalRef 分裂。这个例子说明，移除相似度后，浅树仍可利用页面结构区分样本。')



def build_section7(doc, R):
    res = R["res"][R["res"]["model"].isin(MAIN)]
    s = R["shortcut"]
    add_h1(doc, "七、总结")

    add_h2(doc, "7.1　主要结论")
    add_paragraph(doc, '两条规则在本次留出集上只漏报 5 次。但在训练内的单列消融中，逻辑回归移除相似度后仍达到 0.999659。两项观察共同说明该数据中存在多个可以相互补充的判别信号。')
    add_paragraph(doc, '模型比较中，差异最大的一项来自树深度：在 V 上重新选择深度，决策树 F1 从 0.7835 升至 0.9939。固定参数消融回答的是模型换一组输入后如何表现。')
    add_paragraph(doc, '跨顶级域名时应同时查看误报和漏报。罕见组中逻辑回归的 F1 略升，合法站点误报率却同时上升。若系统用于拦截网址，后一个数字直接影响用户。')

    add_h2(doc, "7.2　局限与未来工作")
    add_paragraph(doc, '实验只使用一个数据来源和一次外层划分。按完整域名分组可隔离同名域的记录，但同一注册主域下的不同子域仍可能分居两侧。现有结果无法估计新来源或未来时段上的识别率。')
    add_paragraph(doc, '下一步可固定特征、模型和阈值，在另一来源按相同口径获取 URL 与页面字段，分别统计误封的合法网址和漏掉的钓鱼网址。')

    add_h2(doc, "参考资料")
    refs = [
        "[1] UCI Machine Learning Repository. PhiUSIIL Phishing URL Dataset (ID 967). https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
        "[2] Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and lexical features. Computers & Security, 136, 103545. DOI: 10.1016/j.cose.2023.103545",
        "[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
        "[4] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
        "[5] scikit-learn developers. Supervised Learning — User Guide. https://scikit-learn.org/stable/supervised_learning.html",
        "[6] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.",
        "[7] 周志华. 机器学习. 清华大学出版社, 2016.",
    ]
    for ref in refs:
        add_paragraph(doc, ref, size=10.5, first_indent=False, line=1.25, after=2)


def build_appendix(doc, R):
    add_h1(doc, "附录")
    add_h2(doc, "运行环境与复现")
    add_paragraph(doc, (
        "Python 3.13.9；numpy 2.3.5；pandas 2.3.3；scipy 1.16.3；scikit-learn 1.7.2；"
        "matplotlib 3.10.6；seaborn 0.13.2；shap 0.54.1；python-docx 1.2.0；python-pptx 1.0.2。"
        "划分种子为 42，学习曲线子样本种子为 42、52、62。完整运行顺序与依赖见项目 README.md；"
        "实验记录汇总在 outputs/metrics/results_all.csv，图表在 outputs/figures/。"
    ))



def main():
    R = load_all()
    if R["res"].empty:
        raise SystemExit("results_all.csv 不存在；请先运行 scripts/03_final_eval.py 等实验脚本。")
    doc = Document()
    setup_document(doc)
    add_page_number_footer(doc)

    build_header(doc)
    build_section1(doc, R)
    build_section2(doc, R)
    build_section3(doc, R)
    build_section4(doc, R)
    build_section5(doc, R)
    build_section6(doc, R)
    build_section7(doc, R)
    build_appendix(doc, R)

    out = ROOT / "实验2报告_钓鱼网站检测_参考版.docx"
    doc.save(out)
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()

