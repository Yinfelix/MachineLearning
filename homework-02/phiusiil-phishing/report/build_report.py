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
    "T": "T 去除全部单向捷径",
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

    add_caption(doc, "表 1　小组分工与贡献（姓名、学号按要求留空）")
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
    add_paragraph(doc, (
        "在正式建模前，我们先做了一次“数据体检”。结果发现这个数据集比方案文档预想的更容易："
        "URLSimilarityIndex 在全部合法样本上恒等于 100，IsHTTPS 在全部合法样本上恒等于 1；"
        "仅用这两条规则，测试集上的 F1 就达到 0.9999，误报为 0。"
        "因此，本项目不把“把分数刷到多高”作为目标，而是把重点放在三件事上："
        "第一，量化捷径特征对结果的贡献；第二，在去掉捷径后公平比较不同分类算法；"
        "第三，检验模型在样本不足、噪声、异常值和跨顶级域名场景下是否仍然可靠。"
    ))

    add_h2(doc, "1.2　课题目标")
    add_paragraph(doc, (
        "项目按“数据准备—模型构建—训练评估—优化分析—拓展实验”展开。"
        "我们先完成缺失值、编码、标准化和分组划分，再实现逻辑回归、高斯朴素贝叶斯、决策树、随机森林和 RBF 支持向量机；"
        "所有模型使用同一套划分、预处理和评价指标，通过网格搜索选参，并用混淆矩阵、ROC、学习曲线和特征重要性解释行为。"
        "最后回答三个问题："
    ))
    add_paragraph(doc, "1. 数据集中哪些特征构成了“捷径”？去掉它们以后，各模型还剩多少真实判别能力？", first_indent=False)
    add_paragraph(doc, "2. 五类算法在同一套严格划分下的表现有何差异，差异能否用算法原理和数据特点解释？", first_indent=False)
    add_paragraph(doc, "3. 当训练样本减少、标签被污染、特征被加噪或遇到罕见 TLD 时，模型是否仍然可靠？", first_indent=False)

    add_h2(doc, "1.3　课题数据集")
    add_paragraph(doc, (
        f"PhiUSIIL 数据集共 {a['n_rows']:,} 行、56 列，其中标签 1 列、文件名 1 列，其余 54 列为特征；"
        f"URL、Domain、Title 是文本列，FILENAME 是样本编号，均不直接入模，因此模型实际使用 {a['n_features_excluding_text']} 个特征。"
        f"数据没有缺失值，也没有完全重复的行；网址基本唯一（{a['unique_url']:,} 个不同 URL），"
        f"但 Domain 有 {a['unique_domains']:,} 个，其中 {a['rows_in_repeated_domains']:,} 行落在重复域名中，最大单域名包含 {a['max_domain_group']:,} 条样本。"
        "这说明如果随机划分，同一域名的样本可能同时出现在训练和测试中，模型会记住域名而不是学习泛化规律。"
    ))
    add_paragraph(doc, (
        f"类别方面，合法样本 {a['label_counts']['1']:,} 条（{pct(a['label_counts']['1']/a['n_rows'])}），"
        f"钓鱼样本 {a['label_counts']['0']:,} 条（{pct(a['label_counts']['0']/a['n_rows'])}），属于轻度不平衡。"
        f"顶级域名共 {a['unique_tld']} 个，最常见的 com、org、net 分别占 {pct(a['top_tld']['com']/a['n_rows'])}、"
        f"{pct(a['top_tld']['org']/a['n_rows'])}、{pct(a['top_tld']['net']/a['n_rows'])}。"
        "这些统计提示我们：划分必须按 Domain 分组，评价时还要单独看误报率，因为把合法网站误封的代价很高。"
    ))
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

    add_paragraph(doc, (
        f"图 1 展示了训练集概况。合法 {a['train_test']['train']['legitimate']:,} 条、钓鱼 {a['train_test']['train']['phishing']:,} 条，"
        f"钓鱼占 {pct(a['train_test']['train']['phishing_rate'])}。"
        f"大多数域名只出现一次，但少数域名包含大量样本，最长的一条有 {a['max_domain_group']:,} 条；"
        f"有 {a['domain_label_nunique_counts'].get('2', 0)} 个域名同时包含两类样本，随机划分会造成同域名泄漏，因此我们按 Domain 分组划分。"
        f"com 占绝对多数，TLD 长尾很长，这是后面做频次编码与跨 TLD 迁移实验的原因。"
    ))
    add_h2(doc, "2.2　捷径特征的发现：这个数据集有多容易？")
    add_paragraph(doc, (
        "图 1(b) 给出了一个不寻常的现象：所有合法样本的 URLSimilarityIndex 都等于 100，而钓鱼样本的取值从 0 到 100 连续分布。"
        f"在训练集上，该特征的单变量 ROC-AUC 为 {f4(float(uni.loc[uni.feature=='URLSimilarityIndex','auc_oriented'].iloc[0]))}；"
        f"钓鱼样本中只有 {pct(a['url_similarity_by_label']['0']['pct_at_100'])} 恰好等于 100。"
        "这意味着仅用“URLSimilarityIndex 是否小于 100”一条规则，就能以接近 100% 的召回率抓到钓鱼样本，而且不会误伤任何合法样本。"
    ))
    add_paragraph(doc, (
        f"第二个单向捷径是 IsHTTPS：全部合法样本都使用 HTTPS，而钓鱼样本中只有 {pct(a['is_https_by_label']['0'])} 使用 HTTPS。"
        "按方向校正后的单变量 AUC 显示：除了 URLSimilarityIndex，"
        "LineOfCode、NoOfExternalRef、NoOfImage、NoOfSelfRef、NoOfJS、NoOfCSS 等页面规模特征的 AUC 也在 0.96–0.99 之间。"
        "它们不是官方标注的“衍生特征”，但同样把两类样本推得很开。"
    ))
    s = R["shortcut"]
    add_paragraph(doc, (
        f"为了把“容易”量化，我们在测试集上评估了一个只有两条规则的基线：URLSimilarityIndex < 100 判为钓鱼；"
        f"否则若 IsHTTPS = 0 判为钓鱼；其余判为合法。它不需要训练，却在 {s['test_metrics']['tn']+s['test_metrics']['fp']+s['test_metrics']['fn']+s['test_metrics']['tp']:,} 条测试样本上取得 "
        f"F1 = {f4(s['test_metrics']['f1'])}、召回率 = {f4(s['test_metrics']['recall'])}、误报率 = {f4(s['test_metrics']['fpr'])}，只漏掉 {s['test_metrics']['fn']} 条钓鱼样本。"
        "这个结果说明：全特征方案 A 上的高准确率主要来自标签泄漏，而不是模型学到了可迁移的判别规律。"
        "后面的模型比较必须配合消融实验才有意义。"
    ))
    add_caption(doc, "表 3　两个单向捷径特征的统计")
    rows = [
        ["URLSimilarityIndex", "合法恒为 100；钓鱼均值 49.62，0.78% 等于 100", "0.9961", "小于 100 判钓鱼"],
        ["IsHTTPS", "合法 100% 为 1；钓鱼 49.2% 为 1", "0.7539", "等于 0 判钓鱼"],
        ["两条规则合并", "测试集仅漏报 5 条，误报 0 条", "F1 0.99988", "—"],
    ]
    add_table(doc, ["特征 / 规则", "取值特点", "方向校正 AUC", "使用方式"], rows,
              widths_in=[1.3, 2.4, 0.95, 1.1], font_size=8.8)

    add_h2(doc, "2.3　预处理")
    add_paragraph(doc, (
        "预处理全部写在 scikit-learn 的 Pipeline 与 ColumnTransformer 中，保证每个训练折内独立拟合。"
        "具体做法如下：删除 FILENAME、URL、Domain、Title 四个文本列，Domain 只用于分组划分；"
        "TLD 采用训练折内的频次编码，把类别映射为该 TLD 在训练折中的出现次数，未见过的 TLD 映射为 0，"
        "这样既保留了“常见/罕见”的信息，又避免高基数独热编码；"
        "对 URL 长度、代码行数、图片/脚本数量等 11 个长尾计数特征做 log1p 变换（该变换不依赖数据，不产生泄漏）；"
        "数值特征用训练折中位数填补缺失值（本数据实际无缺失，保留填补器是为了流程完整）；"
        "逻辑回归与 SVM 额外做标准化，树模型与朴素贝叶斯不缩放。"
    ))
    add_paragraph(doc, (
        "我们没有把 URL、Domain、Title 直接向量化后入模。原因是它们基数极高，词袋或哈希编码会带来巨大的稀疏矩阵，"
        "而且模型很容易记住具体域名。把 Domain 仅用于分组划分，可以更干净地检验模型对新域名的泛化能力。"
    ))

    add_h2(doc, "2.4　数据划分与防泄漏")
    add_paragraph(doc, (
        "我们用 StratifiedGroupKFold(5) 先按 Domain 分组切出 20% 作为独立测试集，其余 80% 作为训练集；"
        "训练集内部再用 StratifiedGroupKFold(5) 做交叉验证。这样同一域名不会同时出现在训练与测试（或训练折与验证折）中，"
        "同时尽量保持两类的比例。最终训练集 188,801 条、测试集 46,994 条，训练/测试的 Domain 重叠为 0，"
        f"钓鱼比例分别为 {pct(tr_rate(R))} 和 {pct(te_rate(R))}。"
        "所有缩放、频次编码、特征选择与 PCA 都只在训练折内拟合；测试集只在最终评估时使用一次，"
        "不参与调参、特征选择或阈值选择。完整检查清单见附录。"
    ))
    add_paragraph(doc, (
        "RBF 支持向量机在二十多万样本上训练的时间和内存开销都明显高于其他模型，因此按方案约定，"
        "在训练集内固定抽取 20,000 条分层子样本用于训练与调参；同时用 LinearSVC 在全量训练集上训练作为对照。"
        "这样既保留了核方法与线性方法的可比性，也避免让 SVM 成为整个实验的瓶颈。"
    ))

    add_h2(doc, "2.5　八套特征方案")
    add_paragraph(doc, (
        "仅做 A/B 两套方案不足以解释数据中的多组捷径，因此我们在方案文档的 A–D 基础上，根据 EDA 结果补充了 S、T、U、V 四套严格方案。"
        "表 4 列出了全部方案。A 是基线；B 去掉论文标注的 6 个衍生特征；C 只保留 URL/域名特征；D 只保留页面内容特征；"
        "S 在 B 上再去掉 6 个页面规模特征；T 在 S 上再去掉 IsHTTPS；U 在 C 上去掉 4 个 URL 衍生特征；V 在 U 上再去掉 IsHTTPS。"
        "S、T、U、V 只用于补充消融，不改变主实验的 A–D 结论。"
        "需要强调，IsHTTPS 与页面规模特征本身都是合理的部署信号；把它们逐步移除，是为了诊断数据集偏差，"
        "而不是说它们在真实系统中没有价值。"
    ))
    add_caption(doc, "表 4　特征方案与特征数")
    rows = [
        ["A 全特征", "51", "全部数值特征 + TLD", "基线，预计接近 100%"],
        ["B 去官方捷径", "45", "A 去掉 6 个官方衍生特征", "检验官方衍生特征的贡献"],
        ["C 仅 URL", "23", "22 个 URL 数值特征 + TLD", "只看网址能做到多好"],
        ["D 仅页面", "28", "28 个页面内容特征", "只看页面能做到多好"],
        ["S 严格去捷径", "39", "B 去掉 6 个页面规模特征", "去掉第二组捷径"],
        ["T 去除全部单向捷径", "38", "S 再去掉 IsHTTPS", "最严格的全特征消融"],
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
    add_paragraph(doc, (
        "我们选择五类建模思路不同的传统分类方法：线性判别（逻辑回归）、生成式概率模型（高斯朴素贝叶斯）、"
        "单棵规则树（决策树）、Bagging 集成（随机森林）和最大间隔核方法（RBF 支持向量机）。"
        "另用 LinearSVC 在全量数据上作为 SVM 的线性对照。所有模型都使用 scikit-learn 的现成实现 [5]，"
        "并放进统一的 Pipeline：预处理 → 模型，保证比较公平。"
    ))

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
    add_paragraph(doc, (
        "其中精确率 P 表示“报警的网站里有多少真的是钓鱼”，召回率 R 表示“真正的钓鱼网站有多少被抓到”，"
        "F1 是两者的调和平均；FPR 是合法网站被误判为钓鱼的比例，对应误封代价。"
        "此外还报告准确率、ROC-AUC、PR-AUC 与平衡准确率。"
        "两类比例约为 57:43，属于轻度不平衡；我们没有做重采样，而是用 F1、PR-AUC 与 FPR 避免只看准确率。"
        "验证阶段使用训练集内部 5 折分组交叉验证的均值 ± 标准差；最终阶段在独立测试集上只评估一次。"
        "所有模型使用同一套划分、同一套预处理流程和同一套指标，超参数只在训练集内选择。"
    ))

    add_h2(doc, "4.2　测试集结果：全特征方案没有区分度")
    a = res[res.scheme == "A"]
    a4 = a[a.model != "Gaussian NB"]
    nb_a = a[a.model == "Gaussian NB"].iloc[0]
    add_paragraph(doc, (
        "表 6 给出了五个主模型在 A–D、S、T、V 方案上的测试集 F1。先看方案 A："
        f"除高斯朴素贝叶斯外，其余四个模型的 F1 都在 {f4(a4.test_f1.min())}–{f4(a4.test_f1.max())} 之间，彼此差距小于千分之一；"
        f"高斯朴素贝叶斯为 {f4(nb_a['test_f1'])}，明显低一截。"
        "前四者的高分并不是因为算法同样优秀，而是因为 URLSimilarityIndex 已经把两类样本几乎完全分开，"
        "任何能够利用该特征的模型都会得到接近完美的结果。方案 C 也包含 URLSimilarityIndex，因此同样接近天花板。"
        "朴素贝叶斯的落后则说明：即使存在捷径，条件独立假设被破坏仍会拖累概率估计。"
        "在这种条件下比较模型，等于在比较“谁先找到同一条捷径”，参考价值有限。"
    ))
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
    add_paragraph(doc, (
        "表 6 中 B、S、T、V 列展示了逐步去掉捷径后的变化。B 去掉了 URLSimilarityIndex 等 6 个官方衍生特征，"
        f"随机森林与逻辑回归的 F1 仍为 {f4(row(res,'Random Forest','B')['test_f1'])} 与 {f4(row(res,'Logistic Regression','B')['test_f1'])}，"
        f"决策树降到 {f4(row(res,'Decision Tree','B')['test_f1'])}，朴素贝叶斯降到 {f4(row(res,'Gaussian NB','B')['test_f1'])}。"
        "原因有两层：第一，B 仍然保留 IsHTTPS，而它在合法类上恒为 1；第二，LineOfCode、NoOfExternalRef、NoOfImage 等页面规模特征本身就有 0.96–0.99 的单变量 AUC。"
        "S 去掉这 6 个页面规模特征后，随机森林与逻辑回归仍在 0.999 以上，决策树降到 0.957，朴素贝叶斯降到 0.493；"
        "T 再去掉 IsHTTPS，线性模型的 F1 仍为 0.998 以上，决策树与朴素贝叶斯基本不变。"
        "这说明官方衍生特征只是第一层捷径，页面规模与页面标志还提供了大量可分类信息。"
    ))
    add_figure(doc, FIG / "fig04_ablation_lines.png", width_in=5.55,
               caption="图 2　两条消融路径上的测试集 F1：(a) A → B → S → T；(b) C → U → V")
    add_paragraph(doc, (
        "真正拉开差距的是 V：它只保留 URL 长度、字符比例、混淆标志等基础词法特征，"
        "去掉了 URLSimilarityIndex、CharContinuationRate、TLDLegitimateProb、URLCharProb 与 IsHTTPS。"
        f"在 V 上，逻辑回归的 F1 为 {f4(row(res,'Logistic Regression','V')['test_f1'])}，随机森林为 {f4(row(res,'Random Forest','V')['test_f1'])}，"
        f"RBF SVM 为 {f4(row(res,'SVM (RBF)','V')['test_f1'])}，但决策树降到 {f4(row(res,'Decision Tree','V')['test_f1'])}，"
        f"朴素贝叶斯只有 {f4(row(res,'Gaussian NB','V')['test_f1'])}。"
        "同时也要承认，即使去掉全部已知单向捷径，逻辑回归仍有 0.996 的 F1——这个数据集本身就不难；"
        "本项目的价值不在于找到一个“很难”的基准，而在于把“为什么容易”拆开，并观察不同算法在特征变化下的相对优劣。"
    ))

    add_h2(doc, "4.4　去掉捷径后，哪种模型更好？")
    def best_on(scheme):
        return res[res.scheme == scheme].sort_values("test_f1", ascending=False).iloc[0]

    bB, bD, bT, bV = best_on("B"), best_on("D"), best_on("T"), best_on("V")
    add_paragraph(doc, (
        "把四个关键方案放在一起看，结论不是“某个模型全面最好”，而是“算法优势随特征类型变化”："
        f"在 B 上，{MODEL_CN[bB['model']]} 与逻辑回归几乎并列（{f4(bB['test_f1'])} 对 {f4(row(res,'Logistic Regression','B')['test_f1'])}）；"
        f"在页面特征主导的 D 上，{MODEL_CN[bD['model']]} 最好（{f4(bD['test_f1'])}）；"
        f"但在 URL 词法主导的 T、V 上，逻辑回归反超（{f4(bT['test_f1'])}、{f4(bV['test_f1'])}），随机森林分别为 "
        f"{f4(row(res,'Random Forest','T')['test_f1'])}、{f4(row(res,'Random Forest','V')['test_f1'])}。"
        "这个反转是本项目最有意思的发现之一。"
    ))
    add_paragraph(doc, (
        "随机森林的优势在 D 上最明显：页面规模、表单、图片、外链等特征带有阈值和交互，树模型可以自动切分，"
        "Bagging 又降低了单棵树的方差。但在 T/V 中，页面结构特征被去掉后，剩下的是 URL 长度、特殊字符比例、"
        "字母/数字比例等连续词法特征，类别边界更接近线性；此时逻辑回归的平滑线性组合比树的阶跃边界更稳，"
        "系数也更容易解释。代价上的差异同样明显：在 V 上逻辑回归的 FPR 为 "
        f"{pct(row(res,'Logistic Regression','V')['test_fpr'],3)}，随机森林为 {pct(row(res,'Random Forest','V')['test_fpr'],3)}，"
        "前者把合法网站误判为钓鱼的比例低了一个数量级。"
    ))
    add_paragraph(doc, (
        "决策树与朴素贝叶斯的失败方式不同。决策树在 A 上被调成 max_depth=3 就达到满分，"
        "但这个深度是为“一条 URLSimilarityIndex 阈值”服务的；换到 V 后，深度 3 远不足以表达 URL 词法中的细节，"
        f"召回率只有 {pct(row(res,'Decision Tree','V')['test_recall'])}，漏掉约三分之一的钓鱼样本。"
        "朴素贝叶斯在 S/T/V 上的 F1 只有 0.49–0.59，但 AUC 仍有 0.81–0.86：它并不是完全不会排序，"
        "而是概率被系统性压低，固定 0.5 阈值下把大量钓鱼样本判成合法（V 的召回率仅 "
        f"{pct(row(res,'Gaussian NB','V')['test_recall'])}）。特征高度相关破坏了条件独立假设，使方差估计失真；"
        "如果做概率校准或重新选阈值，它的 F1 会改善，但仍不适合作为主模型。"
    ))
    ph = R["posthoc"]
    add_paragraph(doc, (
        "为了确认这是超参数问题而不是算法本身的问题，我们在 V 上做了两个诊断实验（阈值与深度都只在训练集内选择，测试集只评估一次）。"
        f"对决策树重新选择深度：验证集选出的最佳深度是“不限制”，测试 F1 从 0.7835 提升到 {f4(ph['dt_V']['test_f1_at_best_depth'])}，"
        "说明 depth=3 只是 A 方案捷径下的产物。"
        f"对朴素贝叶斯下调阈值：从 0.5 降到 {ph['nb_V']['best_threshold_oof']:.1e} 后，测试 F1 从 "
        f"{f4(ph['nb_V']['default_0.5']['f1'])} 提升到 {f4(ph['nb_V']['chosen_threshold']['f1'])}，"
        "但仍远低于其 AUC 所暗示的潜力；概率压缩与校准才是它失效的主因。"
    ))
    add_paragraph(doc, (
        "RBF SVM 在 V 上与随机森林接近（0.994），但没有超过线性模型；全量 LinearSVC 在 T/V 上与逻辑回归几乎相同"
        "（0.9986、0.9961），说明在这个特征空间里核方法带来的非线性收益有限，主要限制来自特征与样本，而不是模型容量。"
        "这些结果共同说明：选模型之前，先看特征的类型与来源，比盲目比较模型更重要。"
    ))


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
    add_paragraph(doc, (
        "表 6 给出了 V 方案下五个模型的 F1，AUC 见 outputs/metrics/results_all.csv。ROC 曲线越靠近左上角越好；"
        "由于正类比例约为 43%，PR 曲线（图略，AP 值见 outputs/metrics/results_all.csv）比 ROC 更能反映钓鱼类的检索质量。"
        f"AUC 最高的是 {MODEL_CN[auc_rank.iloc[0]['model']]}（{f4(auc_rank.iloc[0]['test_roc_auc'])}）"
        f"与 {MODEL_CN[auc_rank.iloc[1]['model']]}（{f4(auc_rank.iloc[1]['test_roc_auc'])}），"
        f"决策树与朴素贝叶斯的 AUC 分别降到 {f4(row(res,'Decision Tree','V')['test_roc_auc'])} 与 "
        f"{f4(row(res,'Gaussian NB','V')['test_roc_auc'])}。"
        "值得注意的是，朴素贝叶斯的 AUC（约 0.81）明显高于它的 F1（0.59），说明它的排序并非完全无效，"
        "问题主要出在概率尺度与 0.5 阈值的错配；ROC 与 F1 的分歧本身就是校准问题的证据。"
    ))

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
    add_paragraph(doc, (
        f"默认随机森林在方案 B 上已经取得测试 F1 = {f4(d0['test_f1'])}。"
        f"在 B 上重新调参后，测试 F1 变为 {f4(d2['test_f1'])}，变化为 {delta:+.5f}（基本不变）；交叉验证 F1 的变化也很小。"
        "这说明在当前特征与样本规模下，随机森林对超参数并不敏感：默认的 100 棵树已经足够，增加树的数量主要降低方差，"
        "而深度和叶节点约束在 B 方案上也没有带来可见的提升。"
        "这与“数据本身接近可分”的判断一致——当特征已经把两类样本分开时，模型容量的边际收益很低。"
    ))

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
    add_paragraph(doc, (
        f"在本次搜索中，验证 F1 最高的最大深度为 {int(best_depth['max_depth']) if best_depth['max_depth'] < 99 else '不限制'}，"
        f"对应验证 F1 = {f4(best_depth['val_f1'])}；不限制深度时验证 F1 为 {f4(dd.iloc[-1]['val_f1'])}。"
        "这组实验说明：单棵决策树必须控制复杂度，否则会把训练集中的噪声也学进去；"
        "而随机森林通过 Bagging 平均多棵树，对这种过拟合更不敏感，因此在 5.3 节中调参收益也更小。"
        "更值得注意的是，在 A 上决策树的最佳深度只有 3（表 5），而在 B 上验证 F1 最高的深度变为 "
        f"{best_depth_label}；深度需求会随特征集变化。把 A 上调出的 depth=3 直接用于 V，"
        "就会得到 4.4 节中 67.6% 的召回率——这正是“在捷径上调参”的代价。"
    ))

    add_h2(doc, "5.5　阈值优化：召回率与误报的权衡")
    th = R["threshold"]
    add_paragraph(doc, (
        "前面的指标都默认使用 0.5 作为分类阈值。实际钓鱼检测中，漏报的代价通常高于误报，因此我们希望在召回率不低于 99% 的前提下，"
        "尽量降低误报率。具体做法是：在训练集上用 5 折交叉验证得到随机森林（方案 V）的袋外预测概率，"
        "在概率网格上选择满足“召回率 ≥ 0.99”的最大阈值，然后把这个阈值固定用于测试集。"
        "阈值只在训练集内部确定，没有使用测试集。"
    ))
    add_caption(doc, "表 8　随机森林（方案 V）在默认阈值与优化阈值下的测试集表现（阈值在 OOF 上选择）")
    d05, dch = th["default_0.5"], th["chosen"]
    rows = [
        ["0.500（默认）", f4(d05["recall"]), f4(d05["precision"]), f4(d05["f1"]), pct(d05["fpr"], 3), f"{d05['fp']} / {d05['fn']}"],
        [f"{th['chosen_threshold']:.3f}（优化）", f4(dch["recall"]), f4(dch["precision"]), f4(dch["f1"]), pct(dch["fpr"], 3), f"{dch['fp']} / {dch['fn']}"],
    ]
    add_table(doc, ["阈值", "召回率", "精确率", "F1", "FPR", "FP / FN"], rows,
              widths_in=[1.25, 0.8, 0.8, 0.75, 0.75, 0.8], font_size=8.6)
    add_paragraph(doc, (
        f"在默认阈值 0.5 下，模型在测试集上的召回率为 {f4(d05['recall'])}、FPR 为 {pct(d05['fpr'], 3)}；"
        f"把阈值调整到 {th['chosen_threshold']:.3f} 后，召回率变为 {f4(dch['recall'])}、FPR 变为 {pct(dch['fpr'], 3)}。"
        "如果召回率已经接近上限，阈值移动主要是在“多抓几条钓鱼”和“多误封几个合法网站”之间做交换。"
        "需要说明的是，该阈值在 OOF 上满足“召回率 ≥ 99%”，迁移到测试集后召回率为 98.87%，略低于目标，"
        "这说明阈值选择本身也有泛化误差。"
        "值得注意的是，逻辑回归在 V 上默认阈值下的 FPR 只有 0.015%，已经优于随机森林调阈值后的 0.082%；"
        "这说明对误报敏感的场景，换一个更适合 URL 词法特征的模型，比在同一模型上调阈值更有效。"
        "本实验说明阈值是一个可调的操作点，而不是模型固有的属性；报告任何单一阈值下的指标时，都应该同时说明阈值和代价假设。"
    ))


def build_section6(doc, R):
    add_h1(doc, "六、拓展任务")

    add_h2(doc, "6.1　泛化与鲁棒性")
    add_paragraph(doc, (
        "这一节回答“模型在不理想条件下还可靠吗”。所有实验都在训练集内部完成：第 1 个内折作固定验证集，其余四个折作抽样池，测试集不参与。"
        "由于 B 已接近天花板，鲁棒性实验改在更严格的 V 上进行，模型使用第 3 章选出的超参数；"
        "决策树例外：A 上调出的 depth=3 在 V 上严重欠拟合，因此改用 V 上重新选择的深度（不限制），以免把欠拟合误读为鲁棒性差。"
    ))

    add_h3(doc, "（1）训练样本量：谁最“省样本”？")
    lc = R["lc"]
    add_paragraph(doc, (
        "我们从抽样池中按 1%、5%、10%、25%、50%、100% 分层抽取训练样本，每个比例重复 3 个随机种子，在固定验证集上评估。"
        "图 5 是逻辑回归、决策树与随机森林的学习曲线。"
    ))
    add_figure(doc, FIG / "fig11_learning_curve.png", width_in=5.0,
               caption="图 5　训练样本比例对验证 F1 的影响（方案 V，3 个种子）")
    lc_sum = lc.groupby(["model", "fraction"])["f1"].mean().reset_index()
    p1 = lc_sum[lc_sum.fraction == 0.01].set_index("model")["f1"]
    p100 = lc_sum[lc_sum.fraction == 1.0].set_index("model")["f1"]
    best1, best100 = p1.idxmax(), p100.idxmax()
    add_paragraph(doc, (
        f"在只使用 1% 训练样本时，逻辑回归、决策树、随机森林的验证 F1 分别为 "
        f"{f4(p1.get('Logistic Regression', np.nan))}、{f4(p1.get('Decision Tree', np.nan))}、{f4(p1.get('Random Forest', np.nan))}；"
        f"使用 100% 时分别为 {f4(p100.get('Logistic Regression', np.nan))}、{f4(p100.get('Decision Tree', np.nan))}、"
        f"{f4(p100.get('Random Forest', np.nan))}。"
        f"1% 样本下最强的是 {MODEL_CN[best1]}，100% 时最强的是 {MODEL_CN[best100]}。"
        "逻辑回归在 1% 时已经领先（约 0.984，而决策树/随机森林约 0.93/0.95），5% 后基本饱和；"
        "树模型需要 25% 以上样本才接近 0.99。原因是 URL 词法特征与 log-odds 近似线性，线性模型只需估计少量系数；"
        "树模型要估计大量分裂阈值，样本效率更低。决策树在样本很少时方差也更大，曲线抖动更明显。"
    ))

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
    add_paragraph(doc, (
        f"20% 标签噪声下，F1 下降最多的是 {MODEL_CN[drops.index[0]]}（{drops.iloc[0]:.4f}），"
        f"下降最少的是 {MODEL_CN[drops.index[-1]]}（{drops.iloc[-1]:.4f}）。"
        "具体来看，随机森林从 0.995 降到 0.963，决策树从 0.994 降到 0.916，逻辑回归从 0.996 降到 0.803。"
        "这与“树模型更容易记住噪声”的直觉相反：逻辑回归在 A 上调出的 C=100 正则化很弱，翻转标签会把线性边界明显拉偏；"
        "随机森林的 Bagging 平均与树的局部性反而更稳健。"
        "特征噪声的影响则更为剧烈：仅 0.1 倍标准差就让三个模型的 F1 降到 0.62–0.69，"
        "说明模型依赖精确的字符比例与长度；其中随机森林的 AUC 仍有 0.90，排序能力尚在，"
        "F1 崩塌主要来自概率/阈值失配；逻辑回归与决策树的 AUC 也降到 0.69 左右，连排序都被破坏。"
        "这提醒我们：高准确率并不等于对特征测量误差稳健，部署前必须做特征稳定性检查与概率校准。"
    ))

    add_h3(doc, "（3）异常值与缩放方式")
    out = R["outliers"]
    add_paragraph(doc, (
        "我们在验证集中随机选取 5% 的样本，把 URLLength、DomainLength、NoOfOtherSpecialCharsInURL 三个特征替换为“均值 + 15 倍标准差”的极端值，"
        "比较 StandardScaler 与 RobustScaler 两种缩放方式下逻辑回归的表现。"
        "在干净验证集上，StandardScaler 与 RobustScaler 的 F1 分别为 0.9964 与 0.9971；"
        "注入 5% 极端值后，两者分别降到 0.9702 与 0.9710，几乎相同。"
        "RobustScaler 使用中位数与四分位距，理论上更稳健，但本实验中优势有限，"
        "因为逻辑回归的决策边界仍然会被极端特征值推远。更根本的解决办法是截断、对数变换或异常值检测，而不是只换缩放器。"
    ))

    add_h3(doc, "（4）跨 TLD 迁移")
    tld = R["tld"]
    add_paragraph(doc, (
        "我们用训练集中最常见的 20 个 TLD 训练模型，分别在同分布的随机留出集和“罕见 TLD”测试集上评估。"
        "罕见 TLD 测试集包含 414 个 TLD、8,774 条样本，钓鱼比例为 55.5%，明显高于总体，"
        "因此这不仅是一个分布迁移测试，也带有类别比例变化。"
    ))
    base = tld[tld.split == "随机划分"].set_index("model")
    rare = tld[tld.split == "罕见 TLD"].set_index("model")
    gaps = (base["f1"] - rare["f1"]).sort_values(ascending=False)
    add_paragraph(doc, (
        f"两个模型在罕见 TLD 上都有小幅但一致的下降：{MODEL_CN[gaps.index[0]]} 的 F1 下降 {gaps.iloc[0]:.4f}，"
        f"{MODEL_CN[gaps.index[-1]]} 下降 {gaps.iloc[-1]:.4f}。"
        "随机森林的下降更大，但两者仍保持在 0.986 以上。"
        "这说明模型学到了与具体 TLD 分布相关的模式，而不是完全通用的钓鱼规律；"
        "常见 TLD（尤其是 com）在训练中占绝对多数，模型会倾向于用“TLD 是否常见”作为信号；"
        "一旦遇到训练中罕见的 TLD，这个信号就变弱。实际部署时，应按 TLD 分层监控性能，或对罕见 TLD 使用单独的阈值。"
    ))

    add_h3(doc, "（5）过拟合分析")
    of = R["overfit"]
    worst_gap = of.sort_values("gap", ascending=False).iloc[0]
    add_paragraph(doc, (
        f"差距最大的是 {MODEL_CN[worst_gap['model']]}（训练 {f4(worst_gap['train_f1'])}，验证 {f4(worst_gap['val_f1'])}，"
        f"差 {worst_gap['gap']:.4f}），这与决策树可以无限划分、记住个别样本的特性一致。"
        "随机森林的差距更小（约 0.0034），因为 Bagging 平均降低了方差；逻辑回归、朴素贝叶斯与 SVM 的差距都在 0.001 以内。"
        "需要说明的是，这些差距都在 1 个百分点以内，在当前特征集上过拟合不是主要矛盾；"
        "相比之下，特征噪声、标签噪声与概率校准才是更值得关注的问题。"
    ))

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
    add_paragraph(doc, (
        "单变量 AUC 与互信息给出的排序高度一致：URLSimilarityIndex、LineOfCode、NoOfExternalRef 等捷径特征遥遥领先，"
        "URL 词法特征中 NoOfOtherSpecialCharsInURL、SpacialCharRatioInURL、URLLength 最强。"
        "这为后面的特征选择提供了依据：如果只保留少数特征，应该优先保留这些信息量大的变量。"
    ))

    add_h3(doc, "（2）过滤法、嵌入法与降维")
    kdf, pca_df = R["kbest"], R["pca"]
    add_paragraph(doc, (
        "我们用 SelectKBest（f_classif）在 Pipeline 内比较 K = 5、10、20 与全部特征；"
        "用 PCA 比较保留 90%、95%、99% 方差；用 L1 逻辑回归的正则化路径观察非零系数个数；"
        "再用随机森林的 impurity importance 做嵌入法排序。"
    ))
    bestk = kdf.sort_values("cv_f1_mean", ascending=False).iloc[0]
    add_paragraph(doc, (
        f"过滤法的最佳组合是 {MODEL_CN[bestk['model']]} + K={bestk['k']}（F1={f4(bestk['cv_f1_mean'])}）。"
        "K 从 5 增加到 20 时，F1 明显上升；从 20 增加到全部 45 个特征时，提升已经很小，甚至略有下降。"
        "这说明前 20 个特征已经承载了大部分信息，剩余特征更多是冗余。"
        "PCA 在保留 90%–99% 方差时基本保持了原性能（逻辑回归 0.9995–0.9998，SVM 0.9983–0.9988），"
        "保留 95% 方差约需 34 个主成分；压缩到 90% 时才开始出现轻微损失。这说明特征之间存在较强的线性相关，降维可行。"
    ))
    l1 = R["l1"]
    add_paragraph(doc, (
        "L1 正则化路径显示，当 C 从 0.001 增大到 0.1 时，非零系数从很少迅速增加；C 超过 1 后，非零系数个数趋于稳定。"
        "这意味着在方案 B 中，真正必要的特征大约只占全部特征的一部分，其余特征可以被压缩掉而不明显损失性能。"
    ))

    add_h3(doc, "（3）特征升维：线性模型能否追上非线性模型？")
    poly = R["poly"]
    add_paragraph(doc, (
        "我们取随机森林重要性最高的 10 个特征，比较两种逻辑回归：一种只使用这 10 个原始特征，另一种做二阶多项式展开（包含平方项与交互项）。"
        
    ))
    lin = poly[poly["mode"] == "linear"].iloc[0]
    quad = poly[poly["mode"] == "quadratic"].iloc[0]
    add_paragraph(doc, (
        f"二阶展开后，验证 F1 从 {f4(lin['cv_f1_mean'])} 变为 {f4(quad['cv_f1_mean'])}，变化 {quad['cv_f1_mean']-lin['cv_f1_mean']:+.4f}。"
        "升维没有带来明显提升，说明在这 10 个特征上，类别边界已经接近线性；"
        "但代价是特征数从 10 增加到 65，训练与解释成本都更高。"
        "因此，单纯增加多项式特征不是提高本任务性能的有效途径。"
    ))

    add_h3(doc, "（4）置换重要性与 SHAP")
    perm = R["perm"]
    add_paragraph(doc, (
        "我们用随机森林（方案 B）在验证集上计算置换重要性：把某一列随机打乱后，F1 下降越多，说明模型越依赖它。"
        "与基于不纯度的 impurity importance 相比，置换重要性是在“已训练模型 + 验证数据”上计算的，"
        "不会偏向取值多的连续特征，因此更可信。"
    ))
    top_perm = perm.head(5)["feature"].tolist()
    add_paragraph(doc, (
        f"置换重要性最高的前五个特征依次是：{'、'.join(top_perm)}。"
        "其中 LineOfCode 与 IsHTTPS 的贡献远高于其余特征，NoOfSubDomain 次之，URL 词法特征的直接贡献很小。"
        "这与 4.3 节的结论一致：B 只是去掉了官方衍生特征，模型仍然依赖页面规模与 HTTPS 标志这两层捷径。"
        "与基于不纯度的 impurity importance 相比，置换重要性把权重集中在少数真正被模型使用的特征上；"
        "impurity importance 还把权重分散到 NoOfImage、NoOfJS、NoOfSelfRef 等多个相关特征上，"
        "这也是相关特征会“分摊”重要性的一种表现。"
    ))
    add_paragraph(doc, (
        "图 7 的 SHAP 值 [6] 给出了方向信息：横轴是 SHAP 值（对模型输出的贡献），颜色表示特征取值高低。"
        "例如，LineOfCode 越低（页面越简单）、IsHTTPS = 0、子域名越多，SHAP 值越正，模型越倾向于判为钓鱼；"
        "这与钓鱼站点“页面简陋、缺少 HTTPS、多用免费子域名”的特点一致。"
        "SHAP 的排序与置换重要性大体一致，但 SHAP 能进一步区分“高值推向钓鱼”与“低值推向钓鱼”，更适合向非技术读者解释。"
    ))
    add_figure(doc, FIG / "fig17_shap.png", width_in=4.9,
               caption="图 7　随机森林（方案 B）的 SHAP 摘要图（验证集 2,000 条）")

    add_h3(doc, "（5）浅层决策树规则")
    add_paragraph(doc, (
        "为了得到一套人类可读的规则，我们在方案 B 上训练了一棵最大深度为 3 的决策树。"
        f"方案 B 已经去掉 URLSimilarityIndex，因此树的根节点首先按 {R['feature_summary'].get('tree_root_feature','NoOfExternalRef')} 等页面规模特征分裂，"
        "后续节点再用 URL 特殊字符比例、域名长度等词法特征补充。"
        "这棵浅树的性能不如随机森林，但它把“模型在看什么”讲得很清楚："
        "先看页面规模，再用 URL 词法特征做细分。如果进一步去掉页面规模特征（方案 S/T），"
        "树的根节点会换成 URL 特殊字符比例或域名长度，说明 URL 词法特征才是更接近“内容”的信号。"
    ))

    add_paragraph(doc, (
        "综合本节结果：本数据集的特征冗余度较高，20 个左右的特征就能保留大部分性能；"
        "PCA 可以压缩特征但会损失细节；多项式升维没有帮助；"
        "置换重要性与 SHAP 一致地指向 URL 词法特征与页面规模特征。"
        "最重要的结论不是“哪个特征排名第一”，而是“排名靠前的特征里有相当一部分是数据集采集方式带来的捷径”。"
    ))


def build_section7(doc, R):
    res = R["res"][R["res"]["model"].isin(MAIN)]
    s = R["shortcut"]
    add_h1(doc, "七、总结")

    add_h2(doc, "7.1　主要结论")
    add_paragraph(doc, (
        f"本项目在 PhiUSIIL 数据集上完成了从数据审计、防泄漏划分、五类模型训练到调参、鲁棒性与特征工程的完整流程。"
        "最重要的发现不是某个模型多强，而是这个数据集本身存在严重的单向捷径："
        f"URLSimilarityIndex 在全部合法样本上恒为 100，IsHTTPS 在全部合法样本上恒为 1；"
        f"两条简单规则在测试集上就取得 F1 = {f4(s['test_metrics']['f1'])}、FPR = 0。"
        "因此，任何使用全特征方案的模型都会接近满分，模型之间的差异被捷径掩盖。"
    ))
    add_paragraph(doc, (
        "为了把“真实能力”从“捷径”中分离出来，我们设计了 A → B → S → T 的逐步消融，以及 C → U → V 的 URL-only 消融。"
        "结果显示：去掉官方衍生特征（B）与页面规模特征（S/T）后，随机森林与逻辑回归仍能保持 0.996 以上的 F1，"
        "说明页面标志与 URL 词法本身也携带了大量信息；真正让模型拉开差距的是 V。"
        "在 V 上，逻辑回归最好（0.9960），随机森林次之（0.9951），RBF SVM 为 0.9942，"
        "决策树降到 0.7835，朴素贝叶斯只有 0.5870。"
        "因此，“哪个模型最好”没有唯一答案：页面特征主导时随机森林占优，URL 词法主导时逻辑回归占优。"
    ))
    add_paragraph(doc, (
        "鲁棒性实验补充了三点。第一，逻辑回归在 V 上最省样本（1% 样本时 F1 已约 0.984），树模型需要更多数据；"
        "第二，标签噪声对逻辑回归影响最大、对随机森林影响最小，特征噪声则会让所有模型严重退化；"
        "第三，跨 TLD 迁移会带来小幅下降，说明模型学到了与域名分布相关的模式。"
        "实际部署时不能只用随机划分的测试集评估，应按 TLD、时间或来源做分布外测试。"
    ))
    add_paragraph(doc, (
        "特征工程方面，约 20 个特征就能保留大部分性能，PCA 可以压缩特征但会损失细节，二阶多项式升维没有帮助；"
        "置换重要性与 SHAP 一致地指向 URL 词法特征和页面规模特征。"
        "这提醒我们：在报告“模型准确率”之前，先检查特征是否含有标签代理，比换模型更重要。"
    ))

    add_h2(doc, "7.2　局限与未来工作")
    add_paragraph(doc, (
        "本项目有以下局限。第一，只使用了一个数据集，PhiUSIIL 的采集方式与标注规则可能带来特定偏差，"
        "结论不能直接外推到真实流量；第二，只做了一次分组划分，没有按时间划分，无法检验概念漂移；"
        "第三，RBF SVM 只在 20,000 条子样本上训练，样本量受限；第四，没有做概率校准与代价敏感学习，"
        "阈值优化只考虑了“召回率 ≥ 0.99”这一种业务约束；第五，页面规模特征需要抓取并解析网页，"
        "在线部署的成本和稳定性都需另做评估；第六，超参数只在方案 A 上选择一次，"
        "导致决策树在 V 上欠拟合、朴素贝叶斯的概率尺度未校准。如果为每个方案单独调参，绝对数值会变化，"
        "但“算法优势随特征类型变化”的主结论预计不变。"
    ))
    add_paragraph(doc, (
        "未来可以从四个方向继续：一是引入 PhishTank、OpenPhish 等外部数据做跨数据集验证；"
        "二是按时间切分训练/测试，观察模型在概念漂移下的衰减；"
        "三是在数据采集阶段就移除 URLSimilarityIndex、IsHTTPS 等标签代理，或明确把它们作为“不可用特征”；"
        "四是尝试概率校准、代价敏感学习与深度学习（字符级 CNN/Transformer、HTML 图神经网络），"
        "并与本项目的传统模型做公平比较。"
    ))

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
    add_h2(doc, "A　防泄漏检查清单")
    items = [
        "测试集按 Domain 分组切出，训练/测试 Domain 重叠为 0；测试集只在最终评估时使用一次。",
        "缩放、频次编码、缺失值填补、特征选择、PCA 全部在训练折内拟合。",
        "超参数、分类阈值、特征选择都没有使用测试集；阈值只用训练集 OOF 预测确定。",
        "URLSimilarityIndex、IsHTTPS 等单向捷径的来源与影响已量化，并做了 B/S/T/U/V 消融。",
        "划分索引、SVC 子样本索引与随机种子已保存到 splits/，所有模型共用同一套边界。",
        "SVC 只在训练集内抽取的 20,000 条分层子样本上训练，未使用测试集。",
    ]
    for it in items:
        add_bullet(doc, it, size=10.5, symbol="☑")

    add_h2(doc, "B　运行环境与复现")
    add_paragraph(doc, (
        "Python 3.13.9；numpy 2.3.5；pandas 2.3.3；scipy 1.16.3；scikit-learn 1.7.2；"
        "matplotlib 3.10.6；seaborn 0.13.2；shap 0.54.1；python-docx 1.2.0；python-pptx 1.0.2。"
        "所有随机过程固定 random_state = 42。完整运行顺序与依赖见项目 README.md；"
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

    out = ROOT / "实验2报告_钓鱼网站检测.docx"
    doc.save(out)
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()

