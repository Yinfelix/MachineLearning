# 机器学习课程项目 2：PhiUSIIL 钓鱼网站检测

本项目以 UCI **PhiUSIIL Phishing URL Dataset** 为对象，完成一个二分类任务：判断一条网址（及其页面）是钓鱼网站还是合法网站。项目按“数据审计 → 防泄漏划分 → 多模型训练 → 调参与优化 → 鲁棒性与特征工程拓展”的流程展开，重点不是把分数刷到 100%，而是回答：

1. 数据中哪些特征构成了“捷径”，它们让任务变得多容易？
2. 去掉捷径后，不同算法还剩多少真实判别能力？
3. 模型在样本不足、标签噪声、特征噪声、异常值和跨 TLD 场景下是否可靠？

结论先行：全特征方案下多数模型接近天花板；去掉捷径后没有单一赢家——页面特征主导时随机森林最好，URL 词法主导时逻辑回归最好，决策树与朴素贝叶斯在严格方案上明显退化。

## 数据

- 数据集：PhiUSIIL Phishing URL Dataset（UCI ID 967）
- 来源：<https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- 论文：Prasad & Chandra, *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and lexical features*, Computers & Security, 2024. DOI: 10.1016/j.cose.2023.103545
- 规模：235,795 条，56 列（54 个特征 + 标签 + FILENAME；URL/Domain/Title 为文本列）
- 标签：原始 `label=1` 为合法、`0` 为钓鱼；本项目重编码为 **`y=1` 表示钓鱼（正类）**
- 许可：CC BY 4.0

原始压缩包位于 `../archive/data/phiusiil+phishing+url+dataset.zip`。解压到 `data/raw/`：

```bash
unzip ../archive/data/phiusiil+phishing+url+dataset.zip -d data/raw/
```

## 目录

```text
phiusiil-phishing/
├── configs/feature_groups.json     # 特征分组与方案定义（脚本生成）
├── data/raw/                       # 原始 CSV（不纳入版本控制）
├── splits/                         # 固定划分索引、SVC 子样本、划分元数据
├── src/
│   ├── config.py                   # 路径、随机种子、特征分组、配色
│   ├── data.py                     # 读取、目标编码、分组分层划分
│   ├── preprocess.py               # 频次编码 + 缺失值 + 标准化 Pipeline
│   ├── models.py                   # 五个主模型 + LinearSVC 对照 + 搜索网格
│   ├── evaluate.py                 # 统一指标、交叉验证、结果保存
│   └── plots.py                    # 全部静态图
├── scripts/
│   ├── 01_audit_eda.py             # 数据审计、EDA、单变量 AUC、相关性
│   ├── 02_tune.py                  # 方案 A 上的完整网格搜索（5 折分组 CV）
│   ├── 03_final_eval.py            # 6 模型 × 6 方案的 CV 与测试集评估
│   ├── 04_optimize.py              # RF 调参对比、DT 剪枝、阈值优化
│   ├── 05_robustness.py            # 拓展一：学习曲线、噪声、异常值、TLD 迁移
│   ├── 06_feature_engineering.py   # 拓展二：筛选、PCA、升维、置换重要性、SHAP
│   ├── 07_report_assets.py         # 汇总报告/PPT 所需图表
│   ├── 08_posthoc.py               # V 方案上 DT 深度 / NB 阈值的补充诊断
│   └── 09_verify.py                # 最终产物与指标一致性检查
├── outputs/
│   ├── figures/                    # PNG 图
│   ├── metrics/                    # CSV/JSON 指标与预测分数
│   └── models/                     # 调参后的最佳模型（joblib）
├── tests/                          # pytest：划分、预处理、指标、方案定义
├── report/                         # 报告与 PPT 生成脚本
├── 实验2报告_钓鱼网站检测.docx
└── 实验2汇报_钓鱼网站检测.pptx
```

## 环境与运行

```bash
python -m pip install -r requirements.txt
```

按顺序执行（总耗时约 90–120 分钟，32 核 CPU 实测）：

```bash
python scripts/01_audit_eda.py
python scripts/02_tune.py          # 约 15–25 分钟
python scripts/03_final_eval.py    # 约 5–10 分钟
python scripts/04_optimize.py      # 约 15–20 分钟
python scripts/05_robustness.py    # 约 5–10 分钟
python scripts/06_feature_engineering.py  # 约 10–15 分钟
python scripts/07_report_assets.py
python scripts/08_posthoc.py
python scripts/09_verify.py
```

所有随机过程固定 `random_state=42`；划分索引保存在 `splits/`，因此所有模型与拓展实验共用同一套训练/测试边界。重新生成划分可使用 `data.ensure_splits(df, force=True)`，但会改变测试集，需重跑全部实验。

运行测试：

```bash
python -m pytest -q
```

## 实验设计要点

- **防泄漏划分**：用 `StratifiedGroupKFold(5)` 先按 Domain 分组切出 20% 独立测试集，同一域名不会同时出现在训练与测试；训练集内部再用 `StratifiedGroupKFold(5)` 做交叉验证。缩放、频次编码、特征选择、PCA 全部在训练折内拟合。
- **八套特征方案**：
  - A 全特征（51 列）
  - B 去官方衍生捷径（45 列：去掉 `URLSimilarityIndex` 等 6 个特征）
  - C 仅 URL（23 列）
  - D 仅页面内容（28 列）
  - S 严格去捷径（39 列：在 B 上再去掉 6 个页面规模特征）
  - U 仅 URL 基础词法（19 列：在 C 上去掉 4 个 URL 捷径特征）
  - T 去除全部单向捷径（38 列：在 S 上再去掉 `IsHTTPS`）
  - V 仅 URL 基础词法且去 HTTPS（18 列：在 U 上再去掉 `IsHTTPS`）
  - S/U/T/V 是 EDA 发现“第二组捷径”后补充的消融方案，用于回答“去掉所有强捷径后还剩多少能力”。
- **模型**：逻辑回归、高斯朴素贝叶斯、决策树、随机森林、RBF 支持向量机；LinearSVC 作为全量数据对照。RBF SVM 在训练集内固定的 20,000 条分层子样本上训练（原因见报告）。
- **调参**：在方案 A 上用完整网格 + 5 折分组 CV 选择超参数，然后固定用于所有特征方案，使消融差异只来自特征而非二次调参。逻辑回归主网格搜索 L2 正则（liblinear+L1 在全量数据上单折过慢），L1 在拓展二中以正则化路径单独分析。
- **测试集**：只在最终评估时使用一次，不参与调参、特征选择或阈值选择。

## 主要产出

- `outputs/metrics/results_all.csv`：模型 × 方案的 CV 与测试集指标
- `outputs/metrics/tuning_summary.json`：各模型最佳超参数
- `outputs/metrics/robustness_*.csv`、`feature_selection_k.csv`、`permutation_importance.csv` 等
- `outputs/figures/fig01…fig22`：报告与 PPT 使用的全部图
- `实验2报告_钓鱼网站检测.docx`、`实验2汇报_钓鱼网站检测.pptx`

## 复现说明

- Python 3.13.9（脚本兼容 3.10+），pandas 2.3.3，scikit-learn 1.7.2，numpy 2.3.5，matplotlib 3.10.6，shap 0.54.1。
- 图表中文字体使用 Microsoft YaHei；报告正文使用宋体 + Times New Roman。
- `URLSimilarityIndex` 在本数据发布中“合法样本恒为 100”，这是数据本身的属性，不是本项目构造的；报告对其泄漏风险做了专门讨论。
