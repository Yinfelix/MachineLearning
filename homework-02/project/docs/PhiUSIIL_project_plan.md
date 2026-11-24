# 机器学习课程项目 2：PhiUSIIL 钓鱼网站检测——项目方案（四人小组）

> 一句话主线：在这个“很容易拿高分”的数据集上，我们不追求分数，而是回答三个问题——
> ① 哪些特征真正在起作用？② 去掉“捷径特征”后，不同算法的表现有何差异？③ 模型在新域名、噪声、小样本下还可靠吗？

---

## 0. 总览

| 项目 | 内容 |
|---|---|
| 任务 | 二分类：判断网址（及其网页）是否为钓鱼网站 |
| 数据集 | PhiUSIIL Phishing URL（UCI，Prasad & Chandra，2024，CC BY 4.0） |
| 数据页 | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset |
| 论文 | Computers & Security，DOI：10.1016/j.cose.2023.103545 |
| 规模 | 235,795 条；54 个特征（UCI 变量表共 56 项）；合法 134,850 / 钓鱼 100,945；无缺失值 |
| 语言与环境 | Python 3.10+ |
| 核心库 | pandas、numpy、scikit-learn、matplotlib、seaborn；拓展二另用 shap |
| 基础任务 | 完整完成（数据→模型→评估→优化） |
| 拓展任务（只选 2 个） | ① 泛化与鲁棒性；② 特征工程（含可解释性） |

> 说明：以下关于特征名称、`URLSimilarityIndex` 性质的描述来自 UCI 页面、论文引用方及第三方复现资料，**尚未用原始数据逐项核实**。第 1.1 节的“开题验证”就是为此准备的，报告里以验证结果为准。

---

## 1. 数据准备与预处理

### 1.1 开题验证（第 1 周前两天，全员参与、成员 1 负责）

目的：把“听说”变成“验证过”，并决定后面是否把某些特征列为捷径特征。

1. 确认列名、类型、行数，与 UCI 页面一致（54 个特征 + 标签 + FILENAME 等）。
2. 统计重复 URL、重复 Domain 的数量与占比；统计每个 Domain 的样本数分布。
3. 画 `URLSimilarityIndex` 在两类上的分布直方图，观察是否近乎完美分开。
4. 计算每个特征的单变量 AUC，按大小排序，列出前 10 名。
5. 检查官方标注的衍生特征（CharContinuationRate、URLTitleMatchScore、URLCharProb、TLDLegitimateProb）与原始特征的相关性。
6. 检查合法类与钓鱼类在 URL 形态上是否有明显差异（如是否以 `https://www.` 开头、IsHTTPS 的取值分布）。

产出：一页“数据体检报告”（图 + 结论），汇报时作为“为什么我们要做消融与分组划分”的依据。

### 1.2 任务与数据说明（写进报告第 2 章）

- 输入特征：URL 结构、域名、混淆、HTML 结构、表单交互、关键词、页面内容等（具体列名以实际 `df.columns` 为准）。
- 分类目标：原标签 1 = 合法、0 = 钓鱼。**重编码为 `y = 1` 表示钓鱼（正类）**，这样 Precision、Recall 直接对应“抓钓鱼站”。
- 样本分布：两类比例约 57% : 43%，轻度不平衡；用柱状图展示。
- 特征类型：连续比例/概率、整数计数、二值标志、高基数类别（URL、Domain、TLD）。

### 1.3 预处理方案

| 问题 | 处理方式 | 库与组件 |
|---|---|---|
| 无用列 | 删除 `FILENAME` | pandas `drop` |
| 高基数文本（URL、Domain、Title 等） | 不直接入模；Domain 仅用于**分组划分** | pandas |
| TLD | 训练集内频次编码，罕见 TLD 合并为“其他”；对 LR/SVM 另做独热对照 | 自定义 `TransformerMixin` 或 `category_encoders` 之外尽量用 sklearn 实现 |
| 长尾计数特征（如 LineOfCode、NoOfJS） | `log1p` 变换 | `FunctionTransformer(np.log1p)` |
| 连续特征尺度 | LR、SVM、朴素贝叶斯使用标准化；树模型不缩放 | `StandardScaler` |
| 缺失值 | 官方无缺失，仍在 Pipeline 中保留，并在报告写明“已检查” | `SimpleImputer` |
| 异常值 | 不删除，先观察；鲁棒性实验中单独分析 | — |

实现要求：全部写进 `Pipeline` + `ColumnTransformer`，保证**每个训练折内独立拟合**预处理参数。

### 1.4 数据划分（防泄漏的关键设计）

```python
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# 1) 按 Domain 分组，切出 20% 独立测试集（同一域名不会同时出现在训练与测试）
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=domain))

# 2) 训练集内部：5 折分层分组交叉验证，作为“验证集”用于选模型、调参
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

- 测试集：只在最终评估时使用一次。
- 验证集：训练集内部 5 折交叉验证（每折的验证部分），不另外固定切一份，减少浪费，也符合任务书“可采用训练集内部交叉验证”。
- 划分索引保存成文件（`splits/train_idx.npy`、`test_idx.npy`、`cv_folds.json`），保证所有成员用同一套划分。
- 若 Domain 重复很少，则退化为普通分层划分，并在报告里如实说明。

### 1.5 四套特征方案（贯穿全项目）

| 方案 | 内容 | 目的 |
|---|---|---|
| A 全特征 | 去掉 FILENAME、URL、Domain 后的全部特征 | 基线，预计“很容易高分” |
| B 去捷径 | A 去掉 `URLSimilarityIndex` 及官方标注的衍生特征 | 考察真实判别能力 |
| C 仅 URL | 只用 URL 词法与域名类特征 | 判断“只看网址”能做到多好 |
| D 仅页面内容 | 只用 HTML、表单、内容类特征 | 判断“只看页面”能做到多好 |

分组以 `df.columns` 实际内容为准，由成员 1 整理成 `feature_groups.json`，所有人共用。

---

## 2. 分类模型构建（Python + scikit-learn）

统一使用 **scikit-learn** 的现成实现，接口一致，便于公平比较。

| 模型 | 实现 | 关键超参数与搜索范围 | 选择理由 |
|---|---|---|---|
| 逻辑回归 | `sklearn.linear_model.LogisticRegression` | `C`∈{0.01, 0.1, 1, 10, 100}；`penalty`∈{l1, l2}（solver=`saga` 或 `liblinear`） | 线性基线，系数可解释 |
| 朴素贝叶斯 | `sklearn.naive_bayes.GaussianNB`（对二值特征可补充 `BernoulliNB`） | `var_smoothing`∈{1e-9 … 1e-5} | 生成式模型，检验“特征独立假设”在相关特征下的影响 |
| 决策树 | `sklearn.tree.DecisionTreeClassifier` | `max_depth`∈{3,5,8,12,None}；`min_samples_leaf`∈{1,5,20,50}；`ccp_alpha` | 规则易讲，易展示过拟合与剪枝 |
| 随机森林 | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`∈{100,300,500}；`max_depth`；`max_features`∈{sqrt,0.3,0.5}；`min_samples_leaf` | Bagging 降方差，通常很强 |
| 支持向量机 | `sklearn.svm.SVC(kernel="rbf")`，对照 `LinearSVC` | `C`∈{0.1,1,10}；`gamma`∈{scale,0.01,0.1} | 最大间隔、非线性边界 |

补充说明：

- SVC 在二十多万样本上训练很慢，**用分层随机子样本（约 2 万条，训练集内抽取）训练与调参**，在报告中说明原因与影响；同时用 `LinearSVC` 跑全量作对照。
- 可选对照：`sklearn.ensemble.HistGradientBoostingClassifier`，展示 Bagging 与 Boosting 的差别（非必做）。
- 每个模型在报告里用三句话写清：基本原理、关键超参数含义、选择理由。
- 所有模型都放进同一个 `Pipeline`（预处理 → 模型），随机种子统一。

---

## 3. 模型训练与评估

### 3.1 公平比较条件

- 同一套划分、同一套预处理流程、同一套指标；
- 每个模型在每套特征方案（A、B、C、D）上各跑一遍；
- 超参数只用训练集的交叉验证选择（`GridSearchCV` 或 `RandomizedSearchCV`，`cv` 使用上面的 `StratifiedGroupKFold`，`scoring="f1"`）。

### 3.2 评价指标（均以“钓鱼 = 正类”）

- Accuracy、Precision、Recall、F1（核心四项）；
- ROC-AUC、PR-AUC（`roc_auc_score`、`average_precision_score`）；
- 对合法网址的误报率（False Positive Rate）：误封正常网站的代价需要单独看。

### 3.3 报告方式

- 验证阶段：5 折均值 ± 标准差；
- 最终阶段：在测试集上各模型只评估一次；
- 所有结果汇总进 `results/experiment_log.csv`（见第 7 节）。

### 3.4 结果分析的写法

对每个模型回答“为什么强或弱”，结合数据特点：

- 朴素贝叶斯：特征高度相关，独立性假设被破坏；
- 逻辑回归：若线性已足够，说明在该特征空间下类别接近线性可分；
- 树与森林：能处理阈值型特征和长尾计数；
- 方案 A 与 B 的差异：量化“捷径特征”的贡献。

---

## 4. 模型优化与分析

### 4.1 可视化（`matplotlib` + `seaborn`）

**真实值与预测值对比**
- 混淆矩阵（计数与按行归一化，`ConfusionMatrixDisplay`）；
- 预测概率分布直方图（按真实类别着色，看分得开不开）；
- PCA 二维散点图（`sklearn.decomposition.PCA`），颜色表示预测对错，直观看错分样本的位置。

**模型对比**
- 指标分组柱状图（带误差线）：模型 × 特征方案；
- ROC 曲线与 PR 曲线（`RocCurveDisplay`、`PrecisionRecallDisplay`）；
- 特征消融热力图：行 = 模型，列 = 特征方案 A/B/C/D，值 = F1。

**过程分析**
- 学习曲线（`learning_curve`）；
- 决策树的训练/验证曲线随 `max_depth`、`ccp_alpha` 的变化。

### 4.2 模型优化（至少两个）

1. **随机森林调参**：`RandomizedSearchCV` 搜索 `n_estimators`、`max_depth`、`max_features`、`min_samples_leaf`；对比默认参数与调优后在验证集（CV）与测试集上的 F1、AUC，并讨论提升是否显著、原因是什么（例如已接近上限，则说明收益有限）。
2. **决策树剪枝**：画出不同 `max_depth` 与 `ccp_alpha` 下的训练/验证曲线，说明过拟合与剪枝的关系。
3. **阈值优化**：在验证集预测概率上调整分类阈值，在召回率达到目标（如 ≥ 0.99）的前提下，报告误报率的变化。**阈值只在验证集上确定**，再用于测试集。

---

## 5. 拓展任务（选 2 个，展开设计）

### 拓展一：泛化与鲁棒性

目的：回答“模型在不理想条件下还可靠吗”。每个实验先写**预测**（哪个模型会更稳），再看结果，便于汇报时讲“预测对不对、为什么”。

| 实验 | 做法 | 观察指标 |
|---|---|---|
| 1. 训练数据规模 | 训练集分别取 1%、5%、10%、25%、50%、100%（分层抽样，每个比例重复 3 个随机种子） | F1、AUC 随样本量的曲线；哪个模型最“省样本” |
| 2. 标签噪声 | 训练集随机翻转 0%、5%、10%、20% 的标签（测试集标签保持不变） | F1 下降幅度；树模型是否更容易记住噪声 |
| 3. 特征噪声 | 对数值特征加高斯噪声，强度为各特征标准差的 0、0.1、0.5、1 倍（噪声参数用训练集统计量） | 对线性、距离、树模型的影响差异 |
| 4. 异常值 | 在测试集随机选少量样本，将若干特征替换为极端值；对比 `StandardScaler` 与 `RobustScaler` | 缩放方式对异常值的敏感性 |
| 5. 跨域名迁移 | 按 TLD 或 Domain 划分：用常见 TLD 训练，在罕见 TLD 上测试；与随机划分对比 | 泛化差距（gap） |
| 6. 过拟合分析 | 汇总每个模型的训练 F1 与验证 F1 之差 | 决策树 vs 随机森林 vs SVM |

交付物：一组带误差线的折线图 + 一页“结论与解释”。

### 拓展二：特征工程（含可解释性）

目的：回答“哪些特征真正起作用，多余特征能不能去掉”。

| 步骤 | 做法与库 | 注意点 |
|---|---|---|
| 1. 特征相关性 | `df.corr(method="spearman")`，`seaborn.heatmap`；列出 \|ρ\| > 0.95 的特征对 | 只在训练集上计算 |
| 2. 单变量筛选 | 单特征 AUC、`mutual_info_classif` | 只在训练集上计算 |
| 3. 过滤法选择 | `SelectKBest` 放进 `Pipeline`，K = 5、10、20、全部，对比 F1 | **选择过程在每个训练折内进行**，避免泄漏 |
| 4. 嵌入法选择 | L1 逻辑回归的正则化路径（不同 C 下非零系数个数）；随机森林 `feature_importances_` | 与过滤法结果对比 |
| 5. 降维 | `PCA` 保留 90%、95%、99% 方差；对 LR、SVM 重新训练并对比 | PCA 在训练折内拟合 |
| 6. 特征升维 | 对重要性前 k 个特征做 `PolynomialFeatures(degree=2)`，再训练 LR | 观察线性模型能否通过升维追上非线性模型 |
| 7. 模型可解释性 | 置换重要性（`sklearn.inspection.permutation_importance`，在验证集上算）；`shap.TreeExplainer` 作用于随机森林，抽样 2000 条；浅层决策树的规则图（`plot_tree`） | 对比“基于树的重要性”与“置换重要性”是否一致，并与消融实验相互印证 |

核心对照：把方案 A/B/C/D 的消融结果、重要性排序、SHAP 结果放在一起，讲清楚“模型到底在看什么”。

---

## 6. 四人分工

| 成员 | 基础任务 | 拓展任务 |
|---|---|---|
| 成员 1：数据负责人 | 开题验证、EDA、预处理 Pipeline、数据划分与泄漏检查、`feature_groups.json` | 拓展二：相关性分析、单变量筛选、过滤法/嵌入法选择 |
| 成员 2：模型负责人（线性与概率类） | 逻辑回归、朴素贝叶斯、LinearSVC 的训练与调参；决策树剪枝实验 | 拓展二：降维（PCA）、特征升维、置换重要性、SHAP |
| 成员 3：模型负责人（树与核方法） | 决策树、随机森林、SVM（子样本）的训练与调参；随机森林调参对比、阈值优化 | 拓展一：实验 1（样本量）、2（标签噪声）、3（特征噪声） |
| 成员 4：评估与汇报负责人 | 统一评估脚本、实验记录表、全部可视化、结果汇总、报告整合与汇报稿 | 拓展一：实验 4（异常值）、5（跨域名迁移）、6（过拟合分析） |

协作规则：
- 所有人使用同一份划分文件和同一个评估函数（`evaluate.py`），避免口径不一致；
- 每周碰头一次，每个人用 3 分钟讲自己的结果，互相提问；
- 汇报时，每个成员都要能解释整个流程，不只是自己负责的部分。

---

## 7. 过程透明与可复现

### 7.1 目录结构

```
project/
├── README.md                 # 运行说明
├── requirements.txt          # 依赖与版本
├── data/                     # 原始数据（获取方式见下）
├── splits/                   # 划分索引（所有人共用）
├── configs/feature_groups.json
├── src/
│   ├── data.py               # 读取、编码目标、分组
│   ├── preprocess.py         # ColumnTransformer / Pipeline
│   ├── models.py             # 五个模型与超参数网格
│   ├── evaluate.py           # 统一指标与记录
│   └── plots.py              # 所有绘图函数
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_tuning.ipynb
│   ├── 04_extension_robustness.ipynb
│   └── 05_extension_features.ipynb
├── results/
│   ├── experiment_log.csv
│   └── figures/
└── report/
```

### 7.2 数据获取与环境

```bash
pip install ucimlrepo pandas numpy scikit-learn matplotlib seaborn shap
```

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=967)      # 或直接下载 CSV：PhiUSIIL_Phishing_URL_Dataset.csv
X, y = ds.data.features, ds.data.targets
```

- 需要在 `requirements.txt` 中固定版本，并在报告里写明 Python 与库版本；
- 所有随机过程固定 `random_state`（如 42），并记录。

### 7.3 实验记录表（`experiment_log.csv`）

| 字段 | 说明 |
|---|---|
| exp_id | 实验编号 |
| 日期、负责人 | 谁在什么时候跑的 |
| 特征方案 | A / B / C / D |
| 模型、超参数 | 完整配置 |
| 随机种子 | — |
| 验证集指标 | 均值 ± 标准差 |
| 测试集指标 | 仅最终评估时填写 |
| 备注 | 观察到的现象 |

### 7.4 防泄漏检查清单（写进报告附录）

- [ ] 测试集按 Domain 分组切出，且只用一次
- [ ] 预处理（缩放、编码、PCA、特征选择）全部在训练折内拟合
- [ ] 超参数、阈值、特征选择都没有使用测试集
- [ ] 衍生特征的来源已说明，并做了去捷径对照（方案 B）
- [ ] 划分索引与随机种子已保存

---

## 8. 时间计划（3 周）

| 周次 | 目标 | 产出 |
|---|---|---|
| 第 1 周 | 开题验证、预处理、划分、五个模型的基线（默认参数）跑通 | 数据体检报告、基线结果表 |
| 第 2 周 | 调参与优化、四套特征方案对比、基础可视化、拓展实验启动 | 优化前后对比、消融热力图 |
| 第 3 周 | 拓展实验收尾、结果分析、报告撰写、汇报彩排 | 完整报告（15 页以内）、代码与运行说明、汇报稿 |

---

## 9. 报告结构（建议正文 15 页以内）

| 章节 | 内容 | 页数 |
|---|---|---|
| 1 课题概述 | 背景、目标、分工 | 1 |
| 2 数据处理 | 数据来源与分布、预处理、划分、四套特征方案 | 3 |
| 3 模型构建 | 五个模型的原理、超参数与理由 | 2 |
| 4 训练与评估 | 评价指标、公平比较、结果分析 | 3 |
| 5 可视化与优化 | 图表与优化前后对比 | 3 |
| 6 拓展任务 | 拓展一、拓展二 | 2 |
| 7 总结 | 结论、局限、未来工作 | 1 |
| 附录 | 防泄漏清单、实验记录表摘要、环境与运行说明 | — |

---

## 10. 汇报主线（约 10 分钟）

1. **为什么选它**：数据新、规模大，但有“容易拿高分”的风险。（1 分钟）
2. **怎么设计实验**：分组划分、防泄漏、四套特征方案。（2 分钟）
3. **五个模型的对比**：谁强、谁弱、为什么。（2 分钟）
4. **去掉捷径后发生了什么**：消融结果。（2 分钟）
5. **调参与优化**：收益有多大，为什么有限或有效。（1 分钟）
6. **拓展**：鲁棒性结论与“模型到底在看什么”。（1.5 分钟）
7. **总结**：模型在什么条件下可靠，什么条件下不可靠。（0.5 分钟）

---

## 11. 风险与应对

| 风险 | 应对 |
|---|---|
| 所有模型都接近 100%，缺乏区分度 | 重点看方案 B/C/D、噪声与跨域名迁移，这些才是分析价值所在 |
| SVM 训练太慢 | 子样本训练 + `LinearSVC` 全量对照，并在报告中说明 |
| Domain 重复很少，分组划分意义不大 | 如实报告，改用分层划分，并保留 TLD 迁移实验 |
| 验证结果推翻了前面的假设（如 `URLSimilarityIndex` 并不关键） | 以数据为准，报告里写“假设—验证—结论” |
| 拓展实验太多做不完 | 优先级：样本量 → 标签噪声 → 跨域名迁移；特征工程优先做相关性、置换重要性、SHAP |
