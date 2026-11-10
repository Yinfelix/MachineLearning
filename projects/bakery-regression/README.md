# 烘焙订单购买数量预测

以订单明细为单位，使用下单时可获得的信息预测 `Quantity`，比较经典回归模型，
并研究特征工程与特征选择。完整研究需求见 [requirements.md](../requirements.md)。

## 阶段计划

| 阶段 | 任务 | 交付与验收 |
| --- | --- | --- |
| 1. 数据探索与预处理 | 审计数据与泄漏字段，按时间划分数据，建立预处理 Pipeline | 已完成：数据审计结果、探索 Notebook、可复用预处理代码和测试 |
| 2. 基础模型实验 | 建立 Dummy、Ridge、决策树和随机森林，统一 MAE、RMSE、R²；以训练集和验证集比较模型并优化参数 | 模型比较结果、参数记录、性能对比图 |
| 3. 特征工程与消融 | 对比 FE-0 至 FE-3，研究时间周期、商品交互、特征选择及客户、时间、价格、天气、商品特征消融 | 特征工程实验矩阵、重要性与消融分析图表 |
| 4. 最终评估与报告 | 固定实验方案，合并训练集与验证集重新训练，在测试集评估并分析残差 | 最终指标、预测与残差图、实验报告 |

后续实验按以下顺序开展：

1. **阶段 2（EXP1–EXP2）**：以训练集中位数建立 Dummy 基准，比较三种回归模型；
   在训练集内按日期开展时序调参，用验证集选择模型和参数。
2. **阶段 3（EXP3–EXP6）**：FE-0 使用基础特征；FE-1 加入时间周期与趋势；
   FE-2 加入商品交互；FE-3 比较互信息、排列重要性和 Top-K 特征选择，并完成特征消融。
3. **阶段 4（EXP7）**：固定方案后合并训练集与验证集重新训练，在测试集计算
   MAE、RMSE、R²，整理预测图、残差图和实验报告。

阶段 1 的数据质量概览统计全量记录的字段、缺失、时间范围及重复情况；目标分布、
相关性和数量关系分析使用训练集。各阶段的模型、参数与特征选择使用训练集和验证集；
测试集用于最终评估。

## 数据与字段

项目直接读取 [bakery_sales_dataset.csv](../archive/data/bakery_sales_dataset.csv)，
一条记录表示一次订单中的一种商品。原始数据包含 19,321 条记录、40 个字段，目标
`Quantity` 表示购买数量。

| 数据集 | 日期范围 | 记录数 |
| --- | --- | ---: |
| 训练集 | 2023-01-01 至 2024-06-30 | 14,340 |
| 验证集 | 2024-07-01 至 2024-09-30 | 2,533 |
| 测试集 | 2024-10-01 至 2024-12-31 | 2,448 |

输入字段采用白名单，包括商品、客户、价格、促销、时间、环境、商品属性与门店信息。
实际客户字段名为 `Customer_Gender` 和 `Customer_Segment`。`Date` 用于划分，
`Year`、`Month`、`Day_of_Week`、`Weekend` 和 `Hour` 用作基础时间特征。

`Total_Bill`、`Profit`、`Units_Sold` 等结果字段、交易后字段和标识符均排除在输入之外。
`Selling_Price` 可由单价与折扣推导，基础实验使用 `Unit_Price` 和
`Discount_Percentage`；`Cost_Price` 和 `Day` 不在基础候选字段中。
完整字段清单及排除原因见探索 Notebook。

数据核验发现，`Units_Sold` 有 5,161 条记录大于 `Quantity`，其余 14,160 条相等。
账单与利润使用 `Quantity` 计算，因此两种数量的业务含义需要分别表述。

## 预处理

- 数值列使用训练集的中位数填补缺失，并可按模型需要使用 `StandardScaler`。
- 类别列使用 One-Hot 编码，验证集和测试集的新类别可正常转换。
- `Loyalty_Member`、`Weekend`、`Promotion_Applied` 使用 `No=0`、`Yes=1` 编码。
- 空字符串和纯空白统一作为缺失处理；普通类别及二元字段使用训练集众数填补。
- `Festival` 缺失值填为 `Not Specified`；未促销且类型缺失时填为 `No Promotion`，
  已促销或状态缺失时的空类型填为 `Not Specified`。
- 时间周期编码、特征交互和特征选择安排在阶段 3。

`build_preprocessor()` 返回尚未拟合的 `ColumnTransformer`。训练时将它放入模型的
`Pipeline`，只对训练集调用 `fit`；可通过 `scale_numeric=False` 为树模型关闭标准化。
审计会核验日期、星期和小时字段的一致性。

数据读取、清理和审计命令正常运行时保持静默。日期缺失或无效、目标值缺失或非正整数、
日期超出配置范围以及订单跨数据集的记录会被跳过，原因和数量写入审计 JSON 的
`cleaning` 字段。同一记录按上述顺序归入首个原因，确保原因计数之和等于跳过记录数。
空时间区间保留为空表，未定义的统计量保存为 `null`。

## 代码与配置

| 文件 | 职责 |
| --- | --- |
| [config.json](config.json) | 数据和输出路径、字段白名单、排除原因、时间区间、填补标签、审计容差 |
| [src/config.py](src/config.py) | 读取 JSON 并统一解析项目相对路径 |
| [src/dataset.py](src/dataset.py) | 读取 CSV、清理异常记录、统计跳过原因、划分数据集 |
| [src/preprocessing.py](src/preprocessing.py) | 缺失填补、类别编码、数值标准化 |
| [src/data_audit.py](src/data_audit.py) | 数据质量与训练集分析、静默生成 JSON 报告 |

`load_dataset()` 返回 `(data, cleaning_report)`，分别是清理后的数据表和清理统计。
将二者传给 `audit_dataset(data, cleaning_report)`，再由 `write_audit_report()` 保存报告。
修改配置后重新运行程序或重启 Notebook 内核即可生效。

依赖按实际用途保留：NumPy、Pandas 和 Scikit-learn 用于数据处理；Matplotlib 用于
Notebook 绘图；Joblib 用于 Pipeline 持久化测试；IPykernel 和 NBConvert 用于
Notebook 运行与执行验证。配置使用 Python 标准库 JSON。

## 运行

从工作区根目录安装共用依赖，再进入项目目录：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
Set-Location .\projects\bakery-regression
```

生成数据审计结果：

```powershell
..\..\.venv\Scripts\python.exe -m src.data_audit
```

使用 VS Code 打开 [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)，
选择工作区 `.venv` 内核并运行全部单元格。Notebook 展示数据审计、字段筛选、训练集
分布与相关性，并验证训练集拟合及验证集转换。运行后生成：

- `outputs\metrics\data_audit.json`
- `outputs\figures\01_quantity_distribution.png`
- `outputs\figures\02_training_correlations.png`

也可以在项目目录中通过命令执行 Notebook：

```powershell
..\..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace .\notebooks\01_data_exploration.ipynb
```

运行阶段 1 测试：

```powershell
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
