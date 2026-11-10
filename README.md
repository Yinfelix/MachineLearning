# MachineLearning

仓库包含两部分：根目录的机器学习学习材料，以及 [projects/](projects/) 下的独立项目。
学习示例、练习、参考答案、模型文件和根目录重复数据保留在本地，并由 `.gitignore` 排除；
`projects/` 下的项目代码、数据、Notebook、测试和结果纳入版本控制。

## 项目代码

### 烘焙订单购买数量预测

[projects/bakery-regression/](projects/bakery-regression/) 根据订单商品、客户、价格、
时间和环境特征预测订单明细中的 `Quantity`，研究经典回归模型与特征工程。

- [项目说明与运行指南](projects/bakery-regression/README.md)
- [完整需求与实验计划](projects/requirements.md)
- [数据探索 Notebook](projects/bakery-regression/notebooks/01_data_exploration.ipynb)
- [预处理代码](projects/bakery-regression/src/preprocessing.py)
- [数据读取与时间划分](projects/bakery-regression/src/dataset.py)
- [数据审计代码](projects/bakery-regression/src/data_audit.py)
- [配置文件](projects/bakery-regression/config.json)
- [阶段一审计结果](projects/bakery-regression/outputs/metrics/data_audit.json)

阶段一已完成数据审计、时间划分、预处理 Pipeline 和探索 Notebook。后续阶段将比较
Dummy、Ridge、决策树和随机森林，开展特征工程与消融，并完成最终测试集评估。

## 本地学习材料

工作区也保留 TensorFlow 演示与练习目录：

- `examples/`：TensorFlow、自动微分、回归、分类、数据管道与模型保存演示。
- `exercises/`：配套练习。
- `solutions/`：参考答案。
- `tf_demo.py`：最初的线性拟合示例。
- `saved_model/`：本地模型文件。

这些学习材料适合本地练习，不纳入 Git 提交。

## 统一安装依赖

仓库共用根目录的 `requirements.txt`，同时涵盖 TensorFlow 学习材料和 `projects/` 中
的项目代码、Notebook 与测试所需依赖。

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

无需 VS Code Python 虚拟环境时，也可以使用当前 Python 安装：

```powershell
python -m pip install -r requirements.txt
```

运行烘焙项目的审计和测试：

```powershell
Set-Location .\projects\bakery-regression
..\..\.venv\Scripts\python.exe -m src.data_audit
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

运行其他 Notebook 的步骤及数据说明请见 [烘焙项目指南](projects/bakery-regression/README.md)。
