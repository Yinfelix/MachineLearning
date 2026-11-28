# 第二次作业：PhiUSIIL 钓鱼网站检测

本目录为机器学习课程项目 2 的提交内容。任务是以 PhiUSIIL Phishing URL Dataset 为对象，完成一个二分类任务：判断网址（及其页面）是钓鱼网站还是合法网站。

```text
homework-02/
├── archive/data/                     # 原始数据压缩包
├── homework - needs/                 # 作业要求 PDF
└── phiusiil-phishing/                # 项目代码、实验产物、报告与 PPT
    ├── src/                          # 数据、预处理、模型、评估、绘图
    ├── scripts/                      # 01–07 实验脚本 + run_all.py
    ├── configs/ splits/              # 特征方案与固定划分
    ├── outputs/figures, metrics/     # 图表与指标
    ├── tests/                        # pytest
    ├── report/                       # 报告 / PPT 生成脚本
    ├── 实验2报告_钓鱼网站检测_参考版.docx
    └── 实验2汇报_钓鱼网站检测_参考版.pptx
```

完整的数据说明、运行方式与实验设计见 [`phiusiil-phishing/README.md`](phiusiil-phishing/README.md)。

核心发现（详见报告）：

- `URLSimilarityIndex` 在全部合法样本上恒为 100，`IsHTTPS` 在全部合法样本上恒为 1；两条规则在测试集上即达到 F1 = 0.9999、误报为 0。
- 全特征方案下多数模型接近天花板，模型比较没有区分度；按 A → B → S → T 与 C → U → V 做多级消融后，才能看出算法差异。
- 没有单一赢家：页面特征主导时随机森林最好，URL 词法主导时逻辑回归最好；决策树/朴素贝叶斯在严格方案上崩塌。
- 标签噪声与跨 TLD 迁移是主要风险。
