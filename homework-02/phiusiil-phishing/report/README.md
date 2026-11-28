# 报告与 PPT 生成

本目录只负责排版，不重复训练模型。先生成全部实验产物：

```bash
python scripts/01_audit_eda.py
python scripts/02_tune.py
python scripts/03_final_eval.py
python scripts/04_optimize.py
python scripts/05_robustness.py
python scripts/06_feature_engineering.py
python scripts/07_report_assets.py
python scripts/08_posthoc.py
python scripts/08b_training_controls.py
```

再生成文档：

```bash
python report/build_report.py   # -> ../实验2报告_钓鱼网站检测.docx
python report/build_ppt.py      # -> ../实验2汇报_钓鱼网站检测_参考版.pptx
```

说明：

- `build_report.py` 使用 python-docx，正文 12 pt、1.5 倍行距、首行缩进 2 字符；表格与图片编号由脚本中的顺序决定。
- 公式使用 Word 原生 OMML（Cambria Math），不是图片或纯文本。
- `build_ppt.py` 使用 python-pptx，16:9，配色为蓝/橙/青/紫，字体为 Microsoft YaHei。
- 报告表 1 保留四位成员的姓名和学号空位，供小组填入。PPT 的分工见第 16 页。
- 中文字体依赖 Windows 的宋体、黑体、Microsoft YaHei；在无这些字体的机器上打开可能出现替换，但不影响内容。
