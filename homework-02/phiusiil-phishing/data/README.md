# 数据说明

原始文件：`../../archive/data/phiusiil+phishing+url+dataset.zip`（56,854,345 字节）

SHA-256：`0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a`

解压后得到 `PhiUSIIL_Phishing_URL_Dataset.csv`，放到 `data/raw/`：

```bash
unzip ../../archive/data/phiusiil+phishing+url+dataset.zip -d data/raw/
```

数据来源与引用：

- UCI Machine Learning Repository, PhiUSIIL Phishing URL Dataset (ID 967)
  <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and lexical features. *Computers & Security*, 136, 103545.
  DOI: 10.1016/j.cose.2023.103545
- 许可：CC BY 4.0

CSV 不纳入版本控制；脚本只读取 `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`。
