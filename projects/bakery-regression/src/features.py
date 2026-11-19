"""按日期构造周期变量，按商品组合类别，并给购买件数的候选特征打分。"""

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.metrics import mutual_info_score

from .config import FEATURE_COLUMNS
from .preprocessing import promotion_type


CYCLES = {"Month": 12, "Weekday": 7, "Hour": 24}
CALENDAR = [f"{name}_{fn}" for name in CYCLES for fn in ("sin", "cos")] + ["Trend"]
PARTNERS = ["Season", "Promotion_Type", "Weekend", "Customer_Segment"]
CROSSES = ["Product_" + col for col in PARTNERS]
GROUPS = {
    "customer": ["Customer_Age", "Customer_Gender", "Customer_Segment", "Loyalty_Member"],
    "calendar": ["Year", "Month", "Day_of_Week", "Weekend", "Hour", "Season", "Festival"],
    "price": ["Unit_Price", "Discount_Percentage", "Promotion_Applied", "Promotion_Type"],
    "weather": ["Weather", "Temperature"],
    "product": ["Product", "Category", "Storage_Type", "Shelf_Life_Days", "Perishability"],
}


def make_features(data, level="base"):
    x = data[FEATURE_COLUMNS].copy()
    if level == "base":
        return x

    date = data["Date"]
    values = {"Month": date.dt.month - 1, "Weekday": date.dt.dayofweek, "Hour": x["Hour"]}
    for name, period in CYCLES.items():
        angle = 2 * np.pi * values[name] / period
        x[name + "_sin"] = np.sin(angle)
        x[name + "_cos"] = np.cos(angle)
    x["Trend"] = (date - np.datetime64("2023-01-01")).dt.days
    if level == "cross":
        promo = promotion_type(x[["Promotion_Type", "Promotion_Applied"]])
        for col in PARTNERS:
            other = promo if col == "Promotion_Type" else x[col]
            x["Product_" + col] = x["Product"].str.cat(other, sep=" / ")
    return x


def quantity_mi(x, y):
    """用互信息衡量单列与实际件数的关联。连续列按训练样本的十分位数分箱。"""
    x = csc_matrix(x)
    scores = []
    for col in range(x.shape[1]):
        values = x.getcol(col).toarray().ravel()
        levels, bins = np.unique(values, return_inverse=True)
        if len(levels) > 10:
            edges = np.unique(np.quantile(values, np.arange(1, 10) / 10))
            bins = np.digitize(values, edges)
        scores.append(mutual_info_score(y, bins))
    return np.array(scores)


def sale_price(x):
    return (x["Unit_Price"] * (1 - x["Discount_Percentage"] / 100)).round(2)
