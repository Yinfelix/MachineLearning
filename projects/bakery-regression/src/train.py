"""从原始订单到验证、选模及独立测试。运行 python -m src.train。"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .config import AUDIT_OUTPUT_PATH, FEATURE_COLUMNS
from .data_audit import audit_dataset, write_audit_report
from .dataset import load_dataset, split_by_date
from .features import GROUPS, sale_price
from .models import GRIDS, MODELS, SEED, ProductMedian, metrics, pipeline


OUT = AUDIT_OUTPUT_PATH.parent
INPUTS = ["Date", *FEATURE_COLUMNS]
QTY, BILL = "Quantity", "Total_Bill"


def date_folds(dates):
    """同一天的订单进入同一折。每一折都用较早日期预测较晚日期。"""
    days = dates.sort_values().unique()
    folds = []
    for past, future in TimeSeriesSplit(n_splits=3).split(days):
        folds.append((np.flatnonzero(dates.isin(days[past])), np.flatnonzero(dates.isin(days[future]))))
    return folds


def compare_models(train, valid, target, folds, scores, cv_rows):
    x, y = train[INPUTS], train[target]
    models, params = {}, {}
    for name in MODELS:
        base = pipeline(name).fit(x, y)
        pred = base.predict(valid[INPUTS])
        label = name + "_initial" if name in GRIDS else name
        scores[label] = metrics(valid[target], pred)
        if name not in GRIDS:
            models[name] = base
            continue

        search = GridSearchCV(pipeline(name), GRIDS[name], cv=folds, scoring="neg_mean_absolute_error")
        search.fit(x, y)
        models[name] = search.best_estimator_
        params[name] = search.best_params_
        scores[name] = metrics(valid[target], models[name].predict(valid[INPUTS]))
        for i, values in enumerate(search.cv_results_["params"]):
            cv_rows.append({
                "target": target, "model": name,
                **{key.removeprefix("model__"): value for key, value in values.items()},
                **{f"fold{j + 1}_MAE": -search.cv_results_[f"split{j}_test_score"][i] for j in range(3)},
                "mean_MAE": -search.cv_results_["mean_test_score"][i], "chosen": i == search.best_index_,
            })
    return models, params


def study_features(train, valid, models, params, scores):
    """在验证集比较时间、商品交互和互信息选列。"""
    for name in ("ridge", "forest"):
        variants = [("calendar", None), ("cross", None), ("cross", 10), ("cross", 20), ("cross", 40)]
        for level, k in variants:
            label = f"{name}_{level}" if k is None else f"{name}_{level}_k{k}"
            model = pipeline(name, level, k).set_params(**params[name])
            model.fit(train[INPUTS], train[QTY])
            models[label] = model
            scores[label] = metrics(valid[QTY], model.predict(valid[INPUTS]))

    ranked = models["ridge_cross_k40"]
    names = ranked["prep"].get_feature_names_out()
    pd.DataFrame({"feature": names, "mutual_information": ranked["select"].scores_}).sort_values(
        "mutual_information", ascending=False
    ).to_csv(OUT / "feature_scores.csv", index=False)


def study_groups(train, valid, name, model, params):
    """分别重新训练去掉一组字段的模型，并测量打乱该组后的误差变化。"""
    x, y = valid[INPUTS], valid[QTY]
    base = metrics(y, model.predict(x))["MAE"]
    rng = np.random.default_rng(SEED)
    orders = [rng.permutation(len(x)) for _ in range(5)]
    rows = []
    for group, cols in GROUPS.items():
        reduced = pipeline(name, drop=cols).set_params(**params)
        reduced.fit(train[INPUTS], train[QTY])
        removed = metrics(y, reduced.predict(x))["MAE"]
        shuffled = []
        for order in orders:
            mixed = x.copy()
            mixed[cols] = x[cols].iloc[order].to_numpy()
            shuffled.append(metrics(y, model.predict(mixed))["MAE"])
        rows.append({"group": group, "model": name, "full_MAE": base, "removed_MAE": removed,
                     "removed_delta": removed - base, "shuffled_delta": np.mean(shuffled) - base})
    pd.DataFrame(rows).to_csv(OUT / "groups.csv", index=False)


def select_models(train, valid):
    folds = date_folds(train["Date"])
    scores = {QTY: {}, BILL: {}}
    cv_rows = []
    qty, params = compare_models(train, valid, QTY, folds, scores[QTY], cv_rows)
    qty["product_median"] = ProductMedian().fit(train[INPUTS], train[QTY])
    scores[QTY]["product_median"] = metrics(valid[QTY], qty["product_median"].predict(valid[INPUTS]))
    print("Comparing quantity features and field groups ...", flush=True)
    study_features(train, valid, qty, params, scores[QTY])
    group_model = min(GRIDS, key=lambda name: scores[QTY][name]["MAE"])
    study_groups(train, valid, group_model, qty[group_model], params[group_model])
    best_qty = min(qty, key=lambda name: scores[QTY][name]["MAE"])

    print("Comparing line amount models ...", flush=True)
    bill, _ = compare_models(train, valid, BILL, folds, scores[BILL], cv_rows)
    indirect = sale_price(valid) * qty[best_qty].predict(valid[INPUTS])
    scores[BILL]["price_x_quantity"] = metrics(valid[BILL], indirect)
    best_bill = min([*bill, "price_x_quantity"], key=lambda name: scores[BILL][name]["MAE"])
    validation = [{"target": target, "model": name, **row}
                  for target, table in scores.items() for name, row in table.items()]
    pd.DataFrame(validation).to_csv(OUT / "validation.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "cv.csv", index=False)
    winners = {QTY: best_qty, BILL: best_bill}
    (OUT / "selection.json").write_text(json.dumps(winners, indent=2) + "\n", encoding="utf-8")

    # 统一评估四个基础方案、商品中位数和验证集选出的件数方案。
    qty_final = {key: qty[key] for key in [*MODELS, "product_median", best_qty]}
    return {QTY: qty_final, BILL: bill}, winners


def evaluate(models, winners, history, test):
    pred = test[["Date", "Order_ID", "Product", "Category", QTY, BILL]].copy()
    rows = []
    for target, candidates in models.items():
        for name, model in candidates.items():
            fitted = clone(model).fit(history[INPUTS], history[target])
            values = fitted.predict(test[INPUTS])
            pred[f"{target}_{name}"] = values
            rows.append({"target": target, "model": name, **metrics(test[target], values)})

    indirect = sale_price(test) * pred[f"{QTY}_{winners[QTY]}"]
    pred[f"{BILL}_price_x_quantity"] = indirect
    rows.append({"target": BILL, "model": "price_x_quantity", **metrics(test[BILL], indirect)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "test.csv", index=False)
    pred.to_csv(OUT / "predictions.csv", index=False)
    return results, pred


def amount_bands(train, pred):
    """按训练金额的第 90 百分位分组，比较两种方法在测试集的误差。"""
    cutoff = train[BILL].quantile(0.90)
    rows = []
    for high, group in pred.groupby(pred[BILL] > cutoff):
        for name in ("forest", "price_x_quantity"):
            rows.append({"above_cutoff": high, "cutoff": cutoff, "rows": len(group), "model": name,
                         **metrics(group[BILL], group[f"{BILL}_{name}"])})
    pd.DataFrame(rows).to_csv(OUT / "amount_bands.csv", index=False)


def main():
    from .plots import draw_figures

    data, cleaning = load_dataset()
    write_audit_report(audit_dataset(data, cleaning))
    parts = split_by_date(data)
    print("Selecting models with three expanding date folds ...", flush=True)
    models, winners = select_models(parts.train, parts.validation)
    history = pd.concat([parts.train, parts.validation])
    results, pred = evaluate(models, winners, history, parts.test)
    amount_bands(parts.train, pred)
    draw_figures(parts.train, results, pred, winners)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
