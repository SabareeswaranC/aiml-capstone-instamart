"""Part B, tasks 4-8 -- lapse label, leakage-free features, tuned models, class balance, persistence.

Run: python -m copilot.part_b_lapse      (writes outputs/lapse_model.joblib and outputs/retention_list.csv)

Leakage rules enforced here:
  * features use only orders dated before the cutoff 2025-09-01; September orders only build the label;
  * the segment labels from part_b_segments are NOT features (they were computed on the full window);
  * every encoder and scaler lives inside a Pipeline, so it is fitted on training rows only
    (and on training folds only inside GridSearchCV); over-sampling touches the training set only.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from copilot.paths import CLEAN, OUT, SEED, Report, ensure_dirs

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")          # September = 2025-09-01 .. 2025-09-30
NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate",
           "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]
MODEL_PATH = os.path.join(OUT, "lapse_model.joblib")


def build_dataset():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])

    before = orders[orders["order_ts"] < CUTOFF]                       # the only rows features may use
    done = before[before["status"] == "completed"]
    cohort = done["customer_id"].unique()

    september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < LABEL_END)
                       & (orders["status"] == "completed")]
    g = done.groupby("customer_id")
    fav = (done.groupby(["customer_id", "category"]).size().reset_index(name="n")
           .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
           .drop_duplicates("customer_id").set_index("customer_id")["category"])
    feats = pd.DataFrame({
        "recency": (CUTOFF - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum(),
        "avg_discount_pct": g["discount_pct"].mean(),
        "avg_delivery_minutes": g["delivery_minutes"].mean(),
        "fav_category": fav,
    })
    status = before[before["customer_id"].isin(cohort)].groupby("customer_id")["status"]
    feats["cancel_rate"] = status.apply(lambda s: (s == "cancelled").mean())
    cust = customers.set_index("customer_id")
    feats = feats.join(cust[["city", "age", "gender"]])
    feats["tenure_days"] = (CUTOFF - cust.loc[feats.index, "signup_date"]).dt.days
    feats["lapsed"] = (~feats.index.isin(september["customer_id"])).astype(int)
    return feats.reset_index().rename(columns={"index": "customer_id"})


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}


def show_eval(rep, name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = metrics(y_true, y_pred)
    rep(f"{name}: confusion matrix [[TN {tn}, FP {fp}], [FN {fn}, TP {tp}]]; lapsed-class precision "
        f"{m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}; accuracy {m['accuracy']:.3f}")
    return m


def main():
    ensure_dirs()
    rep = Report("part_b_lapse")
    data = build_dataset()
    X, y = data[NUMERIC + NOMINAL], data["lapsed"]

    rep("## Task 4 - lapse problem (cutoff 2025-09-01)")
    rep(f"cohort (>=1 completed order before the cutoff): {len(data):,} customers; lapsed {int(y.sum())} "
        f"({100 * y.mean():.2f}%), not lapsed {int((1 - y).sum())}")
    latest = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    rep(f"latest order date used by any feature: {latest[latest.order_ts < CUTOFF].order_ts.max()} (< cutoff)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    ids_te = data.loc[X_te.index, "customer_id"]
    rep(f"train {len(X_tr)} rows ({int(y_tr.sum())} lapsed), test {len(X_te)} rows ({int(y_te.sum())} lapsed)")
    base = show_eval(rep, "baseline (always 'not lapsed') on test", y_te, np.zeros(len(y_te), dtype=int))

    rep("\n## Task 5 - preprocessing without leakage")
    probe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    scaler = probe.named_steps["prep"].named_transformers_["num"]
    rep(f"learned mean of frequency in the fitted scaler (training rows): {scaler.mean_[NUMERIC.index('frequency')]:.4f}")
    rep(f"mean of frequency over the full cohort: {X['frequency'].mean():.4f}")
    rep("they differ because the scaler only ever saw the 924 training rows; the 232 test rows never "
        "influenced its mean or standard deviation.")
    enc = probe.named_steps["prep"].named_transformers_["cat"]
    rep("one-hot columns (k-1 per variable, fitted on train): " + ", ".join(enc.get_feature_names_out(NOMINAL)))
    missing = int(X.isna().sum().sum())
    assert missing == 0, "a feature column contains missing values"
    rep(f"missing values across all feature columns: {missing} (asserted in code)")

    rep("\n## Task 6 - tuning with 5-fold GridSearchCV (scoring = F1 of the lapsed class)")
    lr = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)),
                      {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
    rf = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED)),
                      {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
                      cv=5, scoring="f1").fit(X_tr, y_tr)
    rep(f"logistic regression: best {lr.best_params_}, best CV F1 {lr.best_score_:.3f} "
        f"({len(lr.cv_results_['params']) * 5} fits)")
    rep(f"random forest: best {rf.best_params_}, best CV F1 {rf.best_score_:.3f} "
        f"({len(rf.cv_results_['params']) * 5} fits)")
    rep("test set (used only now, after tuning):")
    m_lr = show_eval(rep, "  tuned logistic regression", y_te, lr.predict(X_te))
    m_rf = show_eval(rep, "  tuned random forest", y_te, rf.predict(X_te))
    rep(f"  baseline F1 {base['f1']:.3f} -> logistic {m_lr['f1']:.3f}, forest {m_rf['f1']:.3f}")
    best = rf if m_rf["f1"] >= m_lr["f1"] else lr
    chosen = "random forest" if best is rf else "logistic regression"
    rep(f"selected model: {chosen}")

    rep("\n## Task 7 - class-balance experiment (over-sample the training set only)")
    train = X_tr.assign(lapsed=y_tr.values)
    minority, majority = train[train.lapsed == 1], train[train.lapsed == 0]
    extra = minority.sample(n=len(majority) - len(minority), replace=True, random_state=SEED)
    balanced = pd.concat([train, extra])
    rep(f"training rows {len(train)} -> {len(balanced)}; class counts {balanced.lapsed.value_counts().to_dict()} "
        f"({len(extra)} lapsed duplicates added)")
    lr_bal = make_pipeline(LogisticRegression(max_iter=1000, C=lr.best_params_["model__C"]))
    lr_bal.fit(balanced[NUMERIC + NOMINAL], balanced["lapsed"])
    m_bal = show_eval(rep, "  balanced logistic regression on the original test set", y_te, lr_bal.predict(X_te))
    rep(f"  un-balanced -> balanced: precision {m_lr['precision']:.3f} -> {m_bal['precision']:.3f}, "
        f"recall {m_lr['recall']:.3f} -> {m_bal['recall']:.3f}, F1 {m_lr['f1']:.3f} -> {m_bal['f1']:.3f}")

    rep("\n## Task 8a - interpretation")
    rf_model = rf.best_estimator_
    names = rf_model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf_model.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    rep("random forest top-5 feature importances: " +
        ", ".join(f"{n.split('__', 1)[1]} {v:.3f}" for n, v in imp.head(5).items()))
    coef = pd.Series(lr.best_estimator_.named_steps["model"].coef_[0],
                     index=lr.best_estimator_.named_steps["prep"].get_feature_names_out())
    for f in ("num__recency", "num__frequency", "num__cancel_rate", "num__monetary", "num__avg_delivery_minutes"):
        rep(f"logistic coefficient {f.split('__', 1)[1]}: {coef[f]:+.3f} (per standard deviation)")

    rep("\n## Task 8b - persistence")
    final = best.best_estimator_
    joblib.dump(final, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    same = np.array_equal(final.predict(X_te), reloaded.predict(X_te)) and \
        np.allclose(final.predict_proba(X_te), reloaded.predict_proba(X_te))
    assert same
    rep(f"saved {chosen} to outputs/lapse_model.joblib; reloaded model reproduces all {len(X_te)} test predictions "
        f"and probabilities exactly: {same}")

    rep("\n## Task 8c - retention list (Premium test-set customers ranked by lapse probability)")
    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
    scored = pd.DataFrame({"customer_id": ids_te.values,
                           "lapse_probability": reloaded.predict_proba(X_te)[:, 1].round(4),
                           "actually_lapsed": y_te.values})
    premium = scored[scored["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)]
    top = premium.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    top.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
    rep(f"Premium customers in the test set: {len(premium)}; wrote top {len(top)} to outputs/retention_list.csv")
    rep(top.to_string(index=False))
    rep(f"of these {len(top)}, {int(top.actually_lapsed.sum())} truly lapsed")
    rep("note: the Premium flag comes from whole-window RFM that includes September, so it is a post-hoc label "
        "and this list is illustrative, not a leakage-free evaluation.")
    rep.save()


if __name__ == "__main__":
    main()
