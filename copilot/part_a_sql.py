"""Part A, tasks 6-7 -- run the seven SQL queries, then cross-check Q2 in pandas and explore order_amount.

Run: python -m copilot.part_a_sql
"""
import re
import sqlite3

import pandas as pd

from copilot.paths import DB, SQL, Report


def load_queries(path=SQL):
    """Split queries.sql on its '-- name:' lines -> {name: sql}."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_sql(con, rep):
    q = {name: pd.read_sql_query(sql, con) for name, sql in load_queries().items()}
    rep("## Task 6 - SQL results (pd.read_sql_query against data/instamart.db)")

    r = q["Q1_where_then_having"]
    rep(f"Q1 WHERE then HAVING: {len(r)} customers (largest: {r.iloc[0].customer_id} INR {r.iloc[0].total_above_500:,.2f})")

    r = q["Q2_three_table_join"]
    top = r.iloc[0]
    rep(f"Q2 three-table join: {len(r)} rows; revenues add up to INR {r.revenue.sum():,.2f}; "
        f"largest cell {top.city} / {top.category} INR {top.revenue:,.2f}")

    r = q["Q3_scalar_subquery"]
    rep(f"Q3 scalar subquery: {len(r)} products above the average catalogue price "
        f"(average INR {pd.read_sql_query('SELECT AVG(price_inr) a FROM products', con).a[0]:,.2f})")

    r = q["Q4_left_join"]
    rep(f"Q4 LEFT JOIN: {len(r)} customers have no completed order")

    r = q["Q5_cte"]
    rep(f"Q5 CTE: {len(r)} customers above the average completed spend of INR {r.avg_spend.iloc[0]:,.2f}")

    r = q["Q6_ranking_window"]
    top = r[r.row_num == 1]
    rep("Q6 window ranking - top customer per city: " +
        ", ".join(f"{t.city} {t.customer_id} ({t.total_spend:,.2f})" for t in top.itertuples()))
    diff = r[r.rnk != r.row_num]
    rep(f"   rows where RANK() differs from ROW_NUMBER(): {len(diff)}")
    for t in diff.itertuples():
        rep(f"     {t.city} {t.customer_id} spend {t.total_spend:,.2f} rank {t.rnk} dense_rank {t.dense_rnk} row_number {t.row_num}")

    r = q["Q7_running_total"]
    rep(f"Q7 running total: {len(r)} daily rows; first day {r.day.iloc[0]}, last day {r.day.iloc[-1]}, "
        f"last cumulative INR {r.cumulative_revenue.iloc[-1]:,.2f}")
    return q


def pandas_cross_check(con, q2, rep):
    rep("\n## Task 7a - pandas cross-check of Q2")
    df = pd.read_sql_query(
        "SELECT o.order_amount, c.city, p.category FROM orders o "
        "JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed'", con)
    pivot = pd.pivot_table(df, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    rep(pivot.round(2).to_string())
    grand = round(pivot.loc["Total", "Total"], 2)
    rep(f"pivot grand total INR {grand:,.2f} vs SQL Q2 total INR {round(q2.revenue.sum(), 2):,.2f} -> "
        f"{'equal' if abs(grand - q2.revenue.sum()) < 0.005 else 'DIFFERENT'}")
    named = df.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                   orders=("order_amount", "count")).sort_values("total", ascending=False)
    rep("\nper-city completed order_amount (groupby + named aggregation):")
    rep(named.round(2).to_string())


def explore(con, rep):
    rep("\n## Task 7b - exploring completed order_amount")
    o = pd.read_sql_query("SELECT order_amount, discount_pct, quantity, delivery_minutes "
                          "FROM orders WHERE status='completed'", con)
    a = o["order_amount"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = [("count", f"{len(a):,}"), ("mean", f"{a.mean():.2f}"), ("median", f"{a.median():.2f}"),
             ("skewness", f"{a.skew():.2f}"), ("min", f"{a.min():.2f}"), ("Q1 (25%)", f"{q1:.2f}"),
             ("Q3 (75%)", f"{q3:.2f}"), ("max", f"{a.max():.2f}"), ("IQR", f"{iqr:.2f}"),
             ("lower fence Q1-1.5xIQR", f"{lo:.2f}"), ("upper fence Q3+1.5xIQR", f"{hi:.2f}"),
             ("outliers below lower fence", f"{int((a < lo).sum())}"),
             ("outliers above upper fence", f"{int((a > hi).sum())}")]
    for k, v in stats:
        rep(f"{k:<28}{v:>12}")
    corr = o.corr()
    rep("\ncorrelation matrix (completed orders):")
    rep(corr.round(3).to_string())
    pairs = corr.where(~(corr.abs() >= 0.9999)).stack()
    pairs = pairs[[a_ < b_ for a_, b_ in pairs.index]]
    order = pairs.abs().sort_values(ascending=False)
    rep("pairs by |r|: " + ", ".join(f"{i[0]}~{i[1]} {pairs[i]:+.3f}" for i in order.index))
    above = o.loc[a > hi]
    cats = pd.read_sql_query("SELECT o.order_amount, p.category FROM orders o JOIN products p USING(product_id) "
                             "WHERE o.status='completed'", con)
    rep("orders above the upper fence by category: " + ", ".join(
        f"{k} {v}" for k, v in cats.loc[cats.order_amount > hi, "category"].value_counts().items()))
    rep("mean / median completed order_amount by category: " + ", ".join(
        f"{k} {m:.2f} / {d:.2f}" for k, (m, d) in
        cats.groupby("category")["order_amount"].agg(["mean", "median"]).iterrows()))
    rep(f"orders above the upper fence: mean quantity {above.quantity.mean():.2f} vs {o.quantity.mean():.2f} overall; "
        f"amount range INR {above.order_amount.min():,.2f}-{above.order_amount.max():,.2f}")


def main():
    rep = Report("part_a_sql")
    with sqlite3.connect(DB) as con:
        q = run_sql(con, rep)
        pandas_cross_check(con, q["Q2_three_table_join"], rep)
        explore(con, rep)
    rep.save()


if __name__ == "__main__":
    main()
