"""Part A, task 2 -- audit the raw exports before touching them.

Run: python -m copilot.part_a_audit
Only blank cells count as missing: the raw files are read with keep_default_na=False so that text such
as "N/A" in order_amount stays visible as an unparseable value instead of silently becoming NaN.
"""
import os

import pandas as pd

from copilot.paths import RAW, Report


def load_raw(name):
    return pd.read_csv(os.path.join(RAW, name), keep_default_na=False, na_values=[""])


def profile(df, title, rep):
    rep(f"\n## {title}: shape {df.shape[0]} rows x {df.shape[1]} columns")
    rep(f"{'column':<18}{'dtype':<10}{'missing':>8}{'missing %':>11}")
    for col in df.columns:
        n = int(df[col].isna().sum())
        rep(f"{col:<18}{str(df[col].dtype):<10}{n:>8}{100 * n / len(df):>10.2f}%")


def main():
    rep = Report("part_a_audit")
    orders, customers, products = load_raw("orders.csv"), load_raw("customers.csv"), load_raw("products.csv")
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)

    rep("# Part A - raw data audit")
    for df, title in ((orders, "orders.csv"), (customers, "customers.csv"), (products, "products.csv")):
        profile(df, title, rep)
    rep(f"\n(tickets.csv read with keep_default_na=False: {len(tickets)} rows, "
        f"{int((tickets['text'].str.strip() == '').sum())} empty texts kept as empty strings)")

    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"]
    figures = [
        ("raw orders.csv rows", len(orders)),
        ("exact duplicate rows", int(orders.duplicated().sum())),
        ("rows with status 'pending'", int((orders["status"] == "pending").sum())),
        ("distinct order_id values", orders["order_id"].nunique()),
        ("order_amount entries not convertible to a number", int(amount_num.isna().sum())),
        ("negative delivery_minutes", int((delivery < 0).sum())),
        ("blank delivery_minutes", int(orders["delivery_minutes"].isna().sum())),
        ("blank age (customers)", int(customers["age"].isna().sum())),
        ("age outside 10-100 (customers)", int(((age < 10) | (age > 100)).sum())),
        ("distinct raw city spellings", customers["city"].nunique()),
        ("distinct non-blank raw gender spellings", gender.dropna().nunique()),
        ("blank gender", int(gender.isna().sum())),
    ]
    rep("\n## Audit figures")
    for label, value in figures:
        rep(f"{label:<52}{value:>8,}")
    bad = orders.loc[amount_num.isna(), "order_amount"].value_counts()
    rep("unparseable order_amount values: " + ", ".join(f"'{k}' x{v}" for k, v in bad.items()))
    rep("raw city spellings: " + ", ".join(repr(c) for c in sorted(customers["city"].unique())))
    rep("raw gender spellings: " + ", ".join(sorted(gender.dropna().unique())))
    rep.save()


if __name__ == "__main__":
    main()
