"""Part A, tasks 3-5 -- clean customers and orders with explicit rules, then store CSVs and SQLite.

Run: python -m copilot.part_a_clean
Every rule prints how many rows or values it changed; those counts are copied into the README.
"""
import os
import sqlite3

import pandas as pd

from copilot.paths import CLEAN, DB, RAW, TS_FORMAT, Report, ensure_dirs

GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}


def read_raw(name):
    return pd.read_csv(os.path.join(RAW, name), keep_default_na=False, na_values=[""])


def clean_customers(rep):
    c = read_raw("customers.csv")
    rep("## customers.csv")

    # (a) city: strip spaces, title case -> exactly six cities
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    rep(f"(a) city strip + title case: {int((before != c['city']).sum())} values changed; "
        f"{before.nunique()} spellings -> {c['city'].nunique()} cities")

    # (b) gender: every spelling -> male / female / unknown (blank -> unknown)
    before = c["gender"].copy()
    c["gender"] = c["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    changed = int((before.fillna("") != c["gender"]).sum())
    rep(f"(b) gender mapped to male/female/unknown: {changed} values changed "
        f"({int(before.isna().sum())} blanks -> unknown)")

    # (c) ages outside 10-100 are impossible -> missing; then fill with the median age of the customer's city
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    impossible = (c["age"] < 10) | (c["age"] > 100)
    rep(f"(c) ages outside 10-100 set to missing: {int(impossible.sum())} "
        f"({', '.join(str(int(a)) for a in c.loc[impossible, 'age'])})")
    c.loc[impossible, "age"] = float("nan")
    missing = int(c["age"].isna().sum())
    city_median = c.groupby("city")["age"].median()
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    rep(f"    missing ages filled with the city median (groupby + transform): {missing} values filled")
    rep("    city medians used: " + ", ".join(f"{k} {v:g}" for k, v in city_median.items()))
    if (c["age"] == c["age"].round()).all():
        c["age"] = c["age"].astype(int)

    # (d) signup_date -> datetime
    c["signup_date"] = pd.to_datetime(c["signup_date"], format="%Y-%m-%d")
    rep(f"(d) signup_date converted to datetime: {len(c)} values")
    return c


def clean_orders(products, rep):
    o = read_raw("orders.csv")
    rep("\n## orders.csv")
    rep(f"start: {len(o):,} rows")

    # i. order_ts -> datetime
    o["order_ts"] = pd.to_datetime(o["order_ts"], format=TS_FORMAT)
    rep(f"i.   order_ts converted to datetime: {len(o):,} values")

    # ii. exact duplicate rows
    n = len(o)
    o = o.drop_duplicates()
    rep(f"ii.  exact duplicate rows removed: {n - len(o):,} -> {len(o):,} rows")

    # iii. one record per order_id: keep the latest order_ts (drops the earlier 'pending' record)
    n = len(o)
    ordered = o.sort_values(["order_id", "order_ts"])
    dropped = ordered[ordered.duplicated("order_id", keep="last")]
    o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").reset_index(drop=True)
    rep(f"iii. earlier records of repeated order_ids removed: {n - len(o):,} -> {len(o):,} rows "
        f"(statuses of removed rows: "
        f"{', '.join(f'{k} {v}' for k, v in dropped['status'].value_counts().items())})")

    # iv. order_amount -> numeric; refill unparseable amounts from the product price
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    missing = o["order_amount"].isna()
    price = o["product_id"].map(products.set_index("product_id")["price_inr"])
    rebuilt = (price * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
    o.loc[missing, "order_amount"] = rebuilt[missing]
    rep(f"iv.  unparseable order_amount -> missing -> refilled from price x quantity x (1 - discount/100): "
        f"{int(missing.sum())} values")

    # v. delivery_minutes: negatives -> missing; completed orders get the category median, cancelled stay missing
    o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = float("nan")
    rep(f"v.   negative delivery_minutes set to missing: {int(neg.sum())}")
    category = o["product_id"].map(products.set_index("product_id")["category"])
    completed = o["status"] == "completed"
    median_by_cat = o["delivery_minutes"].where(completed).groupby(category).transform("median")
    to_fill = completed & o["delivery_minutes"].isna()
    o.loc[to_fill, "delivery_minutes"] = median_by_cat[to_fill]
    rep(f"     completed orders with missing delivery_minutes filled with the category median among completed "
        f"orders (groupby + transform): {int(to_fill.sum())} values")
    rep("     category medians used: " + ", ".join(f"{k} {v:g}" for k, v in
                                                     o.loc[completed & ~to_fill, 'delivery_minutes']
                                                     .groupby(category[completed & ~to_fill]).median().items()))
    rep(f"     cancelled orders left with missing delivery_minutes: "
        f"{int((o['status'] == 'cancelled').sum())} (never delivered, so no time is invented)")
    return o


def summary(customers, orders, rep):
    rep("\n## after cleaning")
    rep("cities: " + ", ".join(f"{k} {v}" for k, v in customers["city"].value_counts().items()))
    rep("genders: " + ", ".join(f"{k} {v}" for k, v in customers["gender"].value_counts().items()))
    rep(f"missing ages: {int(customers['age'].isna().sum())}; age range {customers['age'].min()}-{customers['age'].max()}")
    rep(f"orders: {len(orders):,} rows, {orders['order_id'].nunique():,} distinct order_id")
    rep("statuses: " + ", ".join(f"{k} {v:,}" for k, v in orders["status"].value_counts().items()))
    rep(f"missing order_amount: {int(orders['order_amount'].isna().sum())}")
    by = orders.groupby("status")["delivery_minutes"].apply(lambda s: int(s.isna().sum()))
    rep("missing delivery_minutes by status: " + ", ".join(f"{k} {v:,}" for k, v in by.items()))
    total = orders.loc[orders["status"] == "completed", "order_amount"].sum()
    rep(f"sum of order_amount over completed orders: INR {total:,.2f}")


def store(customers, products, orders, rep):
    ensure_dirs()
    c = customers.copy()
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    o = orders.copy()
    o["order_ts"] = o["order_ts"].dt.strftime(TS_FORMAT)
    for df, name in ((c, "customers"), (products, "products"), (o, "orders")):
        df.to_csv(os.path.join(CLEAN, f"{name}.csv"), index=False)
    if os.path.exists(DB):
        os.remove(DB)
    with sqlite3.connect(DB) as con:
        for df, name in ((c, "customers"), (products, "products"), (o, "orders")):
            df.to_sql(name, con, index=False)
        con.execute("CREATE UNIQUE INDEX idx_orders_id ON orders(order_id)")
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
    rep(f"\nwrote data/clean/{{customers,products,orders}}.csv and data/instamart.db tables {counts}")


def main():
    rep = Report("part_a_clean")
    rep("# Part A - cleaning rules and the rows/values each one changed")
    products = read_raw("products.csv")
    customers = clean_customers(rep)
    orders = clean_orders(products, rep)
    summary(customers, orders, rep)
    store(customers, products, orders, rep)
    rep.save()


if __name__ == "__main__":
    main()
