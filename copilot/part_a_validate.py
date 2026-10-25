"""Part A, task 5 -- assert the reference checkpoints from tasks 3 and 4 against the stored clean data.

Run: python -m copilot.part_a_validate      (prints ALL CHECKPOINTS PASSED when every check holds)
Reads both the clean CSVs and data/instamart.db, so a mismatch between the two is also caught.
"""
import os
import sqlite3
import sys

import pandas as pd

from copilot.paths import CLEAN, DB


def checks():
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    with sqlite3.connect(DB) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        db_rows = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        db_total = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]
        bad_ts = con.execute("SELECT COUNT(*) FROM orders WHERE order_ts NOT GLOB "
                             "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'").fetchone()[0]
    completed = orders["status"] == "completed"
    cancelled = orders["status"] == "cancelled"
    yield "task 3a: exactly six cities with counts Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184", \
        customers["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                                       "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    yield "task 3b: genders female 610, male 560, unknown 30", \
        customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
    yield "task 3c: 0 missing ages", customers["age"].isna().sum() == 0
    yield "task 3c: every age between 18 and 60", customers["age"].between(18, 60).all()
    yield "task 3d: signup_date parses as a date", pd.to_datetime(customers["signup_date"], format="%Y-%m-%d").notna().all()
    yield "task 4: 12,309 rows, one per order_id", len(orders) == 12309 and orders["order_id"].is_unique
    yield "task 4: 11,110 completed and 1,199 cancelled", completed.sum() == 11110 and cancelled.sum() == 1199
    yield "task 4: no pending rows", (orders["status"] == "pending").sum() == 0
    yield "task 4: 0 missing order_amount", orders["order_amount"].isna().sum() == 0
    yield "task 4: 0 missing delivery_minutes among completed orders", orders.loc[completed, "delivery_minutes"].isna().sum() == 0
    yield "task 4: 1,199 missing delivery_minutes among cancelled orders", orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199
    yield "task 4: no negative delivery_minutes", (orders["delivery_minutes"] < 0).sum() == 0
    yield "task 4: completed order_amount sums to INR 2,998,668.80", round(orders.loc[completed, "order_amount"].sum(), 2) == 2998668.80
    yield "task 5: instamart.db holds customers, products, orders", {"customers", "products", "orders"} <= tables
    yield "task 5: database orders table matches the CSV (12,309 rows, INR 2,998,668.80)", db_rows == 12309 and db_total == 2998668.80
    yield "task 5: order_ts stored as text YYYY-MM-DD HH:MM:SS", bad_ts == 0


def main():
    failed = 0
    for label, ok in checks():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        failed += not ok
    if failed:
        print(f"{failed} CHECKPOINT(S) FAILED")
        sys.exit(1)
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
