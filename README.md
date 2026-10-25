# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that turns Instamart's messy order exports into a trusted analytical database, segments customers and predicts who will lapse, triages support tickets safely, and answers policy questions and refund requests with a retrieval-grounded assistant and a tool-calling agent. All data is synthetic and fictional, every amount is in Indian Rupees (INR), and every language-model call goes through the supplied offline `mock_llm.py`: no API key, account, network call or paid service is used anywhere.

Each Part consumes what the previous Part produced:

| Part | Script(s) | Reads | Writes |
|---|---|---|---|
| A. Data foundation | `copilot/part_a_audit.py`, `part_a_clean.py`, `part_a_validate.py`, `part_a_sql.py`, `sql/queries.sql` | `data/raw/*.csv` | `data/clean/*.csv`, `data/instamart.db` |
| B. Segments and lapse model | `copilot/part_b_segments.py`, `part_b_lapse.py` | `data/clean/orders.csv`, `data/clean/customers.csv` | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C. Ticket triage | `copilot/part_c_triage.py` | `data/raw/tickets.csv`, `outputs/customer_segments.csv` | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D. Policy RAG and refund agent | `copilot/part_d_rag.py`, `part_d_agent.py` | `data/policies/*.txt`, `data/instamart.db` | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Each script prints its results and also saves them to `outputs/reports/<script>.txt`. Every measured number below comes from those reports; the few differences and ratios quoted in the text are computed from them. `generate_data.py` and `mock_llm.py` are committed unmodified. Fixed seeds (`random_state=42`, `random.Random(42)`) make every run identical; a clean clone was reproduced on Python 3.11.13 and 3.13.7 with byte-identical reports.

## Reproduce

Run from the repository root with Python 3.11 or newer (macOS/Linux shown; on Windows use `.venv\Scripts\activate` in step 2).

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python -m copilot.part_a_audit`
6. `python -m copilot.part_a_clean`
7. `python -m copilot.part_a_validate` (last line: `ALL CHECKPOINTS PASSED`)
8. `python -m copilot.part_a_sql`
9. `python -m copilot.part_b_segments`
10. `python -m copilot.part_b_lapse`
11. `python -m copilot.part_c_triage`
12. `python -m copilot.part_d_rag`
13. `python -m copilot.part_d_agent`

The whole sequence takes well under a minute. Steps 5 to 13 must run in this order because each Part reads the files written by the earlier ones. Step 13 empties `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and `state/` before it starts, so re-running it never piles up old runs.

## Part A

### Setup

The repository lives on the single branch `main`. A virtual environment created with `python -m venv` holds pandas, numpy, scikit-learn and joblib, installed with `pip`. `requirements.txt` is the output of `pip freeze`, taken from a Python 3.11 environment so that every pinned version also installs on 3.11. `.gitignore` lists `.env`, `.venv/` (plus `venv/` and `env/`) and `__pycache__/`, and also excludes chart images so none can be committed by accident.

### Audit of the raw data (`part_a_audit`)

The raw files are read with `keep_default_na=False, na_values=[""]`. Only truly blank cells count as missing, so text such as `N/A` in `order_amount` stays visible as an unparseable value instead of silently becoming NaN. `tickets.csv` is always read with `keep_default_na=False`, which keeps its 4 empty texts as empty strings.

Shapes: `orders.csv` 13,084 rows × 9 columns, `customers.csv` 1,200 × 5, `products.csv` 40 × 4.

| Table | Column | dtype | Missing | Missing % |
|---|---|---|---:|---:|
| orders | order_id | str | 0 | 0.00% |
| orders | customer_id | str | 0 | 0.00% |
| orders | product_id | str | 0 | 0.00% |
| orders | quantity | int64 | 0 | 0.00% |
| orders | order_ts | str | 0 | 0.00% |
| orders | status | str | 0 | 0.00% |
| orders | discount_pct | int64 | 0 | 0.00% |
| orders | order_amount | str | 0 | 0.00% |
| orders | delivery_minutes | float64 | 2,541 | 19.42% |
| customers | customer_id | str | 0 | 0.00% |
| customers | city | str | 0 | 0.00% |
| customers | signup_date | str | 0 | 0.00% |
| customers | age | float64 | 61 | 5.08% |
| customers | gender | str | 30 | 2.50% |
| products | product_id | str | 0 | 0.00% |
| products | product_name | str | 0 | 0.00% |
| products | category | str | 0 | 0.00% |
| products | price_inr | int64 | 0 | 0.00% |

`order_amount` shows 0 missing because its 120 bad entries are text (`error`, `N/A`), not blanks; they are counted as non-numeric below.

| Audit figure | Value | Reference |
|---|---:|---:|
| raw `orders.csv` rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status `pending` | 656 | 656 |
| distinct `order_id` | 12,309 | 12,309 |
| `order_amount` not convertible to a number (`error` 67, `N/A` 53) | 120 | 120 |
| negative `delivery_minutes` | 106 | 106 |
| blank `delivery_minutes` | 2,541 | 2,541 |
| blank `age` | 61 | 61 |
| `age` outside 10–100 (280, −4, 0, 150, 999, 200) | 6 | 6 |
| distinct raw `city` spellings | 24 | 24 |
| distinct non-blank raw `gender` spellings (F, FEMALE, Female, M, MALE, Male, f, female, m, male) | 10 | 10 |
| blank `gender` | 30 | 30 |

### Cleaning rules and what each one changed (`part_a_clean`)

**customers**

| Rule | Values changed |
|---|---|
| (a) `city`: strip leading/trailing spaces, then title case | 476 values; 24 spellings → 6 cities |
| (b) `gender`: m/M/male/Male/MALE → `male`, f/F/female/Female/FEMALE → `female`, blank → `unknown` | 950 values (30 of them blanks → `unknown`) |
| (c) `age` outside 10–100 → missing | 6 values |
| (c) fill every missing age with the median age of the customer's city (`groupby("city")["age"].transform("median")`) | 67 values (61 blank + 6 impossible). Medians used: Bengaluru 42, Chennai 39, Delhi 43, Hyderabad 37.5, Mumbai 38.5, Pune 39 |
| (d) `signup_date` → datetime | 1,200 values |

**orders**, applied in this order:

| Rule | Rows / values changed |
|---|---|
| i. `order_ts` → datetime | 13,084 values |
| ii. remove exact duplicate rows | 119 rows removed → 12,965 |
| iii. keep only the record with the latest `order_ts` per `order_id` | 656 rows removed (all of them the earlier `pending` records) → 12,309 |
| iv. `order_amount` → numeric (unparseable → missing), then fill from `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 values refilled |
| v. negative `delivery_minutes` → missing | 105 values |
| v. completed orders: fill missing `delivery_minutes` with the median delivery time of the order's product category among completed orders (`groupby` + `transform`) | 774 values. Medians used: Dairy 25, Electronics 25, Grocery 25, Personal Care 25.5, Snacks 25 |
| v. cancelled orders: leave `delivery_minutes` missing, because a cancelled order was never delivered and imputing a time would invent data | 1,199 left missing |

Rules iv and v act on fewer values than the audit found because rules ii and iii run first. One of the 106 negative delivery times sat on an exact-duplicate row, which rule ii removed, leaving 105. Of the 120 unparseable amounts, 2 were on exact duplicates and 6 on the earlier `pending` records, leaving 112.

**After cleaning** (all match the reference): cities Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; genders female 610, male 560, unknown 30; 0 missing ages and every age between 18 and 60; 12,309 orders (one per `order_id`), 11,110 `completed` and 1,199 `cancelled`, no `pending`; 0 missing `order_amount`; 0 missing `delivery_minutes` among completed and 1,199 among cancelled orders; completed `order_amount` sums to **INR 2,998,668.80**.

### Store and validate

`part_a_clean` writes `data/clean/customers.csv`, `products.csv` and `orders.csv`, and the same three tables to the SQLite database `data/instamart.db` (via `sqlite3.connect` and `DataFrame.to_sql`). `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS` and `signup_date` as `YYYY-MM-DD`. `part_a_validate` asserts 16 checkpoints from tasks 3–5 against both the CSVs and the database (city counts, gender counts, no missing or out-of-range ages, 12,309 unique orders, 11,110/1,199 statuses, no pending rows, no missing amounts, the 0/1,199 missing delivery split, no negative delivery times, the INR 2,998,668.80 total in the CSV and in SQLite, the three tables, the timestamp format). It prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run by `part_a_sql` with `pd.read_sql_query`)

Only `completed` orders count as revenue or spend.

| Query | Technique | Result | Reference |
|---|---|---|---|
| Q1 | `WHERE order_amount > 500`, `GROUP BY customer_id`, `HAVING SUM > 8000` | 30 customers (largest C0693, INR 13,776.55) | 30 |
| Q2 | three-table join, revenue by city × category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 | same |
| Q3 | scalar subquery vs `AVG(price_inr)` (INR 289.32) | 11 products | 11 |
| Q4 | `LEFT JOIN … ON … AND status='completed' WHERE o.order_id IS NULL` | 25 customers with no completed order | 25 |
| Q5 | two CTEs (per-customer spend, average spend) | 550 customers above the average of INR 2,552.06 | 550; 2,552.06 |
| Q6 | `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` over `(PARTITION BY city ORDER BY total_spend DESC)` | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK() ≠ ROW_NUMBER() | same; 8 |
| Q7 | daily revenue with `SUM(revenue) OVER (ORDER BY day)` | 183 daily rows (2025-04-01 to 2025-09-30); last cumulative INR 2,998,668.80 | 183; 2,998,668.80 |

**Why RANK() and ROW_NUMBER() differ on 8 rows:** when customers in the same city have exactly the same total spend, `RANK()` gives them the same rank while `ROW_NUMBER()` still numbers them one by one, so the second and later members of each tie disagree. Here there are six tied pairs and one three-way tie (Pune, INR 285.00), which gives 6 + 2 = 8 rows.

### pandas cross-check and exploratory analysis

**(a)** `pd.pivot_table(..., index="city", columns="category", aggfunc="sum", margins=True)` reproduces Q2. Its grand total is INR 2,998,668.80, equal to the SQL total. Per city, `groupby("city").agg(total=..., mean=..., orders=...)` gives:

| City | Sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai | 463,629.15 | 272.72 | 1,700 |

**(b)** Completed `order_amount` (11,110 orders):

| Statistic | Value (INR) |
|---|---:|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| lower / upper fence (1.5 × IQR) | −283.50 / 645.30 |
| outliers below / above | 0 / 897 |

Correlations over completed orders: order_amount–discount_pct −0.095 (largest in absolute value), order_amount–quantity +0.087, and every pair involving delivery_minutes between −0.013 and −0.003.

**What is this data about?** Six months (April–September 2025) of quick-commerce orders: 11,110 completed orders worth INR 2,998,668.80 from 1,175 customers across six cities and five product categories.
**Distributions?** Order value is strongly right-skewed: the mean of INR 269.91 is twice the median of INR 135.00 (skewness 4.61). Most baskets are small groceries, dairy and snacks, while Electronics averages INR 1,189.51.
**Outliers or anomalies?** 897 orders sit above the INR 645.30 upper fence and none below. 635 of them are Electronics, so they are genuine high-value purchases, not errors. The real anomalies (unparseable amounts, negative times, duplicates) were removed in cleaning.
**Relationships?** The largest correlation is order_amount with discount_pct at −0.095 (order_amount–quantity +0.087). Both are very weak, and a correlation of this size does not establish causation; delivery time is unrelated to basket value.

## Part B

**Leakage statement.** No encoder, scaler or resampler ever saw a test row. Every encoder and scaler sits inside a scikit-learn `Pipeline` that is fitted on the 924 training rows only, and on training folds only inside `GridSearchCV`. Over-sampling was applied to the training set only. No order dated 2025-09-01 or later entered any feature: the latest order used by a feature is 2025-08-31 22:35:45, and September orders are used only to build the label. The segment labels from task 3 were **not** used as features.

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, minimum 0, maximum 180 days. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55. All match the reference.

### K-means (StandardScaler on R, F, M; `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at K=4: inertia falls by 525 from K=3 to K=4 but only by 126 from K=4 to K=5. K=4 also has the highest silhouette of the allowed {3, 4, 5}, and its four groups map onto four clear business actions.

**Why inertia alone cannot pick K:** inertia is the total squared distance from each point to its nearest centroid, and adding centroids can only keep that distance the same or shrink it, so K=8 always has lower inertia than K=2 (one cluster per customer would reach zero). A lower inertia therefore cannot tell real structure from over-splitting, which is why the elbow, the silhouette and business meaning decide K.

### Segments (K=4; cluster numbers are labels only and carry no order)

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Premium Big Spenders** (`is_premium = True`) | 138 | 24.12 | 6.77 | 7,129.70 | Premium desk, early access to electronics launches, personal win-back call as soon as recency rises |
| 2 | Loyal Daily Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | Grocery subscription or loyalty points; cross-sell personal care and snacks |
| 0 | Occasional Light Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Bundled staples offers and a "third order this month" coupon to grow frequency |
| 3 | Lapsing Occasional Buyers | 182 | 109.04 | 2.20 | 685.99 | Low-cost win-back (free delivery on the next order); stop spending on those who do not return |

Premium is the segment with the highest mean monetary value. Sizes and profiles match the reference: 138 customers with mean monetary about 7,130 and frequency about 6.8; 419 with frequency about 19.4 and recency about 8.8.

**PCA(n_components=2)** on the scaled RFM matrix: `explained_variance_ratio_` = 0.634 and 0.218, a total of 0.852 (reference 0.634 / 0.218 / 0.852). PC1 loads −0.58 on recency, +0.61 on frequency and +0.53 on monetary, so it measures overall engagement. PC2 loads mainly on monetary (+0.82).

`outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (1,175 rows).

### Lapse problem (cutoff 2025-09-01)

The cohort is every customer with at least one completed order before the cutoff. `lapsed = 1` if the customer has no completed order dated 2025-09-01 through 2025-09-30.

The 11 features use only pre-cutoff orders and the clean customer table:
- recency (days from cutoff to last completed order), frequency, monetary;
- average `discount_pct` and average `delivery_minutes` of completed orders;
- cancellation rate (cancelled ÷ all final-status orders);
- most frequent product category (ties broken alphabetically);
- city, age, gender, tenure in days (cutoff − `signup_date`).

| Figure | Value | Reference |
|---|---:|---:|
| cohort | 1,156 | 1,156 |
| lapsed | 443 (38.32%) | 443 (38.32%) |
| train / test rows (`test_size=0.2, stratify=y, random_state=42`) | 924 / 232 | 924 / 232 |
| baseline "always not lapsed" on test: accuracy | 61.6% | 61.6% |
| baseline lapsed-class F1 (`zero_division=0`) | 0.000 | 0 |

### Preprocessing without leakage

`city`, `gender` and the favourite category are one-hot encoded with `OneHotEncoder(drop="first")`, giving k−1 columns per variable: city_Chennai … city_Pune, gender_male, gender_unknown, fav_category_Electronics … fav_category_Snacks. The eight numeric features are standardised. Both steps sit inside one `Pipeline`. The fitted scaler's learned mean of `frequency` is **8.1147**, while the full-cohort mean is **8.0285**. They differ because the scaler was fitted on the 924 training rows only, so the 232 test rows never influenced it. Code asserts that no feature column has a missing value (0 found).

### Train, tune and evaluate (5-fold `GridSearchCV`, `scoring="f1"`)

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---:|
| `LogisticRegression(max_iter=1000)` | C ∈ {0.01, 0.1, 1, 10, 100} (25 fits) | C = 100 | 0.651 |
| `RandomForestClassifier(n_estimators=100, random_state=42)` | max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20} (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

The features are the brief's minimum list, with no extra features. Logistic regression's CV F1 is flat from C = 0.1 to C = 100 (0.646, 0.645, 0.649, 0.651), so C = 100 wins only a tie-break over the reference's C = 1. Refitting with C = 1, 10 or 100 gives the identical test F1 of 0.648.

Untouched test set (232 rows, 89 lapsed), used only after tuning; metrics are for the lapsed class:

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| Baseline (always not lapsed) | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 | 0.616 |
| Tuned logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 | 0.728 |
| **Tuned random forest (selected)** | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** | 0.737 |

**Choice: the random forest.** It catches the same 58 of 89 lapsing customers as logistic regression, but with 2 fewer loyal customers wrongly targeted (30 vs 32), so it has the best F1 (0.655, above the 0.55 bar and far above the baseline's 0). Missing a lapsing customer costs a whole customer's future orders (the Premium segment averages INR 7,130 over six months), while wrongly targeting a loyal one costs only the price of a voucher. For that reason recall on the lapsed class matters most, and both tuned models are equal there. The forest wins on precision, and the class-balance experiment below shows how to buy more recall when the retention budget allows.

### Class-balance experiment

The 354 lapsed training rows were over-sampled with replacement (`random_state=42`) until both classes had 570 rows, giving 1,140 rows (216 duplicates added). Only the training set was touched. Logistic regression (C = 100) was then refitted on the balanced data and evaluated on the original, unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Balanced (570 / 570) | 0.589 | **0.854** | 0.697 |

Recall moved most, from 0.65 to 0.85, while precision did not rise (it fell to 0.59), matching the reference. For the retention team this means the balanced model finds 76 of the 89 lapsing customers instead of 58, at the cost of contacting 53 rather than 32 customers who would have stayed. That trade is worth it when a voucher costs far less than a lost customer.

### Interpret, persist, use

Random forest, top five `feature_importances_`:

| Feature | Importance | Business reading |
|---|---:|---|
| frequency | 0.280 | How often someone orders is the strongest signal of whether they keep ordering |
| monetary | 0.271 | Total spend separates heavy grocery regulars from light or one-off buyers |
| recency | 0.115 | A long gap since the last order is an early warning |
| avg_delivery_minutes | 0.061 | Slow past deliveries carry some churn signal |
| cancel_rate | 0.052 | Cancellations add a little information |

Logistic regression coefficients (per standard deviation):

| Feature | Coefficient | Business reading |
|---|---:|---|
| frequency | −1.698 | More orders, much lower lapse risk: habitual shoppers stay |
| monetary | +0.285 | At the same frequency, higher spend means somewhat higher lapse risk: big-ticket (electronics) buyers order rarely and drift away |
| recency | +0.040 | More days since the last order, slightly higher risk; most of this signal is already carried by frequency |

The selected random forest was saved with `joblib.dump` to `outputs/lapse_model.joblib` and reloaded with `joblib.load`. The reloaded model reproduces all 232 test predictions and probabilities exactly (asserted in code).

**Retention list.** There are 40 test-set customers in the Premium segment. The top 10 by predicted lapse probability are in `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`): C0862 0.4910, C1112 0.4841, C0060 0.4798, C0802 0.4796, C1008 0.4778, C1000 0.4624, C0525 0.4597, C0133 0.4562, C0267 0.4525, C0565 0.4499. **2 of the 10 truly lapsed** (C0525 and C0565). The Premium flag comes from whole-window RFM that includes the September label window, so it is a post-hoc label and this retention list is illustrative, not a leakage-free evaluation. The model's own features still use only pre-cutoff orders.

## Part C

The pipeline handles all 164 tickets in file order, with `mock_llm.reset()` called right before each full run.

### Input guardrails

| Guardrail | Tickets | Reference |
|---|---|---|
| empty or whitespace text → `skipped_empty`, never sent | 4 (T0157–T0160) | 4 |
| `est_tokens(text) > MAX_INPUT_TOKENS = 400` → `flagged_long`, never sent | 4 (T0161–T0164), each estimated at 734 tokens | 4; 734 |
| eligible for the model | 156 | 156 |

### PII redaction

Three patterns are applied in order: email `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`, phone `\b[6-9]\d{9}\b` → `[PHONE]`, order id `ORD-\d{6}` → `[ORDER_ID]`. Across all 164 tickets this makes **55 email, 91 phone and 132 order-id substitutions, 278 in total** (reference 55 / 91 / 132 / 278). Re-applying the three patterns to the redacted texts finds 0 matches, which is asserted in code. The user message content is exactly the redacted ticket text. `assert_no_pii()` checks every message right before each `mock_llm.chat` call, so no email, phone number or order id can reach the model.

### Prompts

Both prompts contain `CLASSIFY`, list `billing, delivery, technical, fraud`, and require a JSON-only reply `{"category": "...", "reason": "..."}`. The zero-shot prompt (38 estimated tokens) has no `EXAMPLES:` text. The few-shot prompt (117 estimated tokens) adds an `EXAMPLES:` line followed by four labelled examples, one per category, including a fraud example.

### Resilient calls

Every model call goes through `call_with_retry`: at most 3 retries per ticket, waiting `BASE_DELAY × 2**n × random.uniform(0.9, 1.15)` with `BASE_DELAY = 0.05` s before retry n, using a seeded `random.Random(42)` for the jitter. Each retry logs one line with the ticket id, retry number and wait, for example `retry ticket=T0002 retry_number=0 wait=0.0530s error='503 server busy'`.

A full run over the 156 eligible tickets raised exactly **19 `TransientLLMError`s, all recovered on retry 0**. Every wait lay between 0.0453 s and 0.0562 s, inside the allowed 0.045–0.0575 s, and no ticket ended in a permanent error.

### Output validation

A reply is accepted as JSON whose `category` is one of the four allowed values (`classified_json`). If the reply is not valid JSON, the words are extracted with `re.findall(r"[a-z]+", text.lower())` and it is accepted if exactly one category word appears (`classified_keyword`). Everything else becomes `flagged_invalid_output` and goes to a human, including valid JSON with a category outside the list, such as the mock's `"shipping"`; the pipeline never guesses. A full run gives **135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`**, matching the reference.

The 147 classified tickets (135 JSON + 12 keyword) are rule-validated. The other 17 go to humans: 4 empty, 4 long and 9 invalid outputs.

### Zero-shot vs few-shot (full runs, no budget)

| Run | Classified | Accuracy vs `true_category` | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 126 / 147 = **85.7%** | 10,336 | 70.3 |
| Few-shot | 147 | 147 / 147 = **100.0%** | 22,660 | 154.1 |

Zero-shot confusion table (rows = true, columns = predicted, classified tickets only):

| true \ predicted | billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

Every zero-shot error is a fraud ticket predicted billing (11) or a technical ticket predicted billing (10), exactly as in the reference.

**When the few-shot tokens are worth paying for:** the few-shot prompt costs 154.1 tokens per classified ticket against 70.3 for zero-shot (+83.8, about 2.2×), or 12,324 extra tokens over the run. In exchange it fixes all 21 zero-shot errors, about 587 extra tokens per corrected ticket, and lifts accuracy from 85.7% to 100%. It is worth paying here because 11 of those errors are fraud tickets routed to billing, which would miss the security team's 2-hour window. Zero-shot is acceptable only for low-stakes routing where a human reviews every billing ticket anyway.

### Token budget, routing and final output

**(a) Budget demo.** Before each call, if the running sum of `usage.total_tokens` is already ≥ `TOKEN_BUDGET`, no call is made and the ticket gets `budget_exhausted` (routed to human review). The zero-shot run with **TOKEN_BUDGET = 6000** stopped after 6,036 tokens. **65 tickets ended `budget_exhausted`**, starting at file row 95 (T0093). The other statuses were 79 classified_json, 7 classified_keyword, 5 flagged_invalid_output, 4 skipped_empty and 4 flagged_long. The result is saved as `outputs/triage_budget_demo.csv`. With a 38-token prompt, 65 is consistent with the brief's examples (51 at about 29 tokens, 58 at about 33).

**(b) Final run.** A few-shot run with TOKEN_BUDGET = 100000 used 22,660 tokens and is routed with `outputs/customer_segments.csv` from Part B. Customers without a segment row count as not premium.

| Rule (checked in this order) | Route | Priority | Tickets |
|---|---|---|---:|
| status `skipped_empty`, `flagged_long`, `flagged_invalid_output` or `budget_exhausted` | human_review | NORMAL | 17 |
| category `fraud` | security_team | HIGH | 34 |
| customer `is_premium` | premium_desk | HIGH | 11 |
| everything else | support_queue | NORMAL | 102 |

`outputs/triage_results.csv` has 164 rows with columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`; `category` and `segment_name` are blank when not available.

## Part D

### Policy RAG

**Chunking.** Each of the five policy files is split on whitespace into 40-word chunks that start every 30 words (10-word overlap). Chunking stops after the first chunk that reaches the last word, and every chunk keeps its source file name. This gives **15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2**, matching the reference.

**Retrieval.** `TfidfVectorizer()` is fitted on the 15 chunk texts (200-term vocabulary). A query is transformed with the same fitted vectorizer and chunks are ranked by `cosine_similarity`. `retrieve(query, k=3)` returns a list of `{"source", "score", "text"}` dictionaries. For `refund policy late delivery`, the top three are all from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454), matching the reference.

**TF and IDF.** TF (term frequency) measures how often a word appears in a chunk, while IDF (inverse document frequency) measures how rare the word is across all chunks, so common words like "order" count little and specific words like "wallet" decide the ranking. Its limitation is that it only matches identical words: "money back" does not match "refund", and "was my parcel slow" shares almost no words with "delivered in more than 45 minutes", even though the meanings are the same.

**Answering.** `answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`. The system message names the `POLICY ASSISTANT`, tells it to answer only from the supplied context and to say it could not find the information otherwise. The user message is `CONTEXT:`, then each chunk under its own `[<source file name>]` line, then `QUESTION:` and the question. The function returns the model's text and the sources used.

### Retrieval and grounding evaluation (`data/raw/policy_eval.csv`)

| Chunking | expected_doc in top 3 (12 answerable) | answer contains expected_fact (12) | fallback "I could not find…" (2 unanswerable) |
|---|---:|---:|---:|
| 40 words, overlap 10 | **12/12** | **12/12** | **2/2** |
| 40 words, overlap 0 (13 chunks) | 12/12 | **10/12** | 2/2 |

With zero overlap, two answerable questions break because their answer sentence is cut at a chunk boundary:
- *"What delivery time makes a completed order eligible for a full refund?"* The chunk ends `…delivered in more than 45 minutes, the customer` and the next one starts `receives a full refund of the order amount.`, so the assistant returns only the second half (`receives a full refund of the order amount.`), which lacks "more than 45 minutes".
- *"How soon is money returned when a payment fails but is deducted?"* The chunk ends `…the amount is returned automatically` and `within 48 hours.` starts the next chunk, so "48 hours" never reaches the answer.

The 10-word overlap keeps each of these sentences whole inside at least one chunk.

### Tools (`TOOLS_SPEC` + `TOOL_MAP`)

| Tool | Description | Returns |
|---|---|---|
| `get_order(order_id)` | Look up one order in `data/instamart.db` | `{"order_id", "status", "order_amount", "delivery_minutes"}` or `{"error": "order not found"}` |
| `search_policy(query)` | Retrieve the three most relevant policy chunks (the Part D `retrieve`) | `{"results": [...]}` |
| `calculate_refund(order_amount, delivery_minutes)` | 100% of the amount if delivered in more than 45 minutes, else 0 (cancelled orders have no delivery time → 0) | `{"refund_inr": amount}` |
| `issue_refund(order_id, refund_inr)` | Issue a refund of at most INR 1000 and append it to `outputs/refund_ledger.jsonl` with a timestamp and the tool name; a request above INR 1000 is refused (nothing paid, nothing logged) because it must go through `escalate_to_human` | `{"refund_inr": amount}` |
| `escalate_to_human(order_id, refund_inr)` | Send a refund above INR 1000 to a human lead and append it to the ledger | `{"refund_inr": amount}` |

Each `TOOLS_SPEC` entry has a name, a one-sentence description and a JSON-schema parameter specification. The model requests a tool by name with an `arguments` dictionary, and the loop calls `TOOL_MAP[name](**arguments)`.

### ReAct loop: the seven scenarios (`max_iterations=6, token_limit=6000, time_limit_seconds=10`)

`run_agent(user_message, scenario_id, ...)` starts with a system message containing `AGENT` and the user message. Each iteration calls `chat`. If the reply has no tool calls, its content is the final answer and the status is the word before the colon. Otherwise the loop appends the assistant message `{"role": "assistant", "content": None, "tool_calls": [{"id", "name", "arguments"}]}`, runs the tool, and appends `{"role": "tool", "tool_call_id", "name", "content": <JSON result>}`. It returns `status, final_answer, steps, llm_calls, tokens_used`. Each run appends one JSON line to `outputs/agent_runs.jsonl`, holding `scenario_id`, `status`, `final_answer`, `llm_calls`, `tokens_used` and every step's tool, arguments and observation.

| Scenario | Status | Refund (INR) | Expected | Model calls | Tokens used | Tools executed |
|---|---|---:|---|---:|---:|---|
| S1 | REFUND | 128.25 | REFUND 128.25 | 5 | 1,147 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | 135.00 | REFUND 135.00 | 5 | 1,147 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | 1999.00 | ESCALATE 1999.00 | 5 | 1,134 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | 0.00 | DENIED | 4 | 805 | get_order → search_policy → calculate_refund |
| S5 | DENIED | 0.00 | DENIED | 3 | 487 | get_order → search_policy |
| S6 | NOT_FOUND | 0.00 | NOT_FOUND | 2 | 172 | get_order |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | FALLBACK | 6 | 3,300 | search_policy × 5 (the 6th requested call was not executed: no model call was left to read it) |

Every refund equals `expected_refund_inr` to within 0.01. S1 takes 5 model calls, and S7 finished all six model calls under the default 6,000-token limit (3,300 tokens).

### Stop safely

| Limit | Rule | Demonstration | Status |
|---|---|---|---|
| (a) max_iterations | after `max_iterations` model calls without a final answer, stop | S7, default limits | **FALLBACK_MAX_ITERATIONS** |
| (b) token_limit | after every model call (tool request or final answer) add `usage.total_tokens`; if the total exceeds `token_limit`, stop immediately without executing the requested tool | S1 with `token_limit=200`: stopped after 2 model calls (202 tokens); get_order ran, the requested search_policy did not | **FALLBACK_TOKEN_LIMIT** |
| (c) time_limit_seconds | at the top of each iteration, stop if `time.monotonic()` elapsed ≥ `time_limit_seconds` | S1 with `time_limit_seconds=0`: stopped before the first model call (0 calls) | **FALLBACK_TIMEOUT** |

Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`. For example: `HANDOFF: token limit of 200 exceeded (202 used). Request: '…'. Tools completed so far: get_order. Model calls 2, tokens 202. A human agent should review and decide.`

### Crash and resume

After every tool observation is appended, the loop writes `state/<run_id>.json`. The checkpoint holds the run id, scenario id, full message list, `tokens_used`, step number, list of tools already executed, model-call count and elapsed time. `resume_agent(run_id)` loads it and continues the same loop.

For S1, a `SimulatedCrash` was raised right after the **second** tool execution, once its checkpoint had been saved (`state/demo-S1-crash-resume.json`: step 2, 6 messages, 202 tokens). `resume_agent` then finished the run.

Tools executed across the crash and the resume, in order:
1. `get_order` (before the crash)
2. `search_policy` (before the crash)
3. `calculate_refund` (after the resume)
4. `issue_refund` (after the resume)

That is four executions, none repeated. Final status **REFUND, INR 128.25**, 5 model calls in total, matching the uninterrupted run.

`outputs/refund_ledger.jsonl` holds the issued and escalated actions: issue_refund ORD-100048 INR 128.25 (S1), issue_refund ORD-100067 INR 135.00 (S2), escalate_to_human ORD-100145 INR 1999.00 (S3), and issue_refund ORD-100048 INR 128.25 from the crash-and-resume demonstration. Each entry carries its run id, so the demo refund can be told apart from S1's. `outputs/agent_runs.jsonl` holds 10 runs (7 scenarios, 2 stop demonstrations, 1 crash-and-resume) and `state/` holds their 10 checkpoints.

### ReAct vs plan-and-execute

ReAct asks the model after every observation, so model calls grow with the steps taken: 5 calls and 1,147 tokens for S1, 3 for the cancelled order S5, and 2 for the unknown order S6. Each call also re-sends the growing history, so tokens grow faster than calls. Plan-and-execute would ask once for the whole step list (get_order → search_policy → calculate_refund → issue_refund) and then run the tools with no further model calls. That is about 2 calls for S1 and fewer tokens, but only while the plan holds.

The difference shows up on unexpected observations. When `get_order` returns `NOT_FOUND`, or a cancelled order arrives, ReAct simply reads the observation and stops on the next call (S6 in 2 calls, S5 in 3). A plan-and-execute agent has already committed to calculating and issuing a refund. It must notice the error and replan, paying an extra call and risking a refund on a cancelled order if the check is missed.

For this refund workflow I would choose **ReAct**. The task has only four or five steps, so the saving from planning is small (roughly 3 calls per request). What the business cannot afford is acting on stale assumptions, such as refunding a cancelled order or skipping escalation above INR 1000, and ReAct re-decides after each fact. Its known risk, looping, is already contained by the max-iteration, token and time limits (S7 stops at 6 calls).

## Limitations and Next Steps

- **The retention list is illustrative, not a leakage-free evaluation.** The Premium flag comes from the whole-window RFM of Part B task 3, which includes the September label window, so it is a post-hoc whole-window label. A production version would compute segments from pre-cutoff orders only. The lapse model itself is leakage-free: its features use only orders before 2025-09-01, and no segment label is a feature.
- **Logistic regression's best C (100) differs from the reference (C = 1)** because the cross-validated F1 is flat from C = 0.1 to 100 (within 0.006). C = 1, 10 and 100 give the same test F1 of 0.648, so the conclusions do not change.
- **The mock LLM is keyword-based.** The 85.7% / 100% accuracies describe `mock_llm.py`, not a real model. A hosted model would need fresh measurements, but the guardrails (redaction, validation, retries, budget, human review) carry over unchanged.
- **Token counts are estimates** (`ceil(words × 1.3)`), not a real tokenizer, so the budgets and costs are relative measures.
- **TF-IDF retrieval misses synonyms and paraphrases.** Next step: sentence embeddings or a hybrid BM25 + embedding retriever, with a reranker, evaluated on more than 14 questions.
- **The refund ledger is not idempotent.** The crash-and-resume demo issues S1's refund a second time as a separate run, so ORD-100048 appears twice. Next step: an idempotency key per (order_id, action) so that a second refund for the same order is refused.
- **Imputation choices** (city-median age, category-median delivery time) are simple and documented; the 774 imputed delivery times all fall at 25–25.5 minutes and slightly flatten that distribution.
- **Cancellation rate counts only final statuses**, since the earlier `pending` records were removed in cleaning; it carries little signal (importance 0.052).
