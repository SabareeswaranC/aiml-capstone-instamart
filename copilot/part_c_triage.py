"""Part C -- support-ticket triage around mock_llm.py: guardrails, redaction, retries, validation, budget, routing.

Run: python -m copilot.part_c_triage
     (writes outputs/triage_budget_demo.csv and outputs/triage_results.csv)
Tickets are processed in file order. No email, phone number or order id ever reaches the model: every
prompt passes through assert_no_pii() right before mock_llm.chat is called.
"""
import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens
from copilot.paths import OUT, RAW, SEED, Report, ensure_dirs

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
DEMO_BUDGET = 6000
FINAL_BUDGET = 100000

# applied in this order: email, then phone, then order id
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT = (
    "You CLASSIFY customer support tickets for a quick-commerce grocery app. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly {"category": "...", "reason": "..."}, and no other text.'
)
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorised transaction"}
Ticket: I was charged twice for one order. -> {"category": "billing", "reason": "double charge"}
Ticket: The rider has not arrived after an hour. -> {"category": "delivery", "reason": "late order"}
Ticket: The app freezes on the payment page. -> {"category": "technical", "reason": "app error"}"""

LOG = []          # retry log lines, kept so the report can show them


def redact(text):
    """Return (redacted_text, {type: substitutions})."""
    counts = {}
    for name, pattern, placeholder in PII_PATTERNS:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def assert_no_pii(messages):
    for m in messages:
        for name, pattern, _ in PII_PATTERNS:
            assert not pattern.search(m["content"] or ""), f"{name} would reach the model"


def call_with_retry(messages, ticket_id, jitter):
    """At most MAX_RETRIES retries; wait before retry n is BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    assert_no_pii(messages)
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), attempt
        except TransientLLMError as err:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * jitter.uniform(0.9, 1.15)
            line = f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s error='{err}'"
            print(line)
            LOG.append((attempt, wait, line))
            time.sleep(wait)


def validate(reply):
    """-> (status, category). JSON with an allowed category, else exactly one category word, else human review."""
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        found = [c for c in CATEGORIES if c in set(re.findall(r"[a-z]+", (reply or "").lower()))]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""


def load_tickets():
    return pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False, dtype=str)


def run(tickets, system_prompt, budget=None):
    """One full pass in file order. Returns a list of per-ticket dicts."""
    mock_llm.reset()
    jitter = random.Random(SEED)
    spent, rows = 0, []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0, "retries": 0}
        if not t.text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif budget is not None and spent >= budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            resp, retries = call_with_retry(messages, t.ticket_id, jitter)
            used = resp.usage.total_tokens
            spent += used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
            row["tokens_used"], row["retries"] = used, retries
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df, label, rep):
    classified = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = int((classified.category == classified.true_category).sum())
    tokens = int(df.tokens_used.sum())
    rep(f"{label}: statuses {df.status.value_counts().to_dict()}")
    rep(f"  classified {len(classified)}; accuracy {correct}/{len(classified)} = {100 * correct / len(classified):.1f}%; "
        f"total tokens {tokens:,}; tokens per classified ticket {tokens / len(classified):.1f}; "
        f"model calls {int((df.tokens_used > 0).sum())}, retries {int(df.retries.sum())}")
    return classified


def route(df):
    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"), dtype={"customer_id": str})
    seg = seg.set_index("customer_id")
    df = df.copy()
    df["segment_name"] = df.customer_id.map(seg["segment_name"]).fillna("")
    premium = df.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = df.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = ~human & (df.category == "fraud")
    prem = ~human & ~fraud & premium
    df["route"], df["priority"] = "support_queue", "NORMAL"
    df.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    df.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    df.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    return df


def main():
    ensure_dirs()
    rep = Report("part_c_triage")
    tickets = load_tickets()
    rep(f"# Part C - {len(tickets)} tickets in file order")

    rep("\n## Task 1 - input guardrails")
    empty = tickets.text.str.strip() == ""
    long_ = ~empty & (tickets.text.map(est_tokens) > MAX_INPUT_TOKENS)
    rep(f"empty/whitespace tickets -> skipped_empty: {int(empty.sum())} ({', '.join(tickets.ticket_id[empty])})")
    rep(f"est_tokens > {MAX_INPUT_TOKENS} -> flagged_long: {int(long_.sum())} ({', '.join(tickets.ticket_id[long_])}); "
        f"estimates {sorted(set(tickets.text[long_].map(est_tokens)))} tokens")
    rep(f"eligible for the model: {int((~empty & ~long_).sum())}")

    rep("\n## Task 2 - PII redaction (email -> phone -> order id)")
    totals = {name: 0 for name, _, _ in PII_PATTERNS}
    redacted = []
    for text in tickets.text:
        clean, counts = redact(text)
        redacted.append(clean)
        for k, v in counts.items():
            totals[k] += v
    rep(f"substitutions across all {len(tickets)} tickets: emails {totals['email']}, phones {totals['phone']}, "
        f"order ids {totals['order_id']}, total {sum(totals.values())}")
    left = sum(len(p.findall(t)) for t in redacted for _, p, _ in PII_PATTERNS)
    assert left == 0
    rep(f"re-applying the three patterns to the redacted texts finds {left} matches (asserted)")
    sample = tickets.text[~empty & ~long_].iloc[0]
    rep(f"example: {sample!r}\n      -> {redact(sample)[0]!r}")

    rep("\n## Task 3 - prompts")
    rep(f"zero-shot ({est_tokens(ZERO_SHOT)} est. tokens): {ZERO_SHOT}")
    rep(f"few-shot ({est_tokens(FEW_SHOT)} est. tokens) adds an EXAMPLES: block with 4 labelled examples incl. fraud")

    rep("\n## Tasks 4-6 - full runs (mock_llm.reset() before each, no token budget)")
    LOG.clear()
    zero = run(tickets, ZERO_SHOT)
    zero_log = list(LOG)
    rep(f"zero-shot retry log: {len(zero_log)} TransientLLMErrors, all recovered "
        f"(retry numbers used {sorted({a for a, _, _ in zero_log})}; waits "
        f"{min(w for _, w, _ in zero_log):.4f}-{max(w for _, w, _ in zero_log):.4f}s, allowed 0.0450-0.0575s for retry 0)")
    for _, _, line in zero_log:
        rep("  " + line)
    classified_zero = summarise(zero, "zero-shot", rep)
    LOG.clear()
    few = run(tickets, FEW_SHOT)
    rep(f"few-shot retry log: {len(LOG)} TransientLLMErrors, all recovered")
    summarise(few, "few-shot", rep)

    rep("\nzero-shot confusion table (rows = true category, columns = predicted), classified tickets only:")
    rep(pd.crosstab(classified_zero.true_category, classified_zero.category).to_string())
    wrong = classified_zero[classified_zero.category != classified_zero.true_category]
    rep("zero-shot errors: " + ", ".join(f"{k[0]} predicted {k[1]} ({v})" for k, v in
                                         wrong.groupby(["true_category", "category"]).size().items()))

    rep(f"\n## Task 7a - zero-shot run with TOKEN_BUDGET = {DEMO_BUDGET}")
    demo = route(run(tickets, ZERO_SHOT, budget=DEMO_BUDGET))
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    demo[cols].to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
    first = demo.index[demo.status == "budget_exhausted"][0]
    rep(f"budget_exhausted tickets: {int((demo.status == 'budget_exhausted').sum())} (first at row {first + 1}, "
        f"{demo.ticket_id[first]}); tokens spent before stopping {int(demo.tokens_used.sum()):,}")
    rep(f"statuses {demo.status.value_counts().to_dict()}")

    rep(f"\n## Task 7b - final few-shot run with TOKEN_BUDGET = {FINAL_BUDGET} and routing")
    final = route(run(tickets, FEW_SHOT, budget=FINAL_BUDGET))
    final[cols].to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
    rep(f"wrote outputs/triage_results.csv ({len(final)} rows); tokens used {int(final.tokens_used.sum()):,}")
    rep(f"statuses {final.status.value_counts().to_dict()}")
    rep("routes: " + ", ".join(f"{k} {v}" for k, v in final.groupby(['route', 'priority']).size().items()))
    rep(f"categories among model-classified tickets: {final[final.category != ''].category.value_counts().to_dict()}")
    rep.save()


if __name__ == "__main__":
    main()
