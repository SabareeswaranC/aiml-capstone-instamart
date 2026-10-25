"""Part D, tasks 5-8 -- refund agent: tools + TOOL_MAP, ReAct loop, safe stops, checkpoint/resume.

Run: python -m copilot.part_d_agent
     (writes outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and checkpoints in state/)
The agent reads orders from data/instamart.db (Part A) and searches policies with part_d_rag.retrieve.
"""
import csv
import json
import os
import sqlite3
import time
from datetime import datetime

import mock_llm
from copilot.part_d_rag import retrieve
from copilot.paths import DB, OUT, RAW, STATE, Report, ensure_dirs

RUNS = os.path.join(OUT, "agent_runs.jsonl")
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
LATE_MINUTES, AUTO_LIMIT_INR = 45, 1000
SYSTEM_PROMPT = ("You are the refund AGENT for Instamart support. Decide refund requests by calling tools: look up "
                 "the order, check the refund policy, calculate the refund, then issue it or escalate it to a human. "
                 "Finish with one line starting REFUND, ESCALATE, DENIED or NOT_FOUND followed by a colon.")


# ---------------------------------------------------------------- tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Full refund when delivery took more than 45 minutes, otherwise 0 (cancelled orders have no delivery time)."""
    if delivery_minutes is None or float(delivery_minutes) <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger(tool, order_id, refund_inr):
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool, "order_id": order_id,
             "refund_inr": round(float(refund_inr), 2), "run_id": _CURRENT_RUN.get("run_id")}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    if float(refund_inr) > AUTO_LIMIT_INR:       # rule guard: never auto-issue above INR 1000, nothing is paid
        return {"refund_inr": 0.0, "error": "refunds above INR 1000 must use escalate_to_human"}
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


def _spec(name, description, **params):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": params, "required": list(params)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up one order in the Instamart database by its order id.",
          order_id={"type": "string", "description": "Order id such as ORD-100123"}),
    _spec("search_policy", "Retrieve the three policy chunks most relevant to a query.",
          query={"type": "string", "description": "Search text"}),
    _spec("calculate_refund", "Compute the refund: 100% of the amount if delivery took more than 45 minutes, else 0.",
          order_amount={"type": "number", "description": "Order amount in INR"},
          delivery_minutes={"type": ["number", "null"], "description": "Delivery time in minutes"}),
    _spec("issue_refund", "Issue a refund of at most INR 1000 and record it in the refund ledger.",
          order_id={"type": "string"}, refund_inr={"type": "number"}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          order_id={"type": "string"}, refund_inr={"type": "number"}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
_CURRENT_RUN = {}


# ---------------------------------------------------------------- loop
class SimulatedCrash(RuntimeError):
    """Raised on purpose to demonstrate resume_agent; carries the run id to resume."""

    def __init__(self, message, run_id):
        super().__init__(message)
        self.run_id = run_id


def _checkpoint(state):
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _fallback(state, status, reason):
    done = ", ".join(s["tool"] for s in state["steps"] if s.get("executed", True)) or "none"
    state["status"] = status
    state["final_answer"] = (f"HANDOFF: {reason}. Request: '{state['user_message']}'. Tools completed so far: {done}. "
                             f"Model calls {state['llm_calls']}, tokens {state['tokens_used']}. "
                             f"A human agent should review and decide.")
    return state


def _loop(state, crash_after_tools=None):
    """Shared by run_agent and resume_agent. Time is measured from when this (re)start began plus time already used."""
    _CURRENT_RUN["run_id"] = state["run_id"]
    lim = state["limits"]
    started = time.monotonic() - state.get("elapsed_seconds", 0.0)
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {lim['time_limit_seconds']} s reached")
        if state["llm_calls"] >= lim["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS",
                             f"{lim['max_iterations']} model calls without a final answer")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        msg = resp.choices[0].message
        stop = None
        if state["tokens_used"] > lim["token_limit"]:          # checked after every model call
            stop = ("FALLBACK_TOKEN_LIMIT", f"token limit of {lim['token_limit']} exceeded ({state['tokens_used']} used)")
        elif not msg.tool_calls:
            state["final_answer"] = msg.content
            state["status"] = msg.content.split(":", 1)[0].strip()
            return state
        elif state["llm_calls"] >= lim["max_iterations"]:
            # no model call is left to read this tool's observation, so the requested tool is not executed
            stop = ("FALLBACK_MAX_ITERATIONS", f"{lim['max_iterations']} model calls without a final answer")
        if stop:
            for tc in msg.tool_calls:
                state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": None,
                                       "executed": False})
            return _fallback(state, *stop)
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = TOOL_MAP[tc.name](**tc.arguments)
            state["tools_executed"].append(tc.name)
            state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
            if tc.name in ("issue_refund", "escalate_to_human"):
                state["refund_inr"] = result["refund_inr"]
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(result)})
            state["step"] = len(state["tools_executed"])
            state["elapsed_seconds"] = time.monotonic() - started
            _checkpoint(state)
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})", state["run_id"])


_RUN_COUNTER = {"n": 0}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tools=None, run_id=None):
    _RUN_COUNTER["n"] += 1
    run_id = run_id or f"run{_RUN_COUNTER['n']:02d}-{scenario_id}"
    state = {"run_id": run_id, "scenario_id": scenario_id,
             "user_message": user_message,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
             "steps": [], "tools_executed": [], "step": 0, "llm_calls": 0, "tokens_used": 0,
             "refund_inr": 0.0, "status": None, "final_answer": None, "elapsed_seconds": 0.0}
    return _result(_loop(state, crash_after_tools))


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    state["resumed_from_step"] = state["step"]
    return _result(_loop(state))


def _result(state):
    _checkpoint(state)
    return {k: state[k] for k in ("run_id", "scenario_id", "status", "final_answer", "steps", "llm_calls",
                                  "tokens_used", "refund_inr", "tools_executed")}


def log_run(result, label=None):
    line = {"scenario_id": result["scenario_id"], "run_id": result["run_id"], "label": label,
            "status": result["status"], "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"], "refund_inr": result["refund_inr"], "steps": result["steps"]}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------- demo
def main():
    ensure_dirs()
    for path in (RUNS, LEDGER):                      # every Reproduce run starts from empty logs
        open(path, "w").close()
    for name in os.listdir(STATE):
        if name.endswith(".json"):
            os.remove(os.path.join(STATE, name))
    rep = Report("part_d_agent")

    rep("## Task 5 - tools")
    for t in TOOLS_SPEC:
        rep(f"  {t['name']}({', '.join(t['parameters']['properties'])}) - {t['description']}")
    rep(f"TOOL_MAP: {', '.join(TOOL_MAP)}")

    rep("\n## Task 6 - seven scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10)")
    scenarios = list(csv.DictReader(open(os.path.join(RAW, "agent_scenarios.csv"), encoding="utf-8")))
    rep(f"{'id':<4}{'expected':<10}{'status':<26}{'refund INR':>11}{'expected INR':>13}{'calls':>6}{'tokens':>7}  tools")
    all_ok = True
    for sc in scenarios:
        r = run_agent(sc["message"], sc["scenario_id"])
        log_run(r, "scenario")
        expected = sc["expected_outcome"]
        ok = (r["status"] == expected or (expected == "FALLBACK" and r["status"].startswith("FALLBACK"))) and \
            abs(r["refund_inr"] - float(sc["expected_refund_inr"])) < 0.01
        all_ok &= ok
        skipped = [s["tool"] for s in r["steps"] if not s.get("executed", True)]
        rep(f"{sc['scenario_id']:<4}{expected:<10}{r['status']:<26}{r['refund_inr']:>11.2f}"
            f"{float(sc['expected_refund_inr']):>13.2f}{r['llm_calls']:>6}{r['tokens_used']:>7}  "
            f"{' > '.join(r['tools_executed']) or '-'}"
            f"{' (requested, not executed: ' + ', '.join(skipped) + ')' if skipped else ''}  {'OK' if ok else 'MISMATCH'}")
        rep(f"      final answer: {r['final_answer']}")
    rep(f"all seven scenarios match expected outcome and refund: {all_ok}")

    rep("\n## Task 7 - stop safely")
    s1 = scenarios[0]
    r = run_agent(s1["message"], "S1", token_limit=200, run_id="demo-S1-token-limit-200")
    log_run(r, "demo token_limit=200")
    rep(f"S1 with token_limit=200 -> {r['status']} after {r['llm_calls']} model calls, {r['tokens_used']} tokens; "
        f"tools executed {r['tools_executed']}; requested but not executed: "
        f"{[s['tool'] for s in r['steps'] if not s.get('executed', True)]}")
    rep(f"      final answer: {r['final_answer']}")
    r = run_agent(s1["message"], "S1", time_limit_seconds=0, run_id="demo-S1-time-limit-0")
    log_run(r, "demo time_limit_seconds=0")
    rep(f"S1 with time_limit_seconds=0 -> {r['status']} after {r['llm_calls']} model calls")
    rep(f"      final answer: {r['final_answer']}")

    rep("\n## Task 8 - crash and resume (S1, crash right after the 2nd tool execution)")
    try:
        run_agent(s1["message"], "S1", crash_after_tools=2, run_id="demo-S1-crash-resume")
        raise AssertionError("the simulated crash did not happen")
    except SimulatedCrash as err:
        run_id = err.run_id
        saved = json.load(open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8"))
        rep(f"crashed: {err}; checkpoint state/{run_id}.json holds step {saved['step']}, "
            f"tools {saved['tools_executed']}, {len(saved['messages'])} messages, {saved['tokens_used']} tokens")
    r = resume_agent(run_id)
    log_run(r, "crash-and-resume")
    before = saved["tools_executed"]
    after = r["tools_executed"][len(before):]
    rep(f"resumed run {run_id}: status {r['status']}, refund INR {r['refund_inr']:.2f}, "
        f"total model calls {r['llm_calls']}, tokens {r['tokens_used']}")
    rep(f"tools before the crash: {before}; after the resume: {after}")
    rep(f"all tool executions in order: {r['tools_executed']} ({len(r['tools_executed'])} executions, "
        f"{'none repeated' if len(set(r['tools_executed'])) == len(r['tools_executed']) else 'REPEATS'})")
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and abs(r["refund_inr"] - 128.25) < 0.01

    ledger = [json.loads(x) for x in open(LEDGER, encoding="utf-8")]
    rep(f"\nrefund_ledger.jsonl: {len(ledger)} entries - " +
        "; ".join(f"{e['tool']} {e['order_id']} INR {e['refund_inr']:.2f}" for e in ledger))
    rep(f"agent_runs.jsonl: {sum(1 for _ in open(RUNS))} runs; state/: "
        f"{len([n for n in os.listdir(STATE) if n.endswith('.json')])} checkpoint files")
    rep.save()


if __name__ == "__main__":
    main()
