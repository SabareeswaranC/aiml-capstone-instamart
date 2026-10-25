"""Part D, tasks 1-4 -- chunk the policies, TF-IDF retrieval, grounded answers through mock_llm, evaluation.

Run: python -m copilot.part_d_rag
The refund agent (part_d_agent.py) imports retrieve() from here, so both use exactly the same index.
"""
import csv
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm
from copilot.paths import POLICIES, RAW, Report

CHUNK_WORDS, OVERLAP = 40, 10
FALLBACK_PREFIX = "I could not find"
SYSTEM_PROMPT = ("You are the POLICY ASSISTANT for the Instamart support desk. Answer only from the supplied "
                 "CONTEXT. If the answer is not in the context, say: I could not find that information in the "
                 "policy documents.")


def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP):
    """Chunks of `size` words starting every size-overlap words; stop after the first chunk reaching the last word."""
    step, chunks, start = size - overlap, [], 0
    while True:
        chunks.append(words[start:start + size])
        if start + size >= len(words):
            return chunks
        start += step


def load_chunks(overlap=OVERLAP):
    chunks = []
    for name in sorted(os.listdir(POLICIES)):
        if name.endswith(".txt"):
            words = open(os.path.join(POLICIES, name), encoding="utf-8").read().split()
            chunks += [{"source": name, "text": " ".join(c)} for c in chunk_words(words, overlap=overlap)]
    return chunks


class PolicyIndex:
    def __init__(self, overlap=OVERLAP):
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in best]

    def answer(self, question, k=3):
        hits = self.retrieve(question, k)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        reply = mock_llm.chat(messages).choices[0].message.content
        return reply, [h["source"] for h in hits]


_DEFAULT = None


def retrieve(query, k=3):
    """Module-level retrieve over the default index (40-word chunks, 10-word overlap)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex()
    return _DEFAULT.retrieve(query, k)


def answer(question):
    retrieve("")                      # build the default index on first use
    return _DEFAULT.answer(question)


def flat(text):
    return " / ".join(line.strip() for line in text.splitlines() if line.strip())


def evaluate(index, rep, label):
    rows = list(csv.DictReader(open(os.path.join(RAW, "policy_eval.csv"), encoding="utf-8")))
    answerable = [r for r in rows if r["expected_doc"] != "NONE"]
    unanswerable = [r for r in rows if r["expected_doc"] == "NONE"]
    hit = grounded = fallback = 0
    failures = []
    rep(f"\n{label}")
    rep(f"{'question':<74}{'doc@3':>6}{'fact':>6}  answer")
    for r in rows:
        reply, sources = index.answer(r["question"])
        if r["expected_doc"] == "NONE":
            ok_doc, ok_fact = "-", reply.startswith(FALLBACK_PREFIX)
            fallback += ok_fact
        else:
            ok_doc, ok_fact = r["expected_doc"] in sources, r["expected_fact"] in reply
            hit += ok_doc
            grounded += ok_fact
            if not ok_fact:
                failures.append((r, reply, sources))
        rep(f"{r['question'][:73]:<74}{str(ok_doc):>6}{str(ok_fact):>6}  {flat(reply)[:100]}")
    rep(f"expected_doc in top 3: {hit}/{len(answerable)}; answer contains expected_fact: "
        f"{grounded}/{len(answerable)}; unanswerable -> fallback: {fallback}/{len(unanswerable)}")
    return failures


def main():
    rep = Report("part_d_rag")
    index = PolicyIndex()
    rep("## Task 1 - chunks (40 words, overlap 10)")
    per_doc = {}
    for c in index.chunks:
        per_doc[c["source"]] = per_doc.get(c["source"], 0) + 1
    rep(f"{len(index.chunks)} chunks in total: " + ", ".join(f"{k.replace('_policy.txt', '')} {v}"
                                                           for k, v in sorted(per_doc.items(), key=lambda x: -x[1])))
    rep(f"vocabulary size of the fitted TfidfVectorizer: {len(index.vectorizer.vocabulary_)}")

    rep("\n## Task 2 - retrieve('refund policy late delivery', k=3)")
    for h in index.retrieve("refund policy late delivery"):
        rep(f"  {h['source']:<22} score {h['score']:.4f}  {h['text'][:80]}...")

    rep("\n## Task 3 - answer() example")
    reply, sources = index.answer("Who must approve a refund above INR 1000?")
    rep(f"  answer: {flat(reply)}\n  sources: {sources}")

    rep("\n## Task 4 - evaluation on data/raw/policy_eval.csv")
    evaluate(index, rep, "with overlap 10:")
    zero = PolicyIndex(overlap=0)
    rep(f"\nzero-overlap index: {len(zero.chunks)} chunks")
    failures = evaluate(zero, rep, "with overlap 0:")
    for r, reply, sources in failures:
        rep(f"BREAKS with overlap 0: '{r['question']}' expected '{r['expected_fact']}' from {r['expected_doc']}; "
            f"got '{flat(reply)}' (sources {sources})")
        full = " ".join(open(os.path.join(POLICIES, r["expected_doc"]), encoding="utf-8").read().split())
        sentence = next(x for x in re.split(r"(?<=[.])\s+", full) if r["expected_fact"] in x)
        doc = [c["text"] for c in zero.chunks if c["source"] == r["expected_doc"]]
        for a, b in zip(doc, doc[1:]):
            if sentence in a + " " + b and sentence not in a and sentence not in b:
                rep(f"   the sentence is cut at a chunk boundary: '...{' '.join(a.split()[-8:])}' | "
                    f"'{' '.join(b.split()[:8])}...'")
    rep.save()


if __name__ == "__main__":
    main()
