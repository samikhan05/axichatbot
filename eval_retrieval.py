"""
eval_retrieval.py — Measure retrieval quality instead of guessing.

Runs a small labelled question set against the Qdrant collection and
compares three retrieval strategies:
    1. dense            (BGE-M3 only — the original pipeline)
    2. hybrid           (dense + BM25 sparse, fused with RRF)
    3. hybrid + rerank  (cross-encoder bge-reranker-v2-m3 on top)

Metrics:
    Hit@k  — share of questions where a correct chunk is in the top k
    MRR    — mean reciprocal rank of the first correct chunk
    Latency — average search time per question

The set includes English and Urdu-script questions. (Roman Urdu
questions are translated to English by the LLM before searching, see the
system prompt in rag_engine.py, so retrieval sees English for those.)

Usage (Qdrant running, `python ingest.py` already done):
    python eval_retrieval.py
    python eval_retrieval.py --no-rerank      # skip the reranker download
"""

import argparse
import time

from rag_engine import hybrid_search

K = 3

# (question, substring that must appear in a correct chunk)
GOLDEN_SET = [
    ("Where is the head office located?", "Centaurus Business Tower"),
    ("What is the support email address?", "info@axitech.com"),
    ("What are the office working hours?", "09:00 AM"),
    ("When is the Friday prayer break?", "02:30 PM"),
    ("Who leads the Finance department?", "Kamran Mirza"),
    ("Who is the head of cybersecurity?", "Bilal Hassan"),
    ("How many people work in Marketing & Sales?", "Zara Sheikh"),
    ("Which client is the Smart Fleet Manager built for?", "LogiPak"),
    ("What tech stack does the E-Learning Platform use?", "PyTorch"),
    ("How many days of annual paid vacation do employees get?", "20 Days"),
    ("Is a medical certificate needed for sick leave?", "Medical Certificate"),
    ("How often must passwords be rotated?", "90 days"),
    ("Is MFA mandatory?", "Multi-Factor Authentication"),
    ("How long is the probation period for new hires?", "3-month probation"),
    ("When was AXI Technologies founded?", "2019"),
    ("What does the inventory platform for RetailCo do?", "200 physical branches"),
    # exact-keyword lookups — where BM25 sparse search helps most
    ("+92-51-1234567", "+92-51-1234567"),
    ("Celery", "Celery"),
    ("Rhubarb phoneme", "Rhubarb"),
    ("Kamran Mirza", "Kamran Mirza"),
    ("intimation before 10:00 AM", "Intimation"),
    ("VPN tunnel", "VPN"),
    ("Dr. Nida Hussain", "Nida Hussain"),
    ("Google Maps", "Google Maps"),
    # Urdu script — BGE-M3 is cross-lingual, so these should still hit
    ("ہیڈ آفس کہاں واقع ہے؟", "Centaurus Business Tower"),
    ("فنانس ڈیپارٹمنٹ کا سربراہ کون ہے؟", "Kamran Mirza"),
    ("سالانہ چھٹیاں کتنی ہیں؟", "20 Days"),
    ("پاس ورڈ کتنے دن بعد تبدیل کرنا ہوتا ہے؟", "90 days"),
]


def evaluate(name, **search_kwargs):
    hits, rr, latency = 0, 0.0, 0.0
    misses = []
    for question, expected in GOLDEN_SET:
        t0 = time.perf_counter()
        results = hybrid_search(question, top_k=K, **search_kwargs)
        latency += time.perf_counter() - t0

        rank = next((i + 1 for i, r in enumerate(results) if expected.lower() in r["text"].lower()), None)
        if rank:
            hits += 1
            rr += 1.0 / rank
        else:
            misses.append(question)

    n = len(GOLDEN_SET)
    print(f"{name:<18} Hit@{K}: {hits / n:6.1%}   MRR: {rr / n:.3f}   avg latency: {latency / n * 1000:6.0f} ms")
    for q in misses:
        print(f"{'':<18}   miss: {q}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder comparison")
    args = parser.parse_args()

    hybrid_search("warm up", top_k=1, mode="dense", rerank=False)  # load models first

    print(f"\n{len(GOLDEN_SET)} questions, top-{K}\n")
    evaluate("dense", mode="dense", rerank=False)
    evaluate("hybrid (RRF)", mode="hybrid", rerank=False)
    if not args.no_rerank:
        evaluate("hybrid + rerank", mode="hybrid", rerank=True)


if __name__ == "__main__":
    main()
