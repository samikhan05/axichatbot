"""
sparse_encoder.py — Dependency-free BM25-style sparse vectors for Qdrant.

Dense embeddings (BGE-M3) are good at meaning ("who runs the AI team?")
but weak at exact tokens: names, emails, phone numbers, project codes,
"MFA", "VPN". A sparse keyword vector covers that gap. Combining both
with Reciprocal Rank Fusion is "hybrid search".

How it works:
- Tokens are hashed to stable 31-bit integer ids (no vocabulary file to
  keep in sync between ingest and query time).
- Document side stores a saturated term frequency, tf*(k1+1)/(tf+k1),
  so a word repeated 10 times doesn't dominate.
- Query side stores 1.0 per unique term.
- The collection is created with Modifier.IDF, so Qdrant computes the
  inverse document frequency itself, server-side, from the live corpus.
  Together that gives BM25 scoring (without length normalisation).
"""

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

K1 = 1.2

TOKEN_RE = re.compile(r"[\w@.+-]+", re.UNICODE)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "by", "at", "as", "it",
    "this", "that", "from", "what", "which", "who", "how", "do", "does",
    "can", "i", "you", "we", "our", "your", "me", "my", "about", "tell",
    "please", "there", "any", "all",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall((text or "").lower()):
        tok = raw.strip(".-+")
        if not tok or tok in STOPWORDS:
            continue
        tokens.append(tok)
        # also index the parts of compound tokens: "ahmed.raza@axitech.com"
        # -> "ahmed", "raza", "axitech", "com"; "+92-300-1234567" -> parts
        if any(c in tok for c in "@.-+"):
            tokens.extend(p for p in re.split(r"[@.+-]", tok) if p and p not in STOPWORDS)
    return tokens


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def _to_sparse(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[float(weights[i]) for i in indices])


def encode_document(text: str) -> SparseVector:
    counts = Counter(_token_id(t) for t in tokenize(text))
    return _to_sparse({i: tf * (K1 + 1) / (tf + K1) for i, tf in counts.items()})


def encode_query(text: str) -> SparseVector:
    return _to_sparse({_token_id(t): 1.0 for t in set(tokenize(text))})
