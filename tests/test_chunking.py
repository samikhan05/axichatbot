from ingest import (
    chunk_text, chunk_markdown_table, split_into_units, contextual_header,
)
from sparse_encoder import encode_query, encode_document, tokenize


def test_chunks_respect_size_and_never_cut_words():
    text = " ".join(f"Sentence number {i} talks about the office policy." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200
        assert c.startswith("Sentence")  # starts on a sentence boundary
        assert c.endswith(".")


def test_overlap_repeats_whole_previous_sentence():
    text = " ".join(f"Fact {i} is here." for i in range(40))
    chunks = chunk_text(text, chunk_size=100, overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.split(". ")[-1]
        assert last_sentence.rstrip(".") in nxt


def test_numbered_list_items_stay_whole():
    units = split_into_units("1. **MFA:** Mandatory everywhere.\n2. **VPN:** Required off-site.")
    assert units == ["MFA: Mandatory everywhere.", "VPN: Required off-site."]


def test_urdu_full_stop_splits_sentences():
    long = "یہ پہلا جملہ ہے۔ " * 40
    chunks = chunk_text(long, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)


def test_table_row_groups_repeat_header():
    table = [
        "| Name | Head |",
        "| :--- | :--- |",
        *[f"| Dept {i} | Person {i} |" for i in range(10)],
    ]
    chunks = chunk_markdown_table(table, rows_per_chunk=4)
    assert len(chunks) == 3
    assert all(c.startswith("Name | Head") for c in chunks)


def test_contextual_header_drops_repeated_title():
    assert contextual_header("Handbook", "Handbook > Leave") == "Document: Handbook | Section: Leave"


def test_sparse_vectors_match_exact_tokens():
    tokens = tokenize("Email: ahmed.raza@axitech.com")
    assert "ahmed" in tokens and "axitech" in tokens
    q = encode_query("ahmed raza email")
    d = encode_document("Dr. Ahmed Raza, email ahmed.raza@axitech.com")
    assert set(q.indices) <= set(d.indices)
