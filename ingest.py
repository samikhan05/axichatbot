"""
ingest.py — Company knowledge base ingestion for hybrid RAG.

Supports:
- Markdown (.md): header-aware sections, structure-aware chunking
  (paragraphs / list items / sentences), tables split into row groups
  with the header row repeated in every group.
- PDF (.pdf): page-aware chunking, tables extracted separately and their
  area removed from the page text (no duplicated table text), image
  placeholders.

Chunking techniques used:
1. Contextual chunk headers — every chunk is embedded with
   "Document: <title> | Section: <h1 > h2 > h3>" prepended, so a chunk
   like "20 Days | Yes" still carries the meaning "Leave Allocation".
2. Sentence-level overlap — the next chunk repeats the last whole
   sentence(s) of the previous chunk (never a word cut in half).
3. Table row-grouping — large tables are split into groups of rows, each
   with the header row, so a question about one row retrieves a small,
   focused chunk instead of the whole table.

Every chunk gets two vectors in Qdrant:
- "dense":  BGE-M3 embedding (semantic, multilingual incl. Urdu script)
- "sparse": BM25-style keyword vector (exact names, emails, acronyms)
which rag_engine.py fuses with Reciprocal Rank Fusion (hybrid search).

Re-run after changing documents:
    python ingest.py
"""

import glob
import os
import re
import uuid

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, SparseVectorParams, Modifier,
)

from sparse_encoder import encode_document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "company_knowledge"
DOCS_DIR = "sample_docs"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

CHUNK_SIZE = 600        # target characters per chunk (body, without header)
CHUNK_OVERLAP = 150     # max characters of whole sentences carried forward
TABLE_ROWS_PER_CHUNK = 4

assert CHUNK_OVERLAP < CHUNK_SIZE, "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Text cleanup + structure-aware sentence splitting
# ---------------------------------------------------------------------------

# Sentence ends: English . ! ? and Urdu full stop ۔ / question mark ؟
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?۔؟])\s+(?=\S)")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def clean_markdown_inline(text: str) -> str:
    """Remove markdown syntax that adds noise to embeddings and BM25."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # bold
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)  # italics
    text = re.sub(r"`([^`]*)`", r"\1", text)           # inline code
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> text
    text = text.replace("\\|", "|")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def split_into_units(text: str, max_unit: int = CHUNK_SIZE) -> list[str]:
    """Split prose into sentence-like units, respecting document structure:
    each list item / blockquote line / paragraph is its own block first,
    then blocks are split into sentences. This stops "1." in a numbered
    list from being treated as a sentence and keeps list items whole."""
    units = []
    paragraph = []

    def flush_paragraph():
        if paragraph:
            block = clean_markdown_inline(" ".join(paragraph))
            if len(block) <= max_unit:
                # a short paragraph (e.g. one FAQ question + answer) stays
                # together as a single unit so it is never split apart
                units.append(block)
            else:
                units.extend(s.strip() for s in SENTENCE_SPLIT_RE.split(block) if s.strip())
            paragraph.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or HR_RE.match(stripped):
            flush_paragraph()
            continue
        if LIST_MARKER_RE.match(stripped) or stripped.startswith(">"):
            flush_paragraph()
            item = LIST_MARKER_RE.sub("", stripped).lstrip("> ").strip()
            if item:
                units.append(clean_markdown_inline(item))
            continue
        paragraph.append(stripped)
    flush_paragraph()
    return [u for u in units if u]


def _hard_split(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an over-long sentence on word boundaries with overlap."""
    words = sentence.split()
    pieces, current = [], []
    for word in words:
        if current and len(" ".join(current + [word])) > chunk_size:
            pieces.append(" ".join(current))
            carry = []
            while current and len(" ".join([current[-1]] + carry)) <= overlap:
                carry.insert(0, current.pop())
            current = carry
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_units(units: list[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Pack sentence units into chunks of up to chunk_size characters.
    The next chunk starts with the trailing whole sentence(s) of the
    previous one (up to `overlap` chars) so facts at a boundary are
    readable from both sides — without cutting words in half."""
    expanded = []
    for unit in units:
        expanded.extend(_hard_split(unit, chunk_size, overlap) if len(unit) > chunk_size else [unit])

    chunks = []
    current: list[str] = []
    for unit in expanded:
        if current and len(" ".join(current + [unit])) > chunk_size:
            chunks.append(" ".join(current))
            carry = []
            for prev in reversed(current):
                if len(" ".join([prev] + carry)) > overlap:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return chunk_units(split_into_units(text, chunk_size), chunk_size, overlap)


# ---------------------------------------------------------------------------
# Markdown ingestion — header-aware, tables row-grouped
# ---------------------------------------------------------------------------

def split_markdown_into_sections(content):
    """Split a markdown file into (header_path, body_text) sections based
    on # ## ### headers, so each chunk can carry its section context."""
    sections = []
    header_stack = []
    buffer = []
    in_code = False

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(header_stack), body))

    for line in content.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = clean_markdown_inline(match.group(2).strip())
            header_stack = header_stack[: level - 1] + [title]
        else:
            buffer.append(line)

    flush()
    return sections


def extract_tables_from_text(body):
    """Pull markdown tables (lines starting with |) out as separate blocks
    so they never get split mid-row. Returns (remaining_text, tables)."""
    tables, remaining, table_buffer = [], [], []

    for line in body.split("\n"):
        if line.strip().startswith("|"):
            table_buffer.append(line)
        else:
            if table_buffer:
                tables.append(list(table_buffer))
                table_buffer.clear()
            remaining.append(line)
    if table_buffer:
        tables.append(list(table_buffer))

    return "\n".join(remaining), tables


def _parse_table_row(line: str) -> list[str]:
    # split on unescaped pipes only
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [clean_markdown_inline(c) for c in cells]


def chunk_markdown_table(table_lines, rows_per_chunk=TABLE_ROWS_PER_CHUNK):
    """Split a markdown table into row groups, each starting with the
    header row so every chunk is self-describing."""
    rows = [l for l in table_lines if not re.match(r"^\s*\|?\s*:?-{3,}", l)]
    if not rows:
        return []
    header = _parse_table_row(rows[0])
    body = [_parse_table_row(r) for r in rows[1:]]

    def render(row_group):
        lines = [" | ".join(header)]
        lines += [" | ".join(r) for r in row_group]
        return "\n".join(lines)

    if not body:
        return [render([])]
    return [render(body[i:i + rows_per_chunk]) for i in range(0, len(body), rows_per_chunk)]


def chunk_rows_table(rows, rows_per_chunk=TABLE_ROWS_PER_CHUNK):
    """Same row-grouping for tables coming from pdfplumber (list of lists)."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r and any(r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [" | ".join(header)]
    return [
        "\n".join([" | ".join(header)] + [" | ".join(r) for r in body[i:i + rows_per_chunk]])
        for i in range(0, len(body), rows_per_chunk)
    ]


def contextual_header(doc_title: str, section: str) -> str:
    # the H1 is the document title — don't repeat it inside the section path
    if section.startswith(doc_title + " > "):
        section = section[len(doc_title) + 3:]
    if section and section != doc_title:
        return f"Document: {doc_title} | Section: {section}"
    return f"Document: {doc_title}"


def ingest_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    doc_title = clean_markdown_inline(title_match.group(1)) if title_match else os.path.basename(filepath)

    records = []
    for header_path, body in split_markdown_into_sections(content):
        prose, tables = extract_tables_from_text(body)
        ctx = contextual_header(doc_title, header_path)

        for chunk in chunk_text(prose):
            records.append({
                "text": f"{ctx}\n{chunk}",
                "body": chunk,
                "source": os.path.basename(filepath),
                "section": header_path,
                "type": "text",
            })

        for table in tables:
            for table_chunk in chunk_markdown_table(table):
                records.append({
                    "text": f"{ctx}\n{table_chunk}",
                    "body": table_chunk,
                    "source": os.path.basename(filepath),
                    "section": header_path,
                    "type": "table",
                })

    return records


# ---------------------------------------------------------------------------
# PDF ingestion — page-aware, tables extracted separately, image-aware
# ---------------------------------------------------------------------------

def ingest_pdf(filepath):
    records = []
    source = os.path.basename(filepath)
    doc_title = os.path.splitext(source)[0].replace("_", " ")

    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            section = f"page {page_number}"
            ctx = contextual_header(doc_title, section)

            # Remove table areas from the page before extracting prose so
            # table cells aren't embedded twice (once as garbled prose).
            found_tables = page.find_tables()
            text_page = page
            for t in found_tables:
                try:
                    text_page = text_page.outside_bbox(t.bbox)
                except ValueError:
                    pass

            raw_text = text_page.extract_text() or ""
            # join hyphenated line breaks, then turn single line breaks into
            # spaces but keep blank lines as paragraph boundaries
            text = raw_text.replace("-\n", "")
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            text = re.sub(r"[ \t]{2,}", " ", text)

            for chunk in chunk_text(text):
                records.append({
                    "text": f"{ctx}\n{chunk}", "body": chunk,
                    "source": source, "section": section, "type": "text",
                })

            for t in found_tables:
                for table_chunk in chunk_rows_table(t.extract() or []):
                    records.append({
                        "text": f"{ctx}\n{table_chunk}", "body": table_chunk,
                        "source": source, "section": section, "type": "table",
                    })

            # Placeholder chunks for images. This flags that visual content
            # exists on this page so it isn't silently dropped, but it
            # carries no actual visual content. Full support needs OCR or a
            # vision-model captioning step.
            for img_index, _ in enumerate(page.images):
                placeholder = f"[Image {img_index + 1} on {source} page {page_number}]"
                records.append({
                    "text": f"{ctx}\n{placeholder}", "body": placeholder,
                    "source": source, "section": section, "type": "image",
                })

    return records


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def collect_records():
    records = []
    for filepath in sorted(glob.glob(f"{DOCS_DIR}/**/*.md", recursive=True)):
        records.extend(ingest_markdown(filepath))
    for filepath in sorted(glob.glob(f"{DOCS_DIR}/**/*.pdf", recursive=True)):
        records.extend(ingest_pdf(filepath))
    return records


def make_point_id(record):
    # Content-addressed, deterministic UUID: re-ingesting unchanged content
    # yields the same ids, and editing one chunk doesn't shift the others.
    key = f"{record['source']}::{record['section']}::{record['text']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main():
    records = collect_records()
    if not records:
        print(f"No .md or .pdf files found in {DOCS_DIR}")
        return

    # connect first, so we fail fast before spending time on embeddings
    client = QdrantClient(url=QDRANT_URL)
    try:
        exists = client.collection_exists(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(
            f"\nCannot reach Qdrant at {QDRANT_URL} ({e.__class__.__name__}).\n"
            "Start Docker Desktop, then run:  docker compose up -d\n"
            "and run this script again."
        )
        raise SystemExit(1)

    texts = [r["text"] for r in records]
    embeddings = get_model().encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    if exists:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={DENSE_VECTOR: VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
    )

    points = [
        PointStruct(
            id=make_point_id(r),
            vector={
                DENSE_VECTOR: embeddings[i].tolist(),
                SPARSE_VECTOR: encode_document(r["text"]),
            },
            payload={
                "text": r["text"],
                "body": r["body"],
                "source": r["source"],
                "section": r["section"],
                "type": r["type"],
            },
        )
        for i, r in enumerate(records)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    by_type = {}
    for r in records:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"Uploaded {len(records)} chunks to Qdrant ({by_type}) with dense + sparse vectors.")


if __name__ == "__main__":
    main()
