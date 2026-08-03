"""
ingest.py — Company knowledge base ingestion for RAG.

Supports:
- Markdown (.md) files: header-aware chunking, tables kept atomic.
- PDF (.pdf) files: page-aware chunking, tables extracted separately, image placeholders.

Every chunk is embedded with BGE-M3 and stored in Qdrant with metadata
(source file, section/page, chunk type) so retrieval results can be
traced back to exactly where they came from.

Install the new dependency before running:
    pip install pdfplumber
"""

import glob
import re
import hashlib

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "company_knowledge"
DOCS_DIR = "sample_docs"

CHUNK_SIZE = 500       # target characters per chunk
CHUNK_OVERLAP = 100    # characters carried into the next chunk for context

assert CHUNK_OVERLAP < CHUNK_SIZE, "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"

model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Generic sentence-aware chunker with overlap
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Pack sentences into chunks up to chunk_size chars. Each new chunk
    carries the trailing `overlap` chars of the previous chunk forward,
    so a fact split across a boundary is still readable from both sides.
    Sentences longer than chunk_size (e.g. unpunctuated PDF text, URLs)
    are hard-split into overlapping windows so no chunk ever exceeds
    chunk_size."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current.strip():
                chunks.append(current.strip())

            for i in range(0, len(sentence), chunk_size - overlap):
                chunks.append(sentence[i:i + chunk_size].strip())

            current = chunks[-1][-overlap:] if chunks else ""
            continue

        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap < len(current) else current
            current = (tail + " " + sentence).strip()
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------------
# Markdown ingestion — header-aware, tables kept atomic
# ---------------------------------------------------------------------------

def split_markdown_into_sections(content):
    """Split a markdown file into (header_path, body_text) sections based
    on # ## ### headers, so each chunk can carry its section context."""
    lines = content.split("\n")
    sections = []
    header_stack = []
    buffer = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(header_stack), body))

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            header_stack = header_stack[: level - 1] + [title]
        else:
            buffer.append(line)

    flush()
    return sections


def extract_tables_from_text(body):
    """Pull markdown tables (lines starting with |) out as separate atomic
    blocks so they never get split mid-row. Returns (remaining_text, tables)."""
    lines = body.split("\n")
    tables = []
    remaining = []
    table_buffer = []

    def flush_table():
        if table_buffer:
            tables.append("\n".join(table_buffer))

    for line in lines:
        if line.strip().startswith("|"):
            table_buffer.append(line)
        else:
            if table_buffer:
                flush_table()
                table_buffer.clear()
            remaining.append(line)
    flush_table()

    return "\n".join(remaining), tables


def ingest_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    records = []
    for header_path, body in split_markdown_into_sections(content):
        prose, tables = extract_tables_from_text(body)

        for chunk in chunk_text(prose):
            prefixed = f"{header_path}\n{chunk}" if header_path else chunk
            records.append({
                "text": prefixed,
                "source": filepath,
                "section": header_path,
                "type": "text",
            })

        # tables are never split by chunk_text — one atomic chunk each
        for table in tables:
            prefixed = f"{header_path}\n{table}" if header_path else table
            records.append({
                "text": prefixed,
                "source": filepath,
                "section": header_path,
                "type": "table",
            })

    return records


# ---------------------------------------------------------------------------
# PDF ingestion — page-aware, tables extracted separately, image-aware
# ---------------------------------------------------------------------------

def ingest_pdf(filepath):
    records = []
    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            # Clean visual line breaks / hyphenation so sentence splitting
            # (which relies on ". ! ?") actually works on PDF text.
            raw_text = page.extract_text() or ""
            text = raw_text.replace("-\n", "").replace("\n", " ")
            text = re.sub(r"\s{2,}", " ", text)  # compress multiple spaces

            tables = page.extract_tables() or []

            for chunk in chunk_text(text):
                records.append({
                    "text": chunk,
                    "source": filepath,
                    "section": f"page {page_number}",
                    "type": "text",
                })

            for table in tables:
                # render as pipe-separated rows so the LLM can read it
                # the same way it reads a markdown table
                table_text = "\n".join(
                    " | ".join(cell or "" for cell in row) for row in table
                )
                if table_text.strip():
                    records.append({
                        "text": table_text,
                        "source": filepath,
                        "section": f"page {page_number}",
                        "type": "table",
                    })

            # Placeholder chunks for images. This flags that visual content
            # exists on this page so it isn't silently dropped, but it
            # carries no actual visual content — a question about what an
            # image *shows* won't retrieve this chunk. Full support needs
            # OCR (pytesseract) or a vision-model captioning step.
            for img_index, img in enumerate(page.images):
                image_placeholder = (
                    f"[Visual element/Image {img_index + 1} located on "
                    f"{filepath} Page {page_number}]"
                )
                records.append({
                    "text": image_placeholder,
                    "source": filepath,
                    "section": f"page {page_number}",
                    "type": "image",
                })

    return records


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def collect_records():
    records = []
    for filepath in glob.glob(f"{DOCS_DIR}/**/*.md", recursive=True):
        records.extend(ingest_markdown(filepath))
    for filepath in glob.glob(f"{DOCS_DIR}/**/*.pdf", recursive=True):
        records.extend(ingest_pdf(filepath))
    return records


def make_point_id(record, index):
    # deterministic ID so re-running ingest on the same content doesn't
    # create duplicate/drifting IDs across runs
    key = f"{record['source']}::{record['section']}::{index}"
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16) % (2**63)


def main():
    records = collect_records()
    if not records:
        print(f"No .md or .pdf files found in {DOCS_DIR}")
        return

    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    client = QdrantClient(url="http://localhost:6333")
    
    # Check if the collection exists and delete it to mimic 'recreate' behavior
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Create the new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=make_point_id(records[i], i),
            vector=embeddings[i].tolist(),
            payload={
                "text": records[i]["text"],
                "source": records[i]["source"],
                "section": records[i]["section"],
                "type": records[i]["type"],
            },
        )
        for i in range(len(records))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    md_count = len(glob.glob(f"{DOCS_DIR}/**/*.md", recursive=True))
    pdf_count = len(glob.glob(f"{DOCS_DIR}/**/*.pdf", recursive=True))
    print(
        f"Uploaded {len(records)} chunks to Qdrant from "
        f"{md_count} markdown and {pdf_count} PDF files."
    )


if __name__ == "__main__":
    main()