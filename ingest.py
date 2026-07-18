import glob
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

model = SentenceTransformer("BAAI/bge-m3")

MAX_CHUNK_LENGTH = 500  # characters — prevents runaway merging


def is_header_like(paragraph):
    lines = paragraph.split("\n")
    return len(lines) == 1 and len(paragraph) < 60 and not paragraph.endswith((".", "!", "?"))


def smart_chunk(content):
    raw_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        # Safety valve: never let the buffer grow unbounded, even across "header-like" lines
        if len(buffer) > MAX_CHUNK_LENGTH:
            chunks.append(buffer.strip())
            buffer = ""

        if is_header_like(para):
            buffer += para + "\n"
        else:
            chunks.append((buffer + para).strip())
            buffer = ""

    if buffer:
        chunks.append(buffer.strip())

    return chunks


chunks = []
for filepath in glob.glob("sample_docs/*.md"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    chunks.extend(smart_chunk(content))

client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="company_knowledge",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

embeddings = model.encode(chunks)
points = [
    PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]})
    for i in range(len(chunks))
]
client.upsert(collection_name="company_knowledge", points=points)

print(f"Uploaded {len(chunks)} chunks to Qdrant from {len(glob.glob('sample_docs/*.md'))} files.")