import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import create_embeddings
from .loader import POLICY_DIR, SUPPORTED, load_policy
from .splitter import split_policy

DB_DIR = Path(os.getenv("POLICY_CHROMA_DIR", str(POLICY_DIR.parent / "chroma")))


def open_store(embedding_function=None, persist_directory: Path | None = None):
    return Chroma(collection_name="policy_documents",
                  embedding_function=embedding_function or create_embeddings(),
                  persist_directory=str(persist_directory or DB_DIR),
                  collection_metadata={"hnsw:space": "cosine"})


def ingest_directory(base_dir: Path = POLICY_DIR, store=None) -> dict:
    base_dir = base_dir.resolve()
    store = store or open_store()
    added = skipped = 0
    for domain in ("academic", "corporate"):
        for path in sorted((base_dir / domain).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            pages = load_policy(path, base_dir)
            if not pages:
                continue
            metadata = pages[0].metadata
            existing = store.get(where={"document_id": metadata["document_id"]},
                                 include=["metadatas"])
            if existing["ids"] and all(m.get("content_hash") == metadata["content_hash"]
                                       for m in existing["metadatas"]):
                skipped += 1
                continue
            chunks = split_policy(pages)
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = index
            ids = [hashlib.sha256(f"{metadata['document_id']}:{i}".encode()).hexdigest()
                   for i in range(len(chunks))]
            # Embed before deleting old chunks; a failed embedding keeps existing data.
            if chunks:
                vectors = store._embedding_function.embed_documents([c.page_content for c in chunks])
                if existing["ids"]:
                    store.delete(ids=existing["ids"])
                store._collection.upsert(ids=ids,
                                         embeddings=vectors,
                                         documents=[c.page_content for c in chunks],
                                         metadatas=[c.metadata for c in chunks])
                added += 1
    return {"added": added, "skipped": skipped}
