import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import create_embeddings
from .loader import POLICY_DIR, SUPPORTED, load_policy
from .splitter import split_policy

DB_DIR = Path(os.getenv("POLICY_CHROMA_DIR", str(POLICY_DIR.parent / "chroma")))
DOMAINS = ("academic", "corporate")
INDEX_PROFILE = "policy-articles-v3"


def _indexable_chunks(path: Path, pages) -> list:
    """Exclude course catalogs and other long, unstructured PDF appendices.

    Short standalone notices remain searchable even without article headings.
    """
    chunks = split_policy(pages)
    if path.suffix.lower() != ".pdf" or len(chunks) < 50:
        selected = chunks
    else:
        selected = [chunk for chunk in chunks if chunk.metadata.get("article")]
        if len(selected) / len(chunks) < 0.1:
            selected = []
    for index, chunk in enumerate(selected):
        chunk.metadata.update(chunk_index=index, index_profile=INDEX_PROFILE)
    return selected


def open_store(embedding_function=None, persist_directory: Path | None = None):
    return Chroma(collection_name="policy_documents",
                  embedding_function=embedding_function or create_embeddings(),
                  persist_directory=str(persist_directory or DB_DIR),
                  collection_metadata={"hnsw:space": "cosine"})


def ingest_directory(base_dir: Path = POLICY_DIR, store=None) -> dict:
    base_dir = base_dir.resolve()
    store = store or open_store()
    added = skipped = 0
    for domain in DOMAINS:
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
                                       and m.get("index_profile") == INDEX_PROFILE
                                       for m in existing["metadatas"]):
                skipped += 1
                continue
            chunks = _indexable_chunks(path, pages)
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
            elif existing["ids"]:
                store.delete(ids=existing["ids"])
    return {"added": added, "skipped": skipped}


def reset_domain_index(domain: str, base_dir: Path = POLICY_DIR, store=None) -> dict:
    """Replace one domain with its current files, including removal of deleted files.

    Parse and embed every current document before clearing the existing domain so
    a missing source or embedding failure leaves the old index intact.
    """
    if domain not in DOMAINS:
        raise ValueError("domain must be academic or corporate")
    base_dir = base_dir.resolve()
    store = store or open_store()
    prepared = []
    for path in sorted((base_dir / domain).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        pages = load_policy(path, base_dir)
        if not pages:
            continue
        chunks = _indexable_chunks(path, pages)
        if not chunks:
            continue
        metadata = pages[0].metadata
        ids = [hashlib.sha256(f"{metadata['document_id']}:{i}".encode()).hexdigest()
               for i in range(len(chunks))]
        vectors = store._embedding_function.embed_documents([c.page_content for c in chunks])
        prepared.append((ids, vectors, chunks))
    if not prepared:
        raise ValueError(f"{base_dir / domain}에 색인할 수 있는 문서가 없습니다. 기존 색인은 유지했습니다.")

    existing = store.get(where={"domain": domain}, include=["metadatas"])["ids"]
    for start in range(0, len(existing), 1000):
        store.delete(ids=existing[start:start + 1000])
    for ids, vectors, chunks in prepared:
        store._collection.upsert(ids=ids, embeddings=vectors,
                                 documents=[c.page_content for c in chunks],
                                 metadatas=[c.metadata for c in chunks])
    return {"domain": domain, "added": len(prepared), "removed_chunks": len(existing)}
