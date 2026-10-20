from .vectorstore import DB_DIR, open_store


def search_policies(query: str, domain: str = "academic", top_k: int = 3, store=None) -> list[dict]:
    if domain not in {"academic", "corporate"}:
        raise ValueError("domain must be academic or corporate")
    if not query.strip():
        return []
    if store is None and not (DB_DIR / "chroma.sqlite3").exists():
        return []
    store = store or open_store()
    matches = store.similarity_search_with_score(
        query, k=max(5, min(top_k * 3, 20)), filter={"domain": domain})
    ranked = [(doc, distance) for doc, distance in matches if distance <= 0.75]
    # If the same numbered and named article exists in different versions, keep
    # the newer one first while retaining the older text for conflict review.
    for index, (doc, _) in enumerate(list(ranked)):
        article = doc.metadata.get("article")
        if not article:
            continue
        group = [j for j, (other, _) in enumerate(ranked)
                 if other.metadata.get("article") == article]
        if len(group) > 1:
            ordered = sorted((ranked[j] for j in group),
                             key=lambda pair: pair[0].metadata.get("effective_date", ""),
                             reverse=True)
            for position, value in zip(group, ordered):
                ranked[position] = value
    return [{"id": f"{doc.metadata['document_id']}:{doc.metadata.get('page', 0)}:{i}",
             "text": doc.page_content, "score": round(max(0, min(1, 1 - float(distance))), 4),
             **doc.metadata} for i, (doc, distance) in enumerate(ranked[:max(1, min(top_k, 10))])]


def get_document(document_id: str, store=None) -> dict:
    if store is None and not (DB_DIR / "chroma.sqlite3").exists():
        raise KeyError(document_id)
    store = store or open_store()
    data = store.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    if not data["ids"]:
        raise KeyError(document_id)
    chunks = [{"text": text, **meta} for text, meta in zip(data["documents"], data["metadatas"])]
    return {"document_id": document_id,
            "chunks": sorted(chunks, key=lambda x: (x.get("page", 0), x.get("chunk_index", 0)))}
