import re

from .vectorstore import DB_DIR, open_store

STOPWORDS = {"어떤", "경우", "무엇", "뭐가", "뭔가", "필요", "서류", "관련", "규정",
             "알려줘", "있나요", "있어", "어떻게", "하려면", "해야", "제출", "신청",
             "필요해", "무슨", "그", "수", "받을", "혹시", "학칙", "학사",
             "찾아줘", "부득이하게", "필요한"}
MIN_RELEVANCE = 0.30


def _terms(query: str) -> list[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", query.lower())
    terms = []
    for word in words:
        word = re.sub(r"(?:하려면|하는지|할까요|인가요|했는데|으로|에서|은|는|이|가|을|를|에|의|로|와|과)$", "", word)
        if len(word) >= 2 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms


def _lexical_score(query_terms: list[str], text: str, article: str) -> float:
    if not query_terms:
        return 0.0
    compact = re.sub(r"\s+", "", text.lower())
    title = re.sub(r"\s+", "", article.lower())
    weights = [1.5 if index == 0 else 1.0 for index in range(len(query_terms))]
    matched = sum(weight for term, weight in zip(query_terms, weights) if term in compact)
    score = matched / sum(weights)
    if any(term in title for term in query_terms):
        score = min(1.0, score + 0.12)
    return score


def _subject_filter(query: str) -> dict | None:
    """Keep closely named programs apart when the question names only one."""
    if "부전공" in query and "복수전공" not in query:
        return {"$contains": "부전공"}
    if "복수전공" in query and "부전공" not in query:
        return {"$contains": "복수전공"}
    return None


def _document_filter(query: str) -> dict | None:
    subject = _subject_filter(query)
    if subject:
        return subject
    terms = _terms(query)
    return {"$contains": terms[0]} if terms else None


def search_policies(query: str, domain: str = "academic", top_k: int = 3, store=None) -> list[dict]:
    if domain not in {"academic", "corporate"}:
        raise ValueError("domain must be academic or corporate")
    if not query.strip():
        return []
    if store is None and not (DB_DIR / "chroma.sqlite3").exists():
        return []
    store = store or open_store()
    document_filter = _document_filter(query)
    matches = store.similarity_search_with_score(
        query, k=max(20, min(top_k * 6, 40)), filter={"domain": domain},
        where_document=document_filter)
    # A paraphrase can lack the source's exact noun. In that case retain only
    # a strong semantic match from an unrestricted search.
    if not matches and document_filter:
        matches = [(doc, distance) for doc, distance in store.similarity_search_with_score(
            query, k=max(20, min(top_k * 6, 40)), filter={"domain": domain})
            if distance <= 0.35]
    terms = _terms(query)
    compact_texts = [re.sub(r"\s+", "", doc.page_content.lower()) for doc, _ in matches]
    anchor = next((term for term in terms if len(term) >= 3
                   and any(term in text for text in compact_texts)), None)
    ranked = []
    for doc, distance in matches:
        similarity = max(0.0, 1.0 - float(distance))
        if distance > 0.70:
            continue
        if anchor and anchor not in re.sub(r"\s+", "", doc.page_content.lower()):
            continue
        lexical = _lexical_score(terms, doc.page_content, doc.metadata.get("article", ""))
        # Exact topic/action overlap must accompany a marginal vector match.
        if similarity < 0.34 and lexical < 0.5:
            continue
        relevance = 0.62 * similarity + 0.38 * lexical
        title_match = re.search(r"\(([^)]{2,})\)", doc.metadata.get("article", ""))
        if title_match:
            title = re.sub(r"\s+", "", title_match.group(1).lower())
            if sum(term in title for term in terms) >= 2:
                relevance += 0.15
        if relevance < MIN_RELEVANCE:
            continue
        ranked.append((relevance, similarity, doc))
    ranked.sort(key=lambda row: row[0], reverse=True)
    selected = []
    seen = set()
    for relevance, similarity, doc in ranked:
        key = (doc.metadata.get("document_id"), doc.metadata.get("page"),
               doc.metadata.get("chunk_index", doc.page_content))
        if key in seen:
            continue
        seen.add(key)
        selected.append({"id": f"{doc.metadata['document_id']}:{doc.metadata.get('page', 0)}:{len(selected)}",
                         "text": doc.page_content, "score": round(similarity, 4),
                         "relevance": round(relevance, 4), **doc.metadata})
        if len(selected) >= max(1, min(top_k, 10)):
            break
    return selected


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
