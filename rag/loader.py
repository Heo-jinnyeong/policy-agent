import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
SUPPORTED = {".pdf", ".txt", ".md"}


def load_policy(path: Path, base_dir: Path = POLICY_DIR) -> list[Document]:
    path = path.resolve()
    base_dir = base_dir.resolve()
    relative = path.relative_to(base_dir)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    domain = relative.parts[0]
    if domain not in {"academic", "corporate"}:
        raise ValueError("Document must be inside academic/ or corporate/")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    document_id = hashlib.sha256(str(relative).encode()).hexdigest()[:20]
    metadata = {"document_id": document_id, "content_hash": digest,
                "document_name": path.name, "source": str(relative),
                "domain": domain}
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [Document(page_content=page.extract_text() or "",
                          metadata={**metadata, "page": index + 1})
                 for index, page in enumerate(reader.pages)]
    else:
        pages = [Document(page_content=raw.decode("utf-8-sig"),
                          metadata={**metadata, "page": 0})]
    effective = re.search(r"(?:시행일|effective date)\s*[:：]\s*([^\n]+)",
                          pages[0].page_content, flags=re.I) if pages else None
    for page in pages:
        page.metadata["effective_date"] = effective.group(1).strip() if effective else ""
    return [page for page in pages if page.page_content.strip()]
