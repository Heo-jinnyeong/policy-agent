import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLE = re.compile(r"(?=^\s*(?:제\s*\d+\s*조(?:\s*\([^\n)]*\))?|Article\s+\d+\b))", re.M | re.I)
ARTICLE_NAME = re.compile(r"^\s*((?:제\s*\d+\s*조|Article\s+\d+)(?:\s*\([^\n)]*\))?)", re.I)


def split_policy(pages: list[Document], chunk_size: int = 900) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100,
                                               separators=["\n\n", "\n", "。", ". ", " ", ""])
    chunks = []
    for page in pages:
        parts = [part.strip() for part in ARTICLE.split(page.page_content) if part.strip()]
        for part in parts:
            match = ARTICLE_NAME.match(part)
            article = match.group(1).strip() if match else ""
            for chunk in splitter.split_text(part):
                chunks.append(Document(page_content=chunk,
                                       metadata={**page.metadata, "article": article}))
    return chunks
