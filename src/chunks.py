from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TA_MAX_CHUNKS = 50  #for TA quick test (CPU-friendly)
MAX_PAGES = 10  # for TA quick test

def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Read PDF and return a list of dicts:
    - page: 1-based page number
    - text: raw page text
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        if MAX_PAGES and i >= MAX_PAGES:
            break
        pages.append({
            "page": i + 1,
            "text": page.extract_text() or ""
        })
    return pages


def simple_tag_from_text(text: str) -> str:
    """
    Very simple rule-based tagging.
    """
    t = text.lower()
    if "respiratory" in t or "nasal" in t:
        return "respiratory"
    if "gastro" in t or "vomit" in t or "diarrhea" in t:
        return "gastrointestinal"
    return "general"


def build_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split PDF pages into overlapping chunks with metadata:
    - doc_id (int)
    - page (int)
    - text (str)
    - tag (str)
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs: List[Dict[str, Any]] = []
    doc_id = 0
    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        if not page_text.strip():
            continue
        for chunk in splitter.split_text(page_text):
            docs.append({
                "doc_id": doc_id,
                "page": page_num,
                "text": chunk,
                "tag": simple_tag_from_text(chunk),
            })
            doc_id += 1
            if TA_MAX_CHUNKS and len(docs) >= TA_MAX_CHUNKS:
                return docs

    return docs
