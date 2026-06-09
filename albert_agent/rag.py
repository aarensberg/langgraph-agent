"""RAG retriever over the student's official PDF documents.

This is the agent's window onto the *paper trail* the live API does not expose:
the enrollment certificate and the historical transcripts (which grade on the
French /20 scale with letter grades and ECTS, unlike the API's live /100 marks).

Pipeline, per the spec: ``PyPDFLoader`` -> ``RecursiveCharacterTextSplitter`` ->
``Chroma``. Embeddings are produced locally with ``FastEmbedEmbeddings`` (ONNX,
no API key — Groq has no embeddings endpoint). The Chroma store is persisted on
disk and the whole thing is memoised, so documents are embedded once, not on
every Streamlit rerun.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .observability import log_event

_COLLECTION = "albert_documents"


def _load_and_split() -> list:
    """Load every PDF in ``rag/`` and split it into retrievable chunks."""
    pdfs = sorted(config.RAG_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF documents found in {config.RAG_PDF_DIR}. "
            "Place the official PDFs there to enable the document search tool."
        )
    documents = []
    for pdf in pdfs:
        loaded = PyPDFLoader(str(pdf)).load()
        for doc in loaded:
            doc.metadata["source"] = pdf.name  # clean filename, not the full path
        documents.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _count(store: Chroma) -> int:
    try:
        return store._collection.count()
    except Exception:  # noqa: BLE001 - any failure means "treat as empty"
        return 0


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the Chroma store, loading it from disk or building it once.

    Memoised for the process: the first call may download the embedding model
    and embed the PDFs; every later call is free.
    """
    embeddings = FastEmbedEmbeddings(model_name=config.RAG_EMBED_MODEL)
    persist_dir = str(config.CHROMA_DIR)

    if config.CHROMA_DIR.exists():
        store = Chroma(
            collection_name=_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if _count(store) > 0:
            log_event("rag", status="loaded", chunks=_count(store))
            return store

    chunks = _load_and_split()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_COLLECTION,
        persist_directory=persist_dir,
    )
    log_event("rag", status="built", chunks=len(chunks))
    return store


def search_documents(query: str, k: int = config.RAG_TOP_K) -> list[dict]:
    """Return the ``k`` most relevant chunks as ``{source, page, score, content}``."""
    store = get_vectorstore()
    hits = store.similarity_search_with_score(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page"),
            "score": round(float(score), 4),
            "content": doc.page_content.strip(),
        }
        for doc, score in hits
    ]


def warmup() -> int:
    """Build/load the index ahead of time; return the chunk count. Used by the UI."""
    return _count(get_vectorstore())
