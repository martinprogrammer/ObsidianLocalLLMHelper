"""
Local embedding engine using sentence-transformers (all-MiniLM-L6-v2).
No internet required after first download. Runs on CPU, ~90MB model.
"""
from typing import List
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> List[float]:
    """Embed a single string. Returns a flat list of floats."""
    model = get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def embed_batch(texts: List[str], batch_size: int = 32, progress_callback=None) -> List[List[float]]:
    """
    Embed a list of strings. Returns list of float vectors.
    progress_callback(current, total) called in batches.
    """
    model = get_model()
    results = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start: start + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        results.extend(vecs.tolist())
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    return results


def note_to_embed_text(note) -> str:
    """Produce embedding input: title + tags + first 500 chars of content."""
    tag_str = " ".join(note.tags[:10])
    snippet = note.content[:500].replace("\n", " ")
    return f"{note.title}. Tags: {tag_str}. {snippet}"
