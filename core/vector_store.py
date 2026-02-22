"""
Local vector store using hnswlib + JSON metadata.
No server required, persists to disk, works on Python 3.14+.
"""
import os
import json
import pickle
import numpy as np
import hnswlib
from typing import List, Dict, Optional

from config import CACHE_DIR

INDEX_FILE = os.path.join(CACHE_DIR, "hnsw.bin")
META_FILE = os.path.join(CACHE_DIR, "metadata.json")
DIM = 384  # all-MiniLM-L6-v2 dimension


def _load_meta() -> Dict:
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {"id_to_int": {}, "int_to_meta": {}, "next_int": 0}


def _save_meta(meta: Dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def _load_index(max_elements: int = 50000) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=DIM)
    if os.path.exists(INDEX_FILE):
        index.load_index(INDEX_FILE, max_elements=max_elements)
    else:
        index.init_index(max_elements=max_elements, ef_construction=200, M=16)
    index.set_ef(50)
    return index


def _save_index(index: hnswlib.Index):
    os.makedirs(CACHE_DIR, exist_ok=True)
    index.save_index(INDEX_FILE)


def get_indexed_ids() -> set:
    meta = _load_meta()
    return set(meta["id_to_int"].keys())


def index_notes(notes, embeddings: List[List[float]], progress_callback=None):
    """Add notes + embeddings. Skips already-indexed notes (by rel_path)."""
    meta = _load_meta()
    index = _load_index()

    existing = set(meta["id_to_int"].keys())
    to_add_notes = []
    to_add_embs = []

    for note, emb in zip(notes, embeddings):
        if note.rel_path not in existing:
            to_add_notes.append(note)
            to_add_embs.append(emb)

    if not to_add_notes:
        return

    total = len(to_add_notes)
    emb_matrix = np.array(to_add_embs, dtype=np.float32)
    int_ids = list(range(meta["next_int"], meta["next_int"] + total))

    index.add_items(emb_matrix, int_ids)

    for note, iid in zip(to_add_notes, int_ids):
        meta["id_to_int"][note.rel_path] = iid
        meta["int_to_meta"][str(iid)] = {
            "filename": note.filename,
            "title": note.title,
            "tags": ",".join(note.tags[:20]),
            "links": ",".join(note.links[:20]),
            "word_count": note.word_count,
            "rel_path": note.rel_path,
            "snippet": note.content[:500],
        }
        if progress_callback:
            progress_callback(iid - meta["next_int"] + 1, total)

    meta["next_int"] += total
    _save_meta(meta)
    _save_index(index)


def reindex_note(note, embedding: List[float]):
    """Force re-index a single note."""
    meta = _load_meta()

    if note.rel_path in meta["id_to_int"]:
        iid = meta["id_to_int"][note.rel_path]
    else:
        iid = meta["next_int"]
        meta["next_int"] += 1
        meta["id_to_int"][note.rel_path] = iid

    index = _load_index()
    emb_arr = np.array([embedding], dtype=np.float32)
    index.add_items(emb_arr, [iid])
    meta["int_to_meta"][str(iid)] = {
        "filename": note.filename,
        "title": note.title,
        "tags": ",".join(note.tags[:20]),
        "links": ",".join(note.links[:20]),
        "word_count": note.word_count,
        "rel_path": note.rel_path,
        "snippet": note.content[:500],
    }
    _save_meta(meta)
    _save_index(index)


def semantic_search(query_embedding: List[float], n_results: int = 10,
                    exclude_id: Optional[str] = None) -> List[Dict]:
    """Returns list of result dicts sorted by similarity (highest first)."""
    meta = _load_meta()
    if not meta["int_to_meta"]:
        return []

    index = _load_index()
    count = index.get_current_count()
    if count == 0:
        return []

    k = min(n_results + 5, count)
    q = np.array([query_embedding], dtype=np.float32)
    labels, distances = index.knn_query(q, k=k)

    hits = []
    for label, dist in zip(labels[0], distances[0]):
        m = meta["int_to_meta"].get(str(label))
        if not m:
            continue
        if exclude_id and m["rel_path"] == exclude_id:
            continue
        similarity = 1.0 - float(dist)
        hits.append({
            "id": m["rel_path"],
            "title": m["title"],
            "filename": m["filename"],
            "tags": m["tags"].split(",") if m["tags"] else [],
            "rel_path": m["rel_path"],
            "word_count": m["word_count"],
            "distance": round(float(dist), 4),
            "similarity": round(similarity, 4),
            "snippet": m["snippet"][:300],
        })
        if len(hits) >= n_results:
            break

    return hits


def get_store_stats() -> Dict:
    meta = _load_meta()
    return {"indexed_notes": len(meta["id_to_int"])}


def clear_index():
    for path in [INDEX_FILE, META_FILE]:
        if os.path.exists(path):
            os.remove(path)
