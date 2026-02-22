"""
High-level analysis functions combining vault, embeddings, and vector store.
"""
from collections import Counter
from typing import List, Dict, Tuple

from core.vault import Note, find_orphan_notes, find_hub_notes, build_link_graph


def vault_statistics(notes: List[Note]) -> Dict:
    """Compute summary stats for the entire vault."""
    if not notes:
        return {}

    graph = build_link_graph(notes)
    orphans = find_orphan_notes(notes, graph)
    hubs = find_hub_notes(notes, graph, top_n=10)

    all_tags = []
    for n in notes:
        all_tags.extend(n.tags)
    tag_counts = Counter(all_tags)

    no_tags = sum(1 for n in notes if not n.tags)
    avg_words = sum(n.word_count for n in notes) / len(notes)
    avg_links = sum(len(n.links) for n in notes) / len(notes)
    total_links = sum(len(n.links) for n in notes)

    return {
        "total": len(notes),
        "orphans": len(orphans),
        "orphan_notes": orphans,
        "hubs": hubs,
        "no_tags": no_tags,
        "avg_words": avg_words,
        "avg_links": avg_links,
        "total_links": total_links,
        "top_tags": tag_counts.most_common(20),
        "graph": graph,
    }


def detect_potential_duplicates(notes: List[Note], search_fn, embed_fn,
                                threshold: float = 0.92) -> List[Tuple[Note, List[Dict]]]:
    """
    Find notes that are likely duplicates (cosine similarity > threshold).
    Returns list of (note, [similar_hits]) pairs.
    """
    from core.embeddings import note_to_embed_text
    duplicates = []

    for note in notes:
        emb = embed_fn(note_to_embed_text(note))
        hits = search_fn(emb, n_results=5, exclude_id=note.rel_path)
        high_sim = [h for h in hits if h["similarity"] >= threshold]
        if high_sim:
            duplicates.append((note, high_sim))

    return duplicates


def get_notes_needing_attention(notes: List[Note]) -> Dict[str, List[Note]]:
    """
    Categorise notes that need work:
    - stub: < 50 words
    - no_tags: missing tags
    - no_links: no outgoing links
    - title_is_filename: title same as filename (no H1)
    """
    stubs = [n for n in notes if n.word_count < 50]
    no_tags = [n for n in notes if not n.tags]
    no_links = [n for n in notes if not n.links]
    title_is_fn = [n for n in notes if n.title == n.filename]

    return {
        "stubs": stubs,
        "no_tags": no_tags,
        "no_links": no_links,
        "title_is_filename": title_is_fn,
    }


def tag_distribution(notes: List[Note]) -> List[Tuple[str, int]]:
    all_tags = []
    for n in notes:
        all_tags.extend(n.tags)
    return Counter(all_tags).most_common()


def find_notes_by_tag(notes: List[Note], tag: str) -> List[Note]:
    tag = tag.lower().lstrip("#")
    return [n for n in notes if tag in [t.lower() for t in n.tags]]


def cluster_by_tags(notes: List[Note]) -> Dict[str, List[Note]]:
    """Group notes by their primary (first) tag."""
    clusters: Dict[str, List[Note]] = {}
    for note in notes:
        key = note.tags[0] if note.tags else "untagged"
        clusters.setdefault(key, []).append(note)
    return clusters
