"""
Obsidian vault reader: loads .md files, parses frontmatter, extracts wiki-links and tags.
"""
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import frontmatter as fm


@dataclass
class Note:
    path: str             # absolute path
    rel_path: str         # relative to vault root
    filename: str         # stem without .md
    title: str            # first H1 heading or filename
    content: str          # body text (no frontmatter)
    tags: List[str]       # from frontmatter + inline #tags
    links: List[str]      # [[wiki-links]] targets (note names)
    frontmatter: Dict     # raw frontmatter dict
    word_count: int
    char_count: int

    def truncated(self, max_chars: int = 3000) -> str:
        """Return content truncated for LLM context."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + f"\n\n[...truncated {len(self.content) - max_chars} chars]"

    def summary_line(self) -> str:
        return f"{self.title} ({self.word_count}w, {len(self.links)} links, tags: {', '.join(self.tags[:3]) or 'none'})"


def _extract_wiki_links(text: str) -> List[str]:
    """Extract [[link]] and [[link|alias]] targets."""
    raw = re.findall(r'\[\[([^\]]+)\]\]', text)
    targets = []
    for r in raw:
        # [[link|alias]] → use link part
        targets.append(r.split("|")[0].strip())
    return list(dict.fromkeys(targets))  # deduplicate, preserve order


def _extract_inline_tags(text: str) -> List[str]:
    """Extract #tag from body text (not inside code blocks)."""
    # Remove code blocks first
    clean = re.sub(r'```[\s\S]*?```', '', text)
    clean = re.sub(r'`[^`]+`', '', clean)
    tags = re.findall(r'(?<!\w)#([A-Za-z][A-Za-z0-9_/-]*)', clean)
    return [t.lower() for t in tags]


def _extract_title(meta: Dict, content: str, filename: str) -> str:
    """Priority: frontmatter title → first H1 → filename."""
    if meta.get("title"):
        return str(meta["title"]).strip().strip('"').strip("'")
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename


def _clean_tag(tag) -> str:
    """Normalise a tag value: strip quotes, lowercase, replace spaces with -."""
    return str(tag).strip().strip('"').strip("'").lower().replace(" ", "-")


def load_note(path: Path, vault_root: Path) -> Optional[Note]:
    """Parse a single .md file into a Note. Returns None on read error."""
    try:
        post = fm.load(str(path))
        meta: Dict = dict(post.metadata)
        body: str = post.content

        # Tags from frontmatter — handles:
        #   tags: ["quoted", "strings"]  (JSON-style, common in Obsidian)
        #   tags: [bare, yaml, list]
        #   tags: "comma, separated, string"
        fm_tags = meta.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [t.strip() for t in fm_tags.split(",") if t.strip()]
        fm_tags = [_clean_tag(t) for t in fm_tags if t]

        inline_tags = _extract_inline_tags(body)
        all_tags = list(dict.fromkeys(fm_tags + inline_tags))

        links = _extract_wiki_links(body)
        title = _extract_title(meta, body, path.stem)
        words = len(body.split())

        return Note(
            path=str(path),
            rel_path=str(path.relative_to(vault_root)),
            filename=path.stem,
            title=title,
            content=body,
            tags=all_tags,
            links=links,
            frontmatter=meta,
            word_count=words,
            char_count=len(body),
        )
    except Exception:
        return None


def load_vault(vault_path: str, progress_callback=None) -> List[Note]:
    """
    Load all .md files from vault_path recursively.
    progress_callback(current, total) called during load.
    """
    root = Path(vault_path)
    md_files = sorted(root.rglob("*.md"))

    # Skip hidden dirs like .obsidian, .trash
    md_files = [
        f for f in md_files
        if not any(part.startswith(".") for part in f.relative_to(root).parts)
    ]

    notes = []
    total = len(md_files)
    for i, path in enumerate(md_files):
        if progress_callback:
            progress_callback(i + 1, total)
        note = load_note(path, root)
        if note:
            notes.append(note)

    return notes


def build_link_graph(notes: List[Note]) -> Dict[str, List[str]]:
    """
    Returns adjacency dict: filename → list of filenames it links to (resolved).
    Only includes links that resolve to actual notes in the vault.
    """
    name_to_note = {n.filename.lower(): n.filename for n in notes}
    graph: Dict[str, List[str]] = {}

    for note in notes:
        resolved = []
        for link in note.links:
            key = link.lower()
            if key in name_to_note:
                resolved.append(name_to_note[key])
        graph[note.filename] = resolved

    return graph


def find_orphan_notes(notes: List[Note], graph: Dict[str, List[str]]) -> List[Note]:
    """Notes with no outgoing or incoming links."""
    has_incoming = set()
    for targets in graph.values():
        has_incoming.update(targets)

    orphans = []
    for note in notes:
        if not note.links and note.filename not in has_incoming:
            orphans.append(note)
    return orphans


def find_hub_notes(notes: List[Note], graph: Dict[str, List[str]], top_n: int = 10) -> List[tuple]:
    """Return top N notes by incoming link count (most-linked-to)."""
    incoming: Dict[str, int] = {}
    for targets in graph.values():
        for t in targets:
            incoming[t] = incoming.get(t, 0) + 1

    name_map = {n.filename: n for n in notes}
    ranked = sorted(incoming.items(), key=lambda x: x[1], reverse=True)
    return [(name_map[k], v) for k, v in ranked[:top_n] if k in name_map]
