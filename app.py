"""
LLMObsidian — Local AI assistant for Obsidian knowledge management.
Run with: streamlit run app.py
"""
import os
import re
import sys
import json
import random
from datetime import date as _dt_date
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="LLMObsidian",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

import config
from core.vault import load_vault, build_link_graph, find_orphan_notes, find_hub_notes
from core.embeddings import embed_text, embed_batch, note_to_embed_text, get_model
from core.vector_store import (
    index_notes, semantic_search, get_store_stats,
    get_indexed_ids, clear_index, reindex_note,
)
from core.analyzer import (
    vault_statistics, get_notes_needing_attention,
    tag_distribution, find_notes_by_tag,
)
import core.llm as llm
from core.bills import (
    load_bills, add_bill, update_bill, delete_bill,
    get_bills_due, spending_by_category, spending_in_period,
    bills_summary_for_llm, CATEGORIES, FREQUENCIES,
)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "notes": None,
        "stats": None,
        "vault_path": config.VAULT_PATH,
        "indexed": False,
        "page": "Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_notes(vault_path: str):
    progress = st.progress(0, text="Reading vault…")
    total_ref = [1]

    def cb(cur, tot):
        total_ref[0] = tot
        progress.progress(cur / tot, text=f"Loading note {cur}/{tot}…")

    notes = load_vault(vault_path, progress_callback=cb)
    progress.empty()
    return notes


def _ensure_notes():
    if st.session_state.notes is None:
        with st.spinner("Loading vault…"):
            st.session_state.notes = _load_notes(st.session_state.vault_path)
            st.session_state.stats = vault_statistics(st.session_state.notes)
    return st.session_state.notes


def _index_vault(notes):
    """Embed and index all notes not yet in ChromaDB."""
    already = get_indexed_ids()
    to_index = [n for n in notes if n.rel_path not in already]

    if not to_index:
        st.success("All notes already indexed.")
        return

    st.info(f"Indexing {len(to_index)} new notes (embedding model loads on first run)…")
    progress = st.progress(0)

    texts = [note_to_embed_text(n) for n in to_index]

    def cb(cur, tot):
        progress.progress(cur / tot)

    embeddings = embed_batch(texts, progress_callback=cb)
    index_notes(to_index, embeddings)
    progress.empty()
    st.session_state.indexed = True
    st.success(f"Indexed {len(to_index)} notes.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("🧠 LLMObsidian")
        st.caption(f"Model: `{config.LLM_MODEL}`")
        st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")
        st.divider()

        # Vault path
        vault = st.text_input("Vault path", value=st.session_state.vault_path)
        if vault != st.session_state.vault_path:
            st.session_state.vault_path = vault
            st.session_state.notes = None
            st.session_state.stats = None
            st.session_state.indexed = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload Vault", use_container_width=True):
                st.session_state.notes = None
                st.session_state.stats = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Index", use_container_width=True):
                clear_index()
                st.session_state.indexed = False
                st.success("Index cleared.")

        st.divider()
        _pages = [
            "Dashboard", "Semantic Search", "Note Analyser", "Link Suggester",
            "Zettelkasten Advisor", "Research Questions", "Generate Zettel",
            "Conversation Prompter", "Idea Carousel", "Research Bio", "Vault Editor",
            "Bill Tracker",
        ]
        page = st.radio(
            "Navigate",
            _pages,
            index=_pages.index(st.session_state.page) if st.session_state.page in _pages else 0,
        )
        st.session_state.page = page

        # Stats in sidebar
        store = get_store_stats()
        if st.session_state.notes:
            st.divider()
            st.metric("Notes loaded", len(st.session_state.notes))
            st.metric("Notes indexed", store["indexed_notes"])

    return page


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_dashboard(notes):
    st.title("📊 Vault Dashboard")
    stats = st.session_state.stats or {}

    # Index button
    col_idx, _ = st.columns([1, 3])
    with col_idx:
        if st.button("⚡ Index / Update Vault", type="primary", use_container_width=True):
            _index_vault(notes)

    st.divider()

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Notes", stats.get("total", 0))
    c2.metric("Orphan Notes", stats.get("orphans", 0))
    c3.metric("Without Tags", stats.get("no_tags", 0))
    c4.metric("Avg Word Count", f"{stats.get('avg_words', 0):.0f}")
    c5.metric("Avg Links/Note", f"{stats.get('avg_links', 0):.1f}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🏆 Hub Notes (most linked-to)")
        hubs = stats.get("hubs", [])
        if hubs:
            for note, count in hubs[:8]:
                st.markdown(f"- **{note.title}** — {count} incoming links")
        else:
            st.info("No hub notes found (no [[wiki-links]] detected).")

        st.subheader("🏷️ Top Tags")
        tag_dist = stats.get("top_tags", [])
        if tag_dist:
            for tag, count in tag_dist[:12]:
                st.markdown(f"- `#{tag}` — {count} notes")
        else:
            st.info("No tags found.")

    with col_r:
        st.subheader("⚠️ Notes Needing Attention")
        attention = get_notes_needing_attention(notes)
        tab1, tab2, tab3, tab4 = st.tabs(
            [f"Stubs ({len(attention['stubs'])})",
             f"No Tags ({len(attention['no_tags'])})",
             f"No Links ({len(attention['no_links'])})",
             f"No Title ({len(attention['title_is_filename'])})"]
        )
        with tab1:
            for n in attention["stubs"][:15]:
                st.markdown(f"- {n.rel_path} ({n.word_count}w)")
        with tab2:
            for n in attention["no_tags"][:15]:
                st.markdown(f"- {n.rel_path}")
        with tab3:
            for n in attention["no_links"][:15]:
                st.markdown(f"- {n.rel_path}")
        with tab4:
            for n in attention["title_is_filename"][:15]:
                st.markdown(f"- {n.rel_path}")


def page_semantic_search(notes):
    st.title("🔍 Semantic Search")
    st.caption("Find notes by meaning, not keywords. Requires vault to be indexed.")

    store = get_store_stats()
    if store["indexed_notes"] == 0:
        st.warning("Vault not indexed yet. Go to Dashboard and click **Index / Update Vault**.")
        return

    query = st.text_input("Search query", placeholder="e.g. knowledge management and memory consolidation")
    n_results = st.slider("Results", 3, 20, 8)

    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Embedding query…"):
            q_emb = embed_text(query)
        with st.spinner("Searching…"):
            hits = semantic_search(q_emb, n_results=n_results)

        if not hits:
            st.info("No results found.")
            return

        for i, hit in enumerate(hits, 1):
            score_pct = int(hit["similarity"] * 100)
            with st.expander(f"{i}. **{hit['title']}** — {score_pct}% match"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Path:** `{hit['rel_path']}`")
                    tags = ", ".join(f"`#{t}`" for t in hit["tags"][:6] if t)
                    if tags:
                        st.markdown(f"**Tags:** {tags}")
                    st.markdown("**Snippet:**")
                    st.markdown(f"> {hit['snippet'][:400]}")
                with col2:
                    st.metric("Similarity", f"{score_pct}%")
                    st.metric("Words", hit["word_count"])
                    full_path = os.path.join(st.session_state.vault_path, hit["rel_path"])
                    st.code(f"obsidian://open?path={full_path}", language=None)


def page_note_analyser(notes):
    st.title("🧠 Note Analyser")
    st.caption("Select a note for deep AI analysis: atomic idea check, Zettelkasten score, tag suggestions.")

    note_titles = [f"{n.rel_path}" for n in notes]
    selected = st.selectbox("Select note", note_titles)

    note = next((n for n in notes if n.rel_path == selected), None)
    if not note:
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Title:** {note.title}")
        st.markdown(f"**Words:** {note.word_count} | **Links:** {len(note.links)} | **Tags:** {', '.join(note.tags) or 'none'}")
    with col2:
        analyse = st.button("🧠 Analyse with AI", type="primary")

    with st.expander("📄 Note content", expanded=False):
        st.markdown(note.content[:3000])

    if analyse:
        # Gather related notes for context
        related_summaries = []
        store = get_store_stats()
        if store["indexed_notes"] > 0:
            q_emb = embed_text(note_to_embed_text(note))
            hits = semantic_search(q_emb, n_results=5, exclude_id=note.rel_path)
            related_summaries = [f"{h['title']} (sim: {h['similarity']:.2f})" for h in hits]

        st.divider()
        st.subheader("AI Analysis")
        placeholder = st.empty()
        full_text = ""
        for token in llm.analyze_note(note.title, note.content, related_summaries):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)


def page_link_suggester(notes):
    st.title("🔗 Link Suggester")
    st.caption("Find notes that should be linked to each other. Requires indexed vault.")

    store = get_store_stats()
    if store["indexed_notes"] == 0:
        st.warning("Vault not indexed yet. Go to Dashboard and click **Index / Update Vault**.")
        return

    note_titles = [n.rel_path for n in notes]
    selected = st.selectbox("Select note to enrich with links", note_titles)
    note = next((n for n in notes if n.rel_path == selected), None)
    if not note:
        return

    st.markdown(f"**Current links:** {', '.join(f'[[{l}]]' for l in note.links) or 'none'}")

    n_candidates = st.slider("Candidate notes to consider", 5, 15, 8)

    if st.button("🔗 Suggest Links", type="primary"):
        with st.spinner("Finding semantically similar notes…"):
            q_emb = embed_text(note_to_embed_text(note))
            candidates = semantic_search(q_emb, n_results=n_candidates, exclude_id=note.rel_path)

        st.subheader("Candidates found")
        for c in candidates:
            st.markdown(f"- **{c['title']}** (`{c['rel_path']}`) — {int(c['similarity']*100)}% similar")

        st.divider()
        st.subheader("AI Link Recommendations")
        placeholder = st.empty()
        full_text = ""
        for token in llm.suggest_links(note.title, note.content[:400], candidates):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        st.divider()
        st.subheader("📋 Copy to clipboard")
        existing = set(note.links)
        new_links = [c["filename"] for c in candidates if c["filename"] not in existing]
        if new_links:
            links_md = "See also: " + ", ".join(f"[[{l}]]" for l in new_links)
            st.code(links_md, language="markdown")


def page_zettelkasten_advisor(notes):
    st.title("📚 Zettelkasten Advisor")
    st.caption("Vault-wide health report and strategic recommendations.")

    stats = st.session_state.stats or vault_statistics(notes)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Quick Stats")
        st.metric("Total Notes", stats["total"])
        st.metric("Orphans", stats["orphans"])
        pct = int(stats["orphans"] / stats["total"] * 100) if stats["total"] else 0
        st.metric("Orphan %", f"{pct}%")
        st.metric("No Tags", stats["no_tags"])
        st.metric("Avg Links", f"{stats['avg_links']:.1f}")

    with col2:
        if st.button("🧠 Generate Vault Report", type="primary"):
            orphan_names = [n.filename for n in stats["orphan_notes"][:8]]
            hub_names = [f"{n.title} ({c} links)" for n, c in stats["hubs"][:5]]
            tag_names = [t for t, _ in stats["top_tags"][:15]]

            placeholder = st.empty()
            full_text = ""
            for token in llm.zettelkasten_overview(stats, orphan_names, hub_names, tag_names):
                full_text += token
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)

    st.divider()
    st.subheader("🏷️ Tag Cloud")
    tag_dist = tag_distribution(notes)
    if tag_dist:
        tag_html = " ".join(
            f'<span style="font-size:{12 + min(c, 10)*2}px; margin:3px; '
            f'background:#1e3a5f; color:#7dd3fc; padding:2px 6px; border-radius:4px;">'
            f'#{t}</span>'
            for t, c in tag_dist[:40]
        )
        st.markdown(tag_html, unsafe_allow_html=True)


def page_research_questions(notes):
    st.title("💡 Research Questions Generator")
    st.caption("Select a cluster of notes and let the AI identify gaps and research questions.")

    tag_dist = tag_distribution(notes)
    if not tag_dist:
        st.warning("No tags found in vault. Add #tags to notes for best results.")

    mode = st.radio("Select notes by", ["Tag", "Manual search"])

    selected_notes = []
    if mode == "Tag":
        if tag_dist:
            chosen_tag = st.selectbox("Tag", [t for t, _ in tag_dist[:30]])
            selected_notes = find_notes_by_tag(notes, chosen_tag)
            st.info(f"{len(selected_notes)} notes with `#{chosen_tag}`")
    else:
        query = st.text_input("Semantic search to find relevant notes")
        store = get_store_stats()
        if query and store["indexed_notes"] > 0:
            q_emb = embed_text(query)
            hits = semantic_search(q_emb, n_results=10)
            name_map = {n.rel_path: n for n in notes}
            selected_notes = [name_map[h["id"]] for h in hits if h["id"] in name_map]
        elif store["indexed_notes"] == 0:
            st.warning("Index vault first (Dashboard → Index / Update Vault).")

    if selected_notes:
        with st.expander(f"Selected {len(selected_notes)} notes"):
            for n in selected_notes[:15]:
                st.markdown(f"- {n.title} ({n.word_count}w)")

        if st.button("💡 Generate Research Questions", type="primary"):
            snippets = [{"title": n.title, "snippet": n.content[:200]} for n in selected_notes[:10]]
            placeholder = st.empty()
            full_text = ""
            for token in llm.find_research_questions(snippets):
                full_text += token
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)


def page_generate_zettel(notes):
    st.title("✍️ Generate Zettelkasten Note")
    st.caption("Convert an existing note into a proper permanent Zettelkasten note.")

    mode = st.radio("Input", ["From existing note", "Paste raw text"])

    if mode == "From existing note":
        note_titles = [n.rel_path for n in notes]
        selected = st.selectbox("Select note", note_titles)
        note = next((n for n in notes if n.rel_path == selected), None)
        if note:
            source_title = note.title
            content = note.content
            with st.expander("Preview original"):
                st.markdown(content[:2000])
    else:
        source_title = st.text_input("Source title / reference")
        content = st.text_area("Paste raw notes / highlights", height=200)

    if st.button("✍️ Generate Zettel", type="primary") and content:
        st.subheader("Generated Permanent Note")
        placeholder = st.empty()
        full_text = ""
        for token in llm.generate_zettel(content, source_title):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        st.divider()
        st.subheader("📋 Raw Markdown")
        st.code(full_text, language="markdown")

        filename = source_title.lower().replace(" ", "-").replace("/", "-")[:60] + ".md"
        st.download_button(
            label="⬇️ Download Note",
            data=full_text,
            file_name=filename,
            mime="text/markdown",
        )


# ── Research Bio helpers ──────────────────────────────────────────────────────

def _bio_get_sub_vaults(notes) -> list:
    """Extract unique top-level folder names from note paths."""
    vaults = set()
    for n in notes:
        parts = n.rel_path.split("/")
        if len(parts) > 1:
            vaults.add(parts[0])
    return sorted(vaults)


def _bio_parse_themes(text: str) -> list:
    """Parse THEME: Name | Description lines into list of dicts."""
    themes = []
    for line in text.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("THEME:"):
            body = s.split(":", 1)[1].strip()
            if "|" in body:
                name, desc = body.split("|", 1)
                name = name.strip()
                desc = desc.strip()
                if name and desc:
                    themes.append({"name": name, "description": desc})
    return themes


def page_research_bio(notes):
    st.title("📝 Research Biography")
    st.caption(
        "Analyses your vault to identify your main research themes, "
        "then writes a professional biography you can use to introduce yourself."
    )

    stats = st.session_state.stats or vault_statistics(notes)

    hub_notes    = stats.get("hubs", [])
    hub_titles   = [n.title for n, _ in hub_notes[:8]]
    hub_snippets = [n.content[:200].replace("\n", " ").strip() for n, _ in hub_notes[:8]]
    top_tags     = stats.get("top_tags", [])
    top_tag_names = [t for t, _ in top_tags[:15]]
    sub_vaults   = _bio_get_sub_vaults(notes)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        generate = st.button("Generate Biography", type="primary", use_container_width=True)
    with col_info:
        st.caption(
            f"{stats.get('total', 0)} notes · {len(hub_titles)} hub topics · "
            f"{len(top_tags)} tags · sub-vaults: {', '.join(sub_vaults) or 'root'}"
        )

    if generate:
        # ── Step 1: identify themes ───────────────────────────────────────────
        with st.spinner("Step 1/2 — Analysing research themes…"):
            raw_themes = llm.identify_research_themes(
                top_tags=top_tags,
                hub_titles=hub_titles,
                hub_snippets=hub_snippets,
                sub_vaults=sub_vaults,
            )
        themes = _bio_parse_themes(raw_themes)

        if not themes:
            st.warning("Theme extraction didn't return usable results. Try indexing the vault first so hub notes are populated.")
            st.code(raw_themes)
            return

        # Show the identified themes as a reference
        st.divider()
        st.subheader("Research themes identified")
        for t in themes:
            st.markdown(f"**{t['name']}** — {t['description']}")

        # ── Step 2: write the bio ─────────────────────────────────────────────
        st.divider()
        st.subheader("Your biography")
        placeholder = st.empty()
        full_text = ""
        for token in llm.generate_professional_bio(
            themes=themes,
            hub_titles=hub_titles,
            top_tags=top_tag_names,
        ):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        word_count = len(full_text.split())
        st.caption(f"{word_count} words")

        st.divider()
        st.subheader("Copy")
        st.code(full_text, language=None)


# ── Idea Carousel helpers ─────────────────────────────────────────────────────

def _ic_parse_items(text: str) -> list:
    """Parse ITEM: Term | Sentence lines into list of dicts."""
    items = []
    for line in text.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("ITEM:"):
            body = s.split(":", 1)[1].strip()
            if "|" in body:
                term, sentence = body.split("|", 1)
                term = term.strip()
                sentence = sentence.strip()
                if term and sentence:
                    items.append({"term": term, "sentence": sentence})
    return items


def _ic_build_html(items: list, interval_ms: int = 3000) -> str:
    """Return a self-contained HTML carousel page."""
    items_json = json.dumps(items)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    color: white;
    overflow: hidden;
  }}
  #card {{
    text-align: center;
    padding: 32px 48px;
    max-width: 820px;
    width: 100%;
    transition: opacity 0.45s ease;
  }}
  #term {{
    font-size: 52px;
    font-weight: 800;
    color: #7dd3fc;
    line-height: 1.15;
    margin-bottom: 22px;
    letter-spacing: -0.5px;
  }}
  #sentence {{
    font-size: 22px;
    color: #cbd5e1;
    line-height: 1.55;
    max-width: 660px;
    margin: 0 auto;
  }}
  #dots {{
    display: flex;
    gap: 7px;
    justify-content: center;
    margin-top: 32px;
    flex-wrap: wrap;
    max-width: 400px;
  }}
  .dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #1e3a5f;
    transition: background 0.3s;
  }}
  .dot.active {{ background: #7dd3fc; }}
  #counter {{
    font-size: 12px;
    color: #475569;
    letter-spacing: 2px;
    margin-top: 12px;
    text-transform: uppercase;
  }}
  #controls {{
    display: flex;
    gap: 14px;
    margin-top: 28px;
    align-items: center;
  }}
  button {{
    padding: 11px 26px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-size: 15px;
    font-weight: 700;
    transition: background 0.2s, color 0.2s;
    letter-spacing: 0.3px;
  }}
  #btn-prev, #btn-next {{
    background: #1e293b;
    color: #94a3b8;
  }}
  #btn-prev:hover, #btn-next:hover {{ background: #334155; color: white; }}
  #btn-toggle {{
    background: #1d4ed8;
    color: white;
    min-width: 130px;
  }}
  #btn-toggle:hover {{ background: #1e40af; }}
  #btn-toggle.paused {{ background: #166534; }}
  #btn-toggle.paused:hover {{ background: #14532d; }}
  #pause-banner {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    background: #7f1d1d;
    color: #fca5a5;
    text-align: center;
    padding: 10px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
  }}
</style>
</head>
<body>
<div id="pause-banner">⏸ PAUSED</div>
<div id="card">
  <div id="term"></div>
  <div id="sentence"></div>
  <div id="dots"></div>
  <div id="counter"></div>
</div>
<div id="controls">
  <button id="btn-prev" onclick="prev()">&#9664; Prev</button>
  <button id="btn-toggle" onclick="togglePause()">&#9646;&#9646; Stop</button>
  <button id="btn-next" onclick="next()">Next &#9654;</button>
</div>

<script>
const items = {items_json};
const INTERVAL = {interval_ms};
let current = 0;
let running = true;
let timer = null;
let animating = false;

function renderDots() {{
  const c = document.getElementById('dots');
  c.innerHTML = '';
  const max = Math.min(items.length, 30);
  for (let i = 0; i < max; i++) {{
    const d = document.createElement('div');
    d.className = 'dot' + (i === current ? ' active' : '');
    c.appendChild(d);
  }}
}}

function showCard(idx) {{
  if (animating) return;
  animating = true;
  const card = document.getElementById('card');
  card.style.opacity = '0';
  setTimeout(() => {{
    current = ((idx % items.length) + items.length) % items.length;
    document.getElementById('term').textContent = items[current].term;
    document.getElementById('sentence').textContent = items[current].sentence;
    document.getElementById('counter').textContent = (current + 1) + ' / ' + items.length;
    renderDots();
    card.style.opacity = '1';
    animating = false;
  }}, 380);
}}

function startTimer() {{
  clearInterval(timer);
  timer = setInterval(() => {{ if (running) showCard(current + 1); }}, INTERVAL);
}}

function next() {{ showCard(current + 1); if (running) startTimer(); }}
function prev() {{ showCard(current - 1); if (running) startTimer(); }}

function togglePause() {{
  running = !running;
  const btn = document.getElementById('btn-toggle');
  const banner = document.getElementById('pause-banner');
  if (running) {{
    btn.textContent = '⏸ Stop';
    btn.classList.remove('paused');
    banner.style.display = 'none';
    startTimer();
  }} else {{
    btn.innerHTML = '&#9654; Resume';
    btn.classList.add('paused');
    banner.style.display = 'block';
  }}
}}

// Keyboard: space = pause/resume, arrow keys = prev/next
document.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{ e.preventDefault(); togglePause(); }}
  if (e.code === 'ArrowRight') next();
  if (e.code === 'ArrowLeft') prev();
}});

showCard(0);
startTimer();
</script>
</body>
</html>"""


def page_idea_carousel(notes):
    st.title("🎠 Idea Carousel")
    st.caption(
        "One concept at a time, auto-advancing every 3 seconds. "
        "Space bar or the Stop button to pause. Arrow keys or Prev/Next to navigate."
    )

    # State
    for k, v in {"ic_items": None, "ic_keyword": "", "ic_interval": 3}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_kw, col_speed, col_btn = st.columns([3, 1, 1])
    with col_kw:
        keyword = st.text_input(
            "Topic or keyword",
            value=st.session_state.ic_keyword,
            placeholder="e.g. consciousness, emergence, power",
            label_visibility="collapsed",
        )
    with col_speed:
        interval = st.selectbox("Speed", [2, 3, 4, 5, 8], index=1,
                                format_func=lambda x: f"{x}s")
        st.session_state.ic_interval = interval
    with col_btn:
        go = st.button("▶ Start Carousel", type="primary", use_container_width=True)

    if go and keyword.strip():
        st.session_state.ic_keyword = keyword.strip()
        store = get_store_stats()
        with st.spinner("Searching vault…"):
            if store["indexed_notes"] > 0:
                q_emb = embed_text(keyword.strip())
                hits = semantic_search(q_emb, n_results=15)
            else:
                kw_lower = keyword.strip().lower()
                hits = [
                    {"title": n.title, "snippet": n.content[:400]}
                    for n in notes
                    if kw_lower in n.title.lower() or kw_lower in n.content.lower()
                ][:15]

        if not hits:
            st.warning("No matching notes found. Try a different keyword or index the vault first.")
            return

        snippets = [{"title": h["title"], "snippet": h.get("snippet", h.get("snippet", ""))} for h in hits]
        with st.spinner("Generating carousel items with AI…"):
            raw = llm.generate_carousel_items(keyword.strip(), snippets)

        items = _ic_parse_items(raw)
        if not items:
            st.warning("The model didn't return valid items. Try again or rephrase the keyword.")
            st.code(raw)
            return

        st.session_state.ic_items = items

    if st.session_state.ic_items:
        items = st.session_state.ic_items
        interval_ms = st.session_state.ic_interval * 1000
        st.caption(f"{len(items)} items · {st.session_state.ic_interval}s interval · **Space** = pause · **← →** = navigate")
        html = _ic_build_html(items, interval_ms)
        components.html(html, height=520, scrolling=False)

        with st.expander("📋 All items", expanded=False):
            for i, item in enumerate(items, 1):
                st.markdown(f"**{i}. {item['term']}** — {item['sentence']}")
    else:
        st.markdown(
            """
            <div style="margin-top:48px;text-align:center;color:#4b5563;font-size:16px;">
            Type a topic and press <b>▶ Start Carousel</b>.<br><br>
            Cards auto-advance every few seconds — use it as a live prompt feed<br>
            during talks or conversations.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Conversation Prompter helpers ─────────────────────────────────────────────

def _cp_init_state():
    defaults = {
        "cp_paused": False,
        "cp_chips": None,       # {"concepts": [...], "authors": [...], "ideas": [...]}
        "cp_hits": [],          # all search hits for the current keyword
        "cp_batch": 0,          # which batch of 5 hits we're on
        "cp_keyword": "",
        "cp_search_mode": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _cp_search(keyword: str, notes) -> tuple:
    """Return (hits_list, mode_string). Semantic if indexed, text-match fallback."""
    store = get_store_stats()
    if store["indexed_notes"] > 0:
        q_emb = embed_text(keyword)
        hits = semantic_search(q_emb, n_results=20)
        return hits, "semantic"
    # Text fallback
    kw_lower = keyword.lower()
    matches = [n for n in notes if kw_lower in n.title.lower() or kw_lower in n.content.lower()]
    matches.sort(key=lambda n: n.content.lower().count(kw_lower), reverse=True)
    hits = [
        {"title": n.title, "snippet": n.content[:400], "rel_path": n.rel_path}
        for n in matches[:20]
    ]
    return hits, "text"


def _cp_parse_chips(text: str) -> dict:
    """Parse pipe-delimited LLM response into concept/author/idea lists."""
    chips = {"concepts": [], "authors": [], "ideas": []}
    bad = {"none", "n/a", ""}
    for line in text.strip().splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("CONCEPTS"):
            items = s.split(":", 1)[1]
            chips["concepts"] = [x.strip() for x in items.split("|") if x.strip().lower() not in bad]
        elif u.startswith("AUTHORS") or u.startswith("PEOPLE"):
            items = s.split(":", 1)[1]
            chips["authors"] = [x.strip() for x in items.split("|") if x.strip().lower() not in bad]
        elif u.startswith("IDEAS") or u.startswith("IDEA"):
            items = s.split(":", 1)[1]
            chips["ideas"] = [x.strip() for x in items.split("|") if x.strip().lower() not in bad]
    return chips


def _cp_get_batch(hits: list, batch_idx: int, size: int = 5) -> list:
    """Slice hits into a rotating batch."""
    if not hits:
        return []
    start = (batch_idx * size) % len(hits)
    slc = hits[start:start + size]
    if len(slc) < size:          # wrap around
        slc += hits[:size - len(slc)]
    return slc


def _cp_render_chips(chips: dict, paused: bool):
    """Render concept/author/idea chips as large coloured HTML badges."""
    opacity = "0.35" if paused else "1.0"

    def badge_row(items, bg, fg):
        if not items:
            return '<span style="color:#6b7280;font-size:16px;">—</span>'
        return "".join(
            f'<span style="display:inline-block;margin:5px 6px;padding:10px 22px;'
            f'background:{bg};color:{fg};border-radius:24px;font-size:24px;'
            f'font-weight:600;letter-spacing:0.3px;opacity:{opacity};">{item}</span>'
            for item in items
        )

    label_style = (
        "font-size:11px;text-transform:uppercase;letter-spacing:2px;"
        "margin-bottom:6px;padding-left:4px;"
    )

    if paused:
        pause_bar = (
            '<div style="text-align:center;padding:10px 0;margin-bottom:14px;'
            'background:#7f1d1d;border-radius:8px;font-size:17px;font-weight:700;'
            'color:#fca5a5;">⏸ PAUSED — press Resume when ready</div>'
        )
    else:
        pause_bar = ""

    html = f"""
    {pause_bar}
    <div style="padding:8px 4px;">
      <div style="color:#93c5fd;{label_style}">Concepts &amp; Frameworks</div>
      <div style="margin-bottom:18px;">{badge_row(chips['concepts'], '#1e3a5f', '#bfdbfe')}</div>
      <div style="color:#86efac;{label_style}">People &amp; Authors</div>
      <div style="margin-bottom:18px;">{badge_row(chips['authors'], '#14532d', '#bbf7d0')}</div>
      <div style="color:#fed7aa;{label_style}">Ideas &amp; Claims</div>
      <div style="margin-bottom:8px;">{badge_row(chips['ideas'], '#7c2d12', '#fed7aa')}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def page_conversation_prompter(notes):
    st.title("💬 Conversation Prompter")
    st.caption(
        "Type a keyword → get concept chips from your vault. "
        "Use as memory-joggers during talks or conversations."
    )

    _cp_init_state()

    # ── Search row ────────────────────────────────────────────────────────────
    col_kw, col_btn = st.columns([4, 1])
    with col_kw:
        keyword = st.text_input(
            "Topic or keyword",
            value=st.session_state.cp_keyword,
            placeholder="e.g. consciousness, power dynamics, emergence",
            disabled=st.session_state.cp_paused,
            label_visibility="collapsed",
        )
    with col_btn:
        search_hit = st.button(
            "🔍 Find Prompts",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.cp_paused,
        )

    # ── Control row ───────────────────────────────────────────────────────────
    col_stop, col_next, col_clear = st.columns(3)

    with col_stop:
        if st.session_state.cp_paused:
            if st.button("▶ Resume", use_container_width=True, type="primary"):
                st.session_state.cp_paused = False
                st.rerun()
        else:
            stop_disabled = st.session_state.cp_chips is None
            if st.button("⏸ Stop", use_container_width=True, disabled=stop_disabled):
                st.session_state.cp_paused = True
                st.rerun()

    with col_next:
        next_disabled = (not st.session_state.cp_hits) or st.session_state.cp_paused
        if st.button("⏭ Next Ideas", use_container_width=True, disabled=next_disabled):
            st.session_state.cp_batch += 1
            batch = _cp_get_batch(st.session_state.cp_hits, st.session_state.cp_batch)
            with st.spinner("Generating new prompts…"):
                raw = llm.extract_conversation_prompts(st.session_state.cp_keyword, batch)
            st.session_state.cp_chips = _cp_parse_chips(raw)
            st.rerun()

    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            for k in ("cp_paused", "cp_chips", "cp_hits", "cp_batch", "cp_keyword", "cp_search_mode"):
                st.session_state[k] = (False if k == "cp_paused" else (None if k == "cp_chips" else ([] if k == "cp_hits" else (0 if k == "cp_batch" else ""))))
            st.rerun()

    # ── Search action ─────────────────────────────────────────────────────────
    if search_hit and keyword.strip():
        st.session_state.cp_keyword = keyword.strip()
        st.session_state.cp_batch = 0
        st.session_state.cp_paused = False

        with st.spinner("Searching vault…"):
            hits, mode = _cp_search(keyword.strip(), notes)
        st.session_state.cp_hits = hits
        st.session_state.cp_search_mode = mode

        if not hits:
            st.warning("No matching notes found. Try a different keyword or index the vault first.")
        else:
            batch = _cp_get_batch(hits, 0)
            with st.spinner("Extracting concepts with AI…"):
                raw = llm.extract_conversation_prompts(keyword.strip(), batch)
            st.session_state.cp_chips = _cp_parse_chips(raw)
            st.rerun()

    # ── Display chips ─────────────────────────────────────────────────────────
    if st.session_state.cp_chips:
        st.divider()

        # Show search context
        mode_label = "🔭 semantic" if st.session_state.cp_search_mode == "semantic" else "🔤 text"
        total = len(st.session_state.cp_hits)
        batch_num = st.session_state.cp_batch + 1
        batch_size = 5
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        st.caption(
            f"**{st.session_state.cp_keyword}** · {total} notes matched ({mode_label}) "
            f"· batch {batch_num}/{total_batches} · press **Next Ideas** to rotate"
        )

        _cp_render_chips(st.session_state.cp_chips, st.session_state.cp_paused)

        # Source notes expander
        with st.expander("📄 Source notes for this batch", expanded=False):
            batch = _cp_get_batch(st.session_state.cp_hits, st.session_state.cp_batch)
            for h in batch:
                st.markdown(f"- **{h['title']}** — `{h.get('rel_path', '')}`")

    elif not search_hit:
        st.markdown(
            """
            <div style="margin-top:40px;text-align:center;color:#4b5563;font-size:16px;">
            Type a keyword and press <b>Find Prompts</b> to begin.<br><br>
            During a presentation: press <b>⏸ Stop</b> while you talk,<br>
            <b>▶ Resume</b> or <b>⏭ Next Ideas</b> when you need a new angle.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Vault Editor ──────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r'^---\r?\n.*?\r?\n---\r?\n?', re.DOTALL)

_POSITION_OPTIONS = [
    "After frontmatter (or top if none)",
    "Prepend — very top of file",
    "Append — bottom of file",
    "Replace frontmatter",
]

_DEFAULT_SNIPPET = """---
type: field-note
status: to-atomize
date: 2026-02-26
---"""


def _ve_apply(original: str, insert_text: str, position: str) -> str:
    """Return modified file content with insert_text applied."""
    text = insert_text.rstrip("\n") + "\n"

    if position == "Prepend — very top of file":
        return text + original

    elif position == "Append — bottom of file":
        return original.rstrip("\n") + "\n" + text

    elif position == "Replace frontmatter":
        body = _FRONTMATTER_RE.sub("", original).lstrip("\n")
        return text + body

    else:  # "After frontmatter (or top if none)"
        m = _FRONTMATTER_RE.match(original)
        if m:
            after = original[m.end():]
            return original[: m.end()] + text + after.lstrip("\n")
        return text + original


def page_vault_editor(notes):
    st.title("✏️ Vault Editor")
    st.caption(
        "Batch-edit notes in a directory: prepend, append, or replace text/frontmatter. "
        "**Changes are written to disk** — use the preview and confirm before applying."
    )

    vault_root = Path(st.session_state.vault_path)

    # ── Directory selector ────────────────────────────────────────────────────
    dirs = {"(all notes — entire vault)"}
    for n in notes:
        p = Path(n.rel_path).parent
        while str(p) != ".":
            dirs.add(str(p))
            p = p.parent
    dir_list = ["(all notes — entire vault)"] + sorted(dirs - {"(all notes — entire vault)"})

    col_dir, col_pos = st.columns([2, 1])
    with col_dir:
        selected_dir = st.selectbox("Target directory", dir_list)
    with col_pos:
        position = st.selectbox("Insert position", _POSITION_OPTIONS)

    # ── Filter notes ──────────────────────────────────────────────────────────
    if selected_dir == "(all notes — entire vault)":
        target_notes = list(notes)
    else:
        prefix = selected_dir + "/"
        target_notes = [n for n in notes if n.rel_path.startswith(prefix)]

    with st.expander("⚙️ Additional filters", expanded=False):
        skip_has_fm = st.checkbox("Skip notes that already have frontmatter")
        skip_has_tags = st.checkbox("Skip notes that already have tags")
        only_no_tags = st.checkbox("Only notes WITHOUT any tags")
        filename_contains = st.text_input("Only files whose path contains (leave blank for all)")

    if skip_has_fm:
        target_notes = [n for n in target_notes if not n.frontmatter]
    if skip_has_tags:
        target_notes = [n for n in target_notes if not n.tags]
    if only_no_tags:
        target_notes = [n for n in target_notes if not n.tags]
    if filename_contains.strip():
        target_notes = [n for n in target_notes if filename_contains.strip().lower() in n.rel_path.lower()]

    st.info(f"**{len(target_notes)} notes** match the current filters.")

    # ── Text to insert ────────────────────────────────────────────────────────
    st.subheader("Text to insert")
    insert_text = st.text_area(
        "Content",
        value=_DEFAULT_SNIPPET,
        height=160,
        help="Paste any text here — frontmatter YAML, a section header, a footer, anything.",
    )

    # ── Preview ───────────────────────────────────────────────────────────────
    with st.expander(f"📋 Affected notes ({len(target_notes)})", expanded=False):
        for n in target_notes[:50]:
            st.markdown(f"- `{n.rel_path}`")
        if len(target_notes) > 50:
            st.caption(f"… and {len(target_notes) - 50} more")

    if target_notes and insert_text.strip():
        if st.button("👁 Preview first note"):
            n = target_notes[0]
            try:
                original = Path(n.path).read_text(encoding="utf-8")
                modified = _ve_apply(original, insert_text, position)
                col_b, col_a = st.columns(2)
                with col_b:
                    st.caption(f"**Before** — `{n.rel_path}`")
                    st.code(original[:800], language="markdown")
                with col_a:
                    st.caption("**After**")
                    st.code(modified[:800], language="markdown")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    # ── Confirm + Apply ───────────────────────────────────────────────────────
    st.divider()
    confirm = st.checkbox(
        f"I understand this will overwrite {len(target_notes)} file(s) on disk and cannot be undone."
    )

    if st.button("✏️ Apply to All Notes", type="primary", disabled=not confirm):
        if not insert_text.strip():
            st.error("Nothing to insert — text area is empty.")
        elif not target_notes:
            st.warning("No notes match the current filters.")
        else:
            progress = st.progress(0, text="Applying changes…")
            errors = []
            for i, n in enumerate(target_notes):
                try:
                    original = Path(n.path).read_text(encoding="utf-8")
                    modified = _ve_apply(original, insert_text, position)
                    Path(n.path).write_text(modified, encoding="utf-8")
                except Exception as exc:
                    errors.append(f"{n.rel_path}: {exc}")
                progress.progress((i + 1) / len(target_notes), text=f"{i + 1}/{len(target_notes)}")
            progress.empty()

            if errors:
                st.error(f"{len(errors)} file(s) failed:")
                for e in errors:
                    st.code(e)
                st.success(f"✅ Modified {len(target_notes) - len(errors)} notes (with {len(errors)} errors).")
            else:
                st.success(f"✅ Successfully modified {len(target_notes)} notes.")

            # Reload vault so in-memory notes reflect disk state
            st.session_state.notes = None
            st.session_state.stats = None


# ── Bill Tracker ──────────────────────────────────────────────────────────────

def _bt_bill_form(bill, form_key):
    is_edit = bill is not None
    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name *", value=bill["name"] if is_edit else "")
            cat_idx = CATEGORIES.index(bill["category"]) if is_edit and bill.get("category") in CATEGORIES else 0
            category = st.selectbox("Category *", CATEGORIES, index=cat_idx)
            amount = st.number_input("Amount *", min_value=0.01, step=0.01,
                                     value=float(bill["amount"]) if is_edit else 0.01)
            currency = st.text_input("Currency", max_chars=3,
                                     value=bill.get("currency", "EUR") if is_edit else "EUR")
        with col2:
            freq_idx = FREQUENCIES.index(bill["frequency"]) if is_edit and bill.get("frequency") in FREQUENCIES else 2
            frequency = st.selectbox("Frequency", FREQUENCIES, index=freq_idx)
            try:
                dp_val = _dt_date.fromisoformat(bill["date_paid"]) if is_edit and bill.get("date_paid") else _dt_date.today()
            except ValueError:
                dp_val = _dt_date.today()
            date_paid = st.date_input("Date paid *", value=dp_val)
            try:
                nd_val = _dt_date.fromisoformat(bill["next_due"]) if is_edit and bill.get("next_due") else _dt_date.today()
            except ValueError:
                nd_val = _dt_date.today()
            next_due = st.date_input("Next due date", value=nd_val)
        notes = st.text_area("Notes", value=bill.get("notes", "") if is_edit else "", height=70)
        submitted = st.form_submit_button("💾 Save Changes" if is_edit else "➕ Add Bill", type="primary")

    if submitted:
        if not name.strip():
            st.error("Name is required.")
        elif amount <= 0:
            st.error("Amount must be greater than zero.")
        else:
            if is_edit:
                update_bill(bill["id"], name=name.strip(), category=category,
                            amount=amount, currency=currency.upper(),
                            date_paid=date_paid.isoformat(), next_due=next_due.isoformat(),
                            frequency=frequency, notes=notes.strip())
                st.success(f"Updated '{name}'.")
            else:
                add_bill(name=name.strip(), category=category, amount=amount,
                         currency=currency.upper(), date_paid=date_paid.isoformat(),
                         next_due=next_due.isoformat(), frequency=frequency, notes=notes.strip())
                st.success(f"Added '{name}'.")
            st.rerun()


def _bt_add_edit_tab(bills):
    mode = st.radio("", ["➕ Add new bill", "✏️ Edit existing bill"], horizontal=True)
    st.divider()
    if mode == "➕ Add new bill":
        _bt_bill_form(bill=None, form_key="bt_add_form")
    else:
        if not bills:
            st.info("No bills yet — add one first.")
            return
        choices = {f"{b['name']} ({b['category']}) — {b.get('currency','')} {float(b.get('amount',0)):.2f}": b
                   for b in bills}
        chosen = choices[st.selectbox("Select bill to edit", list(choices.keys()))]
        _bt_bill_form(bill=chosen, form_key="bt_edit_form")


def _bt_all_bills_tab(bills):
    if not bills:
        st.info("No bills recorded yet. Use **Add / Edit** to add your first bill.")
        return
    st.caption(f"{len(bills)} bills · sorted by next due date")
    for b in sorted(bills, key=lambda x: x.get("next_due", "")):
        label = (f"**{b['name']}** — {b.get('currency','')} {float(b.get('amount',0)):.2f}"
                 f" · {b.get('category','')} · next due {b.get('next_due','?')}")
        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Paid:** {b.get('date_paid','?')}")
            c2.markdown(f"**Frequency:** {b.get('frequency','?')}")
            c3.markdown(f"**ID:** `{b['id']}`")
            if b.get("notes"):
                st.caption(b["notes"])
            if st.button("🗑️ Delete this bill", key=f"del_{b['id']}"):
                delete_bill(b["id"])
                st.success(f"Deleted '{b['name']}'.")
                st.rerun()


def _bt_due_tab():
    days = st.slider("Show bills due within (days)", 3, 90, 14)
    due = get_bills_due(days_ahead=days)
    if not due:
        st.success(f"No bills due in the next {days} days. ✓")
        return
    for b in due:
        dl = b["days_left"]
        label = f"**{b['name']}** — {b.get('currency','')} {float(b.get('amount',0)):.2f}"
        if dl < 0:
            st.error(f"🔴 OVERDUE by {abs(dl)} day(s) — {label} (was due {b.get('next_due','')})")
        elif dl == 0:
            st.error(f"🔴 DUE TODAY — {label}")
        elif dl <= 3:
            st.warning(f"🟠 Due in {dl} day(s) — {label} on {b.get('next_due','')}")
        elif dl <= 7:
            st.warning(f"🟡 Due in {dl} day(s) — {label} on {b.get('next_due','')}")
        else:
            st.info(f"📅 Due in {dl} day(s) — {label} on {b.get('next_due','')}")


def _bt_reports_tab():
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=_dt_date.today().replace(day=1))
    with col2:
        end = st.date_input("To", value=_dt_date.today())
    if start > end:
        st.error("Start date must be before end date.")
        return

    by_cat = spending_by_category(start.isoformat(), end.isoformat())
    period_bills = spending_in_period(start.isoformat(), end.isoformat())
    total = sum(by_cat.values())

    st.metric("Total spent in period", f"{total:.2f}")
    st.divider()

    if not by_cat:
        st.info("No bills paid in this period.")
        return

    st.subheader("By category")
    for cat, amt in by_cat.items():
        pct = (amt / total * 100) if total else 0
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(pct / 100, text=f"**{cat}** — {amt:.2f}")
        with col_b:
            st.caption(f"{pct:.0f}%")

    st.divider()
    st.subheader("Individual payments")
    for b in period_bills:
        st.markdown(
            f"- **{b['name']}** ({b.get('category','')}) — "
            f"{b.get('currency','')} {float(b.get('amount',0)):.2f} · paid {b.get('date_paid','?')}"
        )


def _bt_ai_tab(bills):
    st.caption(
        "Ask anything about your bills in plain English. "
        "Python handles the maths — the AI just formats the answer."
    )
    query = st.text_input("Your question",
                          placeholder="e.g. How much did I spend on insurance this year?")
    if st.button("🤖 Ask", type="primary") and query.strip():
        context = bills_summary_for_llm(bills)
        placeholder = st.empty()
        full_text = ""
        for token in llm.answer_bill_query(query.strip(), context):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)


def page_bill_tracker(_notes):
    st.title("💳 Bill Tracker")
    st.caption("Track recurring bills, get alerts for upcoming payments, and query spending history.")

    bills = load_bills()

    due_soon = get_bills_due(days_ahead=7)
    if due_soon:
        overdue = [b for b in due_soon if b["days_left"] <= 0]
        upcoming = [b for b in due_soon if b["days_left"] > 0]
        if overdue:
            st.error(f"⚠️ Overdue: {', '.join(b['name'] for b in overdue)}")
        if upcoming:
            names = ", ".join(f"{b['name']} ({b['days_left']}d)" for b in upcoming)
            st.warning(f"📅 Due soon: {names}")

    tab_add, tab_all, tab_due, tab_report, tab_ai = st.tabs(
        ["➕ Add / Edit", "📋 All Bills", "⚠️ Due Soon", "📊 Reports", "🤖 Ask AI"]
    )
    with tab_add:
        _bt_add_edit_tab(bills)
    with tab_all:
        _bt_all_bills_tab(bills)
    with tab_due:
        _bt_due_tab()
    with tab_report:
        _bt_reports_tab()
    with tab_ai:
        _bt_ai_tab(bills)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    page = sidebar()
    notes = _ensure_notes()

    if not notes:
        st.error(f"No notes found in `{st.session_state.vault_path}`. Check the vault path in the sidebar.")
        return

    if page == "Dashboard":
        page_dashboard(notes)
    elif page == "Semantic Search":
        page_semantic_search(notes)
    elif page == "Note Analyser":
        page_note_analyser(notes)
    elif page == "Link Suggester":
        page_link_suggester(notes)
    elif page == "Zettelkasten Advisor":
        page_zettelkasten_advisor(notes)
    elif page == "Research Questions":
        page_research_questions(notes)
    elif page == "Generate Zettel":
        page_generate_zettel(notes)
    elif page == "Conversation Prompter":
        page_conversation_prompter(notes)
    elif page == "Idea Carousel":
        page_idea_carousel(notes)
    elif page == "Research Bio":
        page_research_bio(notes)
    elif page == "Vault Editor":
        page_vault_editor(notes)
    elif page == "Bill Tracker":
        page_bill_tracker(notes)


if __name__ == "__main__":
    main()
