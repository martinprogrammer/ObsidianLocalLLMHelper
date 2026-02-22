"""
LLMObsidian — Local AI assistant for Obsidian knowledge management.
Run with: streamlit run app.py
"""
import os
import sys
import streamlit as st

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
        page = st.radio(
            "Navigate",
            ["Dashboard", "Semantic Search", "Note Analyser", "Link Suggester",
             "Zettelkasten Advisor", "Research Questions", "Generate Zettel"],
            index=["Dashboard", "Semantic Search", "Note Analyser", "Link Suggester",
                   "Zettelkasten Advisor", "Research Questions", "Generate Zettel"]
            .index(st.session_state.page),
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


if __name__ == "__main__":
    main()
