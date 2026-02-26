"""
Ollama LLM interface for Obsidian analysis tasks.
All prompts are designed for local 7B-class models like Qwen2.
"""
from typing import List, Dict, Generator
import ollama

from config import LLM_MODEL, MAX_NOTE_CHARS_FOR_LLM


def _chat(messages: List[Dict], stream: bool = False):
    """Low-level Ollama chat call."""
    if stream:
        return ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response.message.content


def stream_response(messages: List[Dict]) -> Generator[str, None, None]:
    """Yield tokens from streaming Ollama response."""
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        token = chunk.message.content
        if token:
            yield token


# ── Prompt templates ──────────────────────────────────────────────────────────

def analyze_note(title: str, content: str, related_summaries: List[str] = None) -> str:
    """
    Deep analysis of a single note: atomic idea check, gaps, tags, Zettelkasten fit.
    Returns streaming generator.
    """
    truncated = content[:MAX_NOTE_CHARS_FOR_LLM]
    related_block = ""
    if related_summaries:
        related_block = "\n\nSemantically related notes in the vault:\n" + \
                        "\n".join(f"- {s}" for s in related_summaries[:5])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge management expert specialising in Zettelkasten method "
                "and academic research note organisation. Give concise, actionable advice."
            ),
        },
        {
            "role": "user",
            "content": f"""Analyse this Obsidian note and give structured feedback:

**Title:** {title}
**Content:**
{truncated}
{related_block}

Please provide:
1. **Atomic Idea Check** – Does this note contain one clear idea? If not, suggest how to split it.
2. **Zettelkasten Score** (1-10) – How well does it fit atomic note principles? Explain briefly.
3. **Suggested Tags** – 3-6 specific tags as Obsidian frontmatter, e.g. `tags: ["epistemology", "qualitative-method", "research"]` (lowercase, hyphenated, quoted strings).
4. **Backlink Suggestions** – Concepts that likely exist or should exist as separate notes.
5. **Gaps & Expansion** – What is missing or underdeveloped?
6. **Rewrite Hint** – Suggest a stronger permanent note title as a declarative sentence (e.g. "Banal language erodes meaning by normalising cliché"), suitable for Obsidian `title:` frontmatter.

Be direct and specific. Skip preamble.""",
        },
    ]
    return stream_response(messages)


def suggest_links(note_title: str, note_snippet: str, candidates: List[Dict]) -> str:
    """
    Given a note and semantically similar candidates, suggest which links to add.
    Returns streaming generator.
    """
    candidate_lines = "\n".join(
        f"- [[{c['filename']}]] – {c['snippet'][:120]}..."
        for c in candidates[:8]
    )

    messages = [
        {
            "role": "system",
            "content": "You are a Zettelkasten expert. Suggest meaningful wiki-links between notes.",
        },
        {
            "role": "user",
            "content": f"""Note to enrich: **{note_title}**
Snippet: {note_snippet[:400]}

Candidate notes (semantically similar):
{candidate_lines}

For each candidate, decide:
- LINK: should be linked from this note (explain why in one line)
- SKIP: not relevant enough

Then write the wikilinks to add as Obsidian markdown, e.g.:
See also: [[Note A]], [[Note B]]

Be concise.""",
        },
    ]
    return stream_response(messages)


def zettelkasten_overview(stats: Dict, sample_orphans: List[str],
                          sample_hubs: List[str], tag_sample: List[str]) -> str:
    """
    Vault-wide Zettelkasten health report.
    Returns streaming generator.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a knowledge management consultant. Give a strategic vault audit.",
        },
        {
            "role": "user",
            "content": f"""Obsidian vault audit — give a Zettelkasten health report:

**Stats:**
- Total notes: {stats['total']}
- Orphan notes (no links in or out): {stats['orphans']}
- Notes with no tags: {stats['no_tags']}
- Average word count: {stats['avg_words']:.0f}
- Average outgoing links per note: {stats['avg_links']:.1f}

**Top hub notes (most linked-to):**
{chr(10).join(f"- {h}" for h in sample_hubs[:5])}

**Sample orphan notes:**
{chr(10).join(f"- {o}" for o in sample_orphans[:8])}

**Common tags:**
{', '.join(tag_sample[:15])}

Provide:
1. **Overall Health** – brief verdict on the vault's knowledge graph quality.
2. **Top 3 Problems** – most impactful issues to fix first.
3. **Quick Wins** – 3 immediate actions (specific, doable in <30 min).
4. **Structure Recommendation** – folder/MOC strategy that suits this vault.
5. **Missing Index Notes** – suggest 2-3 Map of Content (MOC) notes to create.

Keep it under 400 words. Be direct.""",
        },
    ]
    return stream_response(messages)


def generate_zettel(raw_content: str, source_title: str) -> str:
    """
    Convert a passage/note into a proper Zettelkasten permanent note.
    Returns streaming generator.
    """
    truncated = raw_content[:MAX_NOTE_CHARS_FOR_LLM]
    messages = [
        {
            "role": "system",
            "content": "You write concise, permanent Zettelkasten notes in Markdown.",
        },
        {
            "role": "user",
            "content": f"""Convert this raw note into a proper permanent Zettelkasten note.

Source: "{source_title}"
Content:
{truncated}

Output a complete Markdown note using exactly this format:
---
tags: ["tag1", "tag2", "tag3", "tag4"]
title: "Atomic title — a declarative sentence stating one idea"
---

# Atomic title — a declarative sentence stating one idea

[2-4 paragraph body: state the idea clearly, explain why it matters, give one example]

## Connections
- [[Concept A]] – how it relates
- [[Concept B]] – how it relates

## Source
- {source_title}

Rules:
- tags must be lowercase, hyphenated (e.g. "knowledge-management"), quoted strings in a JSON array
- title must be a complete declarative sentence, not a question, in the frontmatter AND as the H1
- Only output the note. No preamble.""",
        },
    ]
    return stream_response(messages)


def find_research_questions(notes_snippets: List[Dict]) -> str:
    """
    Given a set of notes, suggest research questions and gaps.
    Returns streaming generator.
    """
    notes_block = "\n\n".join(
        f"**{n['title']}**: {n['snippet'][:200]}"
        for n in notes_snippets[:10]
    )
    messages = [
        {
            "role": "system",
            "content": "You are a research advisor. Identify intellectual gaps and generative questions.",
        },
        {
            "role": "user",
            "content": f"""Based on these research notes, suggest:

{notes_block}

1. **5 Research Questions** – questions these notes raise but don't answer.
2. **3 Synthesis Opportunities** – ideas that could be combined into a stronger argument.
3. **Missing Perspectives** – viewpoints or disciplines not represented.
4. **Next Reading Suggestions** – 3 types of sources that would fill gaps (no URLs).

Be specific to the content above.""",
        },
    ]
    return stream_response(messages)


def identify_research_themes(
    top_tags: List[tuple],
    hub_titles: List[str],
    hub_snippets: List[str],
    sub_vaults: List[str],
) -> str:
    """
    Step 1: Cluster tags and hub notes into 3-5 broader research themes.
    Returns full string (non-streaming).
    """
    tags_block = "\n".join(f"  {t} ({c} notes)" for t, c in top_tags[:30])
    hubs_block = "\n".join(
        f"  [{title}]: {snippet}"
        for title, snippet in zip(hub_titles[:8], hub_snippets[:8])
    )
    vaults = ", ".join(sub_vaults) if sub_vaults else "general"

    messages = [
        {
            "role": "system",
            "content": "You are a research analyst who identifies intellectual patterns in a person's body of work.",
        },
        {
            "role": "user",
            "content": f"""Analyse these research notes and identify the 3-5 main intellectual themes.

Research areas / sub-vaults: {vaults}

Most-referenced hub notes (with opening text):
{hubs_block}

Tags by frequency (most written about first):
{tags_block}

Output ONLY in this exact format — one theme per line, nothing else:
THEME: Short Theme Name | One sentence describing what this cluster covers

Rules:
- Group related tags together under a meaningful broader label
- Theme names should be broad (e.g. "Knowledge Systems & Learning" not "zettelkasten")
- Capture real intellectual substance, not surface keywords
- 3 themes minimum, 5 maximum
- No preamble, no explanation, only THEME: lines""",
        },
    ]
    return _chat(messages, stream=False)


def generate_professional_bio(
    themes: List[Dict],
    hub_titles: List[str],
    top_tags: List[str],
) -> Generator:
    """
    Step 2: Write a professional first-person biography from extracted themes.
    Returns streaming generator.
    """
    themes_block = "\n".join(
        f"  - {t['name']}: {t['description']}" for t in themes
    )
    hubs = ", ".join(hub_titles[:6])
    tags = ", ".join(top_tags[:12])

    messages = [
        {
            "role": "system",
            "content": (
                "You write clear, professional biographies for researchers and independent thinkers. "
                "You turn a collection of interests into a coherent intellectual identity."
            ),
        },
        {
            "role": "user",
            "content": f"""Write a professional 150-200 word biography for someone with these research interests.

Their main research themes:
{themes_block}

Their most-referenced topics: {hubs}
Their key research tags: {tags}

Requirements:
- 150-200 words
- First person: "I work on...", "My research...", "I am interested in..."
- Opens with a single sentence capturing their overall intellectual focus
- Weaves through the themes as flowing prose — not a bullet list
- Concrete and specific — mention actual concepts from the themes, not vague generalities
- Conversational but professional — suitable for answering "what do you do?" at an event
- Ends with what drives the work or the broader question it all points toward
- No jargon, no buzzwords, no mention of note-taking apps
- Output only the bio, no preamble""",
        },
    ]
    return stream_response(messages)


def generate_carousel_items(keyword: str, notes_snippets: List[Dict]) -> str:
    """
    Generate flashcard pairs for the Idea Carousel: a short term and one punchy sentence.
    Returns the full response string (non-streaming).
    """
    notes_block = "\n\n".join(
        f"Note — {n['title']}:\n{n['snippet'][:300]}"
        for n in notes_snippets[:8]
    )
    messages = [
        {
            "role": "system",
            "content": "You create concise flashcards from research notes. Each card has a short label and one crisp sentence.",
        },
        {
            "role": "user",
            "content": f"""From these notes about "{keyword}", generate flashcard carousel items.

{notes_block}

Reply with 10-14 items in EXACTLY this format, one per line:
ITEM: Term | One sentence capturing its spirit

Rules:
- Mix concepts, author names, key claims, and frameworks
- Term: 1-4 words, punchy
- Sentence: 8-15 words, memorable — the essence, not a definition
- No numbering, no preamble, only ITEM: lines
- Example:
ITEM: Qualia | The felt redness of red that no brain scan can ever explain
ITEM: David Chalmers | The philosopher who named the gap between neurons and experience""",
        },
    ]
    return _chat(messages, stream=False)


def extract_conversation_prompts(keyword: str, notes_snippets: List[Dict]) -> str:
    """
    Extract concepts, authors, and ideas from notes for live conversation prompting.
    Returns the full response string (non-streaming) in pipe-delimited format.
    """
    notes_block = "\n\n".join(
        f"Note — {n['title']}:\n{n['snippet'][:300]}"
        for n in notes_snippets[:6]
    )
    messages = [
        {
            "role": "system",
            "content": "You extract key concepts, people, and ideas from research notes. Be concise.",
        },
        {
            "role": "user",
            "content": f"""From these notes about "{keyword}", extract conversation prompts.

{notes_block}

Reply in EXACTLY this format — three lines, pipe-separated values, nothing else:
CONCEPTS: concept1 | concept2 | concept3 | concept4 | concept5
AUTHORS: name1 | name2 | name3 | name4
IDEAS: short idea 1 | short idea 2 | short idea 3 | short idea 4

Rules:
- CONCEPTS: theories, frameworks, phenomena, technical terms (2-4 words each)
- AUTHORS: real thinkers, researchers, writers relevant to this topic
- IDEAS: specific claims, insights, or questions (5-9 words each)
- Include 4-6 items per line
- If nothing fits a category, write: AUTHORS: none
- Output only the three lines. No preamble, no explanation.""",
        },
    ]
    return _chat(messages, stream=False)


def summarize_note(title: str, content: str) -> str:
    """One-paragraph summary. Returns streaming generator."""
    truncated = content[:MAX_NOTE_CHARS_FOR_LLM]
    messages = [
        {"role": "system", "content": "Summarise academic notes concisely."},
        {
            "role": "user",
            "content": f"Summarise this note in 2-3 sentences:\n\n**{title}**\n\n{truncated}",
        },
    ]
    return stream_response(messages)
