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
