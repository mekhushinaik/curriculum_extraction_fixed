from __future__ import annotations

import json
import os
import re
import time

from typing import List, Literal, TypedDict

import instructor
from groq import Groq
from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from schema import AIChapter, AITopic, AISubTopic, FinalChapter, SubjectCurriculum
from parsers import (
    parse_with_docling,
    chunk_markdown_by_chapter,
    clean_markdown_content,
    validate_markdown_has_text,
)


from pathlib import Path

# ============================================================
# LOAD .ENV AUTOMATICALLY IF PRESENT
# ============================================================

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 3

STRUCTURING_MODEL = os.getenv("STRUCTURING_MODEL", "openai/gpt-oss-20b")

# Safe maximum chars per single LLM call (~3000 tokens input, within Groq context window)
CHARS_PER_LLM_CALL = 12000


# ============================================================
# GROQ CLIENT HELPER
# ============================================================

def get_groq_client() -> instructor.Instructor | None:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    return instructor.from_groq(Groq(api_key=groq_api_key), mode=instructor.Mode.JSON)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert curriculum structuring agent.

Extract the REAL curriculum hierarchy from the supplied textbook content.

The structure must contain:

Chapter
    -> Topics
        -> Subtopics

CRITICAL INSTRUCTIONS:

1. Extract ONLY actual section headings, subheadings, and real concepts discussed in the textbook text.

2. Do NOT generate generic fallback titles. The following types of output are STRICTLY FORBIDDEN:
   - "Overview of [Chapter]"
   - "Introduction to [Chapter]"
   - "Key Concepts in [Chapter]"
   - "General Studies"
   - Descriptions containing "AI-generated explanation" or template phrases.

3. Do NOT use the chapter title as a topic title. A topic title must represent a specific section or concept discussed within the chapter.

4. Do NOT extract generic textbook callout boxes, activity headings, or sidebars (e.g. "Activity", "Activities", "Do you know?", "Find out", "Let's discuss") as topic titles or topic IDs. These are not main curriculum topics.

5. The chapter theme must NOT be "General Studies". It must be derived from the subject and chapter content (e.g. "Geography", "Social Science", "Chemistry", "Water Conservation").

6. SUBTOPIC EXTRACTION & AI GENERATION RULE:
   - Case 1 (Textbook Subtopics): If a topic has explicit textbook subheadings/subtopics present in the text, extract those subtopics with source = "tb". Do NOT generate additional AI subtopics when explicit textbook subtopics already exist.
   - Case 2 (AI-Generated Subtopics): ONLY when a textbook topic has NO explicit subheadings in the text, you MUST analyze the content/paragraphs under that topic and generate meaningful subtopics based strictly on the discussed concepts. Assign source = "ai" ONLY to these generated subtopics.
   - Every topic MUST contain at least one subtopic (len(subtopics) >= 1). Never return an empty subtopics list.

7. Every subtopic MUST have a specific description that explains what the subtopic covers based ONLY on the textbook text.
   - Do NOT write generic text like "AI-generated explanation of the topic".
   - Do NOT repeat the title without explanation.

8. Do NOT generate IDs (id, topic_id, sub_topic_id). Python will generate all IDs.

9. Schema JSON Key Names (MUST be exact):
   - Chapter object keys: "chapterNumber", "chapterTitle", "theme", "topics"
   - Topic object keys: "title", "source", "subtopics" (Do NOT use "topicTitle")
   - Subtopic object keys: "title", "description", "source" (Do NOT use "subtopicTitle")
   - "source" values can ONLY be: "tb" or "ai"

Return ONLY valid JSON matching the AIChapter schema.
"""


# ============================================================
# PIPELINE STATE
# ============================================================

class PipelineState(TypedDict, total=False):

    pdf_path: str

    strategy: Literal[
        "docling_text"
    ]

    subject: str
    subject_slug: str
    grade: str
    curriculum: str

    chunks: List[str]
    chapters: List[dict]

    errors: List[str]
    retry_count: int
    failed_indices: List[int]  # chapter indices (0-based) that failed validation


# ============================================================
# SUBJECT SLUG
# ============================================================

def slugify_subject(
    subject: str,
) -> str:

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        subject.lower(),
    ).strip("-")

    return slug or "subject"


# ============================================================
# PARSE PDF
# ============================================================

def parse_node(
    state: PipelineState,
) -> PipelineState:

    print("Parsing PDF...")

    markdown = parse_with_docling(
        state["pdf_path"]
    )

    # 1. Clean extracted text IMMEDIATELY after docling parsing and BEFORE chunking
    cleaned_markdown = clean_markdown_content(markdown)

    # 2. Check minimum content quality (at least 30 meaningful words)
    validate_markdown_has_text(cleaned_markdown, min_words=30)

    # 3. Chunk markdown by chapter
    chunks = chunk_markdown_by_chapter(cleaned_markdown)

    # 4. Filter chunks: reject any chunk containing only placeholders or insufficient words
    valid_chunks = []
    for chunk in chunks:
        c_clean = clean_markdown_content(chunk)
        words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", c_clean)
        if len(words) >= 30:
            valid_chunks.append(c_clean)

    if not valid_chunks:
        raise ValueError(
            f"Extracted textbook content from '{state['pdf_path']}' contains insufficient meaningful text. "
            f"The PDF may contain only images, garbage, or unreadable OCR content."
        )

    state["chunks"] = valid_chunks

    print(
        f"Found {len(valid_chunks)} valid chapter chunks."
    )

    return state


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def clean_json_response(
    text: str,
) -> str:

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# CONTENT SEGMENTATION HELPER
# ============================================================

def _split_content_into_segments(content: str, max_chars: int) -> List[str]:
    """Split content into segments of at most max_chars, breaking at newline boundaries."""
    if len(content) <= max_chars:
        return [content]
    segments = []
    start = 0
    while start < len(content):
        end = start + max_chars
        if end < len(content):
            # Prefer splitting at a newline to avoid cutting mid-sentence
            nl = content.rfind("\n", start, end)
            if nl > start:
                end = nl
        segments.append(content[start:end].strip())
        start = end
    return [s for s in segments if s]


# ============================================================
# SINGLE-SEGMENT LLM CALL (exponential back-off on rate limits)
# ============================================================

def _call_llm_for_segment(
    client,
    content: str,
    subject_slug: str,
    chapter_num: int,
    error_feedback: str = "",
) -> dict:
    """Call Groq + Instructor for one content segment; returns AIChapter as dict."""

    user_message = f"""
Subject: {subject_slug}

Chapter number: {chapter_num}

TEXTBOOK CONTENT:

{content}
"""

    if error_feedback:
        user_message += f"""

PREVIOUS VALIDATION ERROR:

{error_feedback}

Fix the error completely. Do NOT use generic fallback titles like 'Overview of...', 'Key Concepts in...', or 'General Studies'.
Return the COMPLETE corrected chapter matching AIChapter schema.
"""

    for attempt in range(4):
        try:
            chapter_obj = client.chat.completions.create(
                model=STRUCTURING_MODEL,
                response_model=AIChapter,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
                max_retries=3,
            )
            return AIChapter.model_validate(chapter_obj).model_dump()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                # Exponential back-off: 3.5s, 7s, 14s, 28s
                backoff = 3.5 * (2 ** attempt)
                print(f"  Rate limit hit (attempt {attempt + 1}), backing off {backoff:.1f}s...")
                time.sleep(backoff)
            elif attempt == 3:
                # Fallback to Mode.TOOLS if Mode.JSON failed on final attempt
                try:
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    tools_client = instructor.from_groq(
                        Groq(api_key=groq_api_key), mode=instructor.Mode.TOOLS
                    )
                    chapter_obj = tools_client.chat.completions.create(
                        model=STRUCTURING_MODEL,
                        response_model=AIChapter,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.1,
                        max_tokens=4096,
                        max_retries=3,
                    )
                    return AIChapter.model_validate(chapter_obj).model_dump()
                except Exception as e2:
                    print(f"\nAI STRUCTURE EXTRACTION FAILED: {e2}")
                    raise RuntimeError(
                        f"Failed to extract real curriculum structure via Groq AI. Details: {e2}"
                    )

    raise RuntimeError(f"Chapter {chapter_num}: LLM call failed after all retries.")


# ============================================================
# STRUCTURE ONE CHAPTER  (large chapters processed in segments)
# ============================================================

def structure_one(
    content: str,
    subject_slug: str,
    chapter_num: int,
    error_feedback: str = "",
) -> dict:

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is missing or empty. "
            "Please set your Groq API key: $env:GROQ_API_KEY='your_api_key'"
        )

    # If content fits in one LLM call, process it directly
    if len(content) <= CHARS_PER_LLM_CALL:
        return _call_llm_for_segment(
            client, content, subject_slug, chapter_num, error_feedback
        )

    # Large chapter: split into sequential segments, merge topics
    segments = _split_content_into_segments(content, CHARS_PER_LLM_CALL)
    print(
        f"  Chapter {chapter_num} is large ({len(content)} chars); "
        f"processing {len(segments)} segments sequentially..."
    )

    merged = None
    for seg_idx, segment in enumerate(segments):
        if seg_idx > 0:
            time.sleep(2.0)  # respect rate limits between segments
        # Only pass error_feedback to the first segment (it applies to the chapter, not a segment)
        seg_feedback = error_feedback if seg_idx == 0 else ""
        try:
            seg_result = _call_llm_for_segment(
                client, segment, subject_slug, chapter_num, seg_feedback
            )
            if merged is None:
                merged = seg_result
            else:
                # Accumulate topics from subsequent segments into the merged chapter
                merged["topics"].extend(seg_result.get("topics", []))
        except Exception as e:
            print(
                f"  Warning: segment {seg_idx + 1}/{len(segments)} of "
                f"chapter {chapter_num} failed: {e}"
            )

    if merged is None:
        raise RuntimeError(
            f"All {len(segments)} segments of chapter {chapter_num} failed to process."
        )

    return merged


# ============================================================
# ADD IDS AFTER AI RESPONSE
# ============================================================

def add_ids_to_chapter(
    chapter: dict,
    subject_slug: str,
    chapter_num: int,
) -> dict:

    # --------------------------------------------------------
    # Chapter ID
    # --------------------------------------------------------
    chapter["id"] = (
        f"chap-{subject_slug}-tb-{chapter_num}"
    )

    # --------------------------------------------------------
    # Filter out Activity / Callout topics
    # --------------------------------------------------------
    from schema import FORBIDDEN_TOPIC_TITLES

    filtered_topics = []
    for topic in chapter.get("topics", []):
        t_clean = topic.get("title", "").strip().lower().rstrip("?:!.")
        is_forbidden = False
        for f in FORBIDDEN_TOPIC_TITLES:
            f_clean = f.lower().rstrip("?:!.")
            if t_clean == f_clean or t_clean.startswith(f_clean + " "):
                is_forbidden = True
                break
        if not is_forbidden:
            filtered_topics.append(topic)

    chapter["topics"] = filtered_topics

    # --------------------------------------------------------
    # Topic IDs & Subtopic IDs
    # --------------------------------------------------------
    for topic_index, topic in enumerate(
        chapter["topics"],
        start=1,
    ):
        topic_source = topic.pop(
            "source",
            "tb",
        )

        topic["topic_id"] = (
            f"top-{subject_slug}-"
            f"{topic_source}-"
            f"{chapter_num}-"
            f"{topic_index}"
        )

        for subtopic_index, subtopic in enumerate(
            topic["subtopics"],
            start=1,
        ):
            subtopic_source = subtopic.pop(
                "source",
                "ai",
            )

            subtopic["sub_topic_id"] = (
                f"sub-{subject_slug}-"
                f"{subtopic_source}-"
                f"{chapter_num}-"
                f"{topic_index}-"
                f"{subtopic_index}"
            )

    return chapter


# ============================================================
# STRUCTURE NODE  (retries only failed chapters, not all chapters)
# ============================================================

def structure_node(
    state: PipelineState,
) -> PipelineState:

    error_feedback = (
        state["errors"][-1]
        if state.get("retry_count", 0) > 0 and state.get("errors")
        else ""
    )

    # On retry: reprocess only the specific chapters that failed validation.
    # This avoids wasting API quota by re-sending all chapters.
    failed_indices = state.get("failed_indices", [])
    if state.get("retry_count", 0) > 0 and failed_indices:
        chapters = list(state.get("chapters", []))  # preserve good chapters
        indices_to_process = failed_indices
        print(
            f"Retrying {len(failed_indices)} failed chapter(s): "
            f"{[idx + 1 for idx in failed_indices]}"
        )
    else:
        chapters = []
        indices_to_process = list(range(len(state["chunks"])))

    for pos, chunk_idx in enumerate(indices_to_process):
        i = chunk_idx + 1  # 1-based chapter number
        chunk = state["chunks"][chunk_idx]

        print(f"Structuring chapter {i}...")

        # ----------------------------------------------------
        # AI CREATES CHAPTER STRUCTURE
        # ----------------------------------------------------
        chapter = structure_one(
            content=chunk,
            subject_slug=state["subject_slug"],
            chapter_num=i,
            error_feedback=error_feedback,
        )

        # ----------------------------------------------------
        # PYTHON GENERATES IDs AND REMOVES SOURCE
        # ----------------------------------------------------
        chapter = add_ids_to_chapter(
            chapter=chapter,
            subject_slug=state["subject_slug"],
            chapter_num=i,
        )

        if state.get("retry_count", 0) > 0 and failed_indices:
            chapters[chunk_idx] = chapter  # patch only the failed slot in place
        else:
            chapters.append(chapter)

        if pos < len(indices_to_process) - 1:
            time.sleep(2.0)

    state["chapters"] = chapters

    return state


# ============================================================
# VALIDATE NODE  (validates each chapter individually, tracks which ones failed)
# ============================================================

def validate_node(
    state: PipelineState,
) -> PipelineState:

    validated_chapters = []
    failed_indices: List[int] = []
    new_errors: List[str] = []

    for idx, chapter in enumerate(state["chapters"]):
        try:
            validated = FinalChapter.model_validate(chapter)
            validated_chapters.append(validated.model_dump())
        except ValidationError as e:
            print(f"Validation error in chapter {idx + 1}: {e}")
            validated_chapters.append(chapter)  # preserve original for retry
            failed_indices.append(idx)
            new_errors.append(str(e))

    state["chapters"] = validated_chapters
    state["failed_indices"] = failed_indices

    if new_errors:
        state["errors"] = state.get("errors", []) + new_errors
        state["retry_count"] = state.get("retry_count", 0) + 1
        print(
            f"Validation completed: {len(failed_indices)} chapter(s) failed. "
            f"Will retry chapters: {[idx + 1 for idx in failed_indices]}"
        )
    else:
        state["errors"] = []
        print("Validation successful.")

    return state


# ============================================================
# ROUTING
# ============================================================

def route_after_validate(
    state: PipelineState,
) -> str:

    if not state.get("errors"):
        return "finalize"

    if (
        state.get(
            "retry_count",
            0,
        )
        >= MAX_RETRIES
    ):
        return "finalize"

    return "structure"


# ============================================================
# FINALIZE
# ============================================================

def finalize_node(
    state: PipelineState,
) -> PipelineState:

    output = [
        {
            "curriculum": state[
                "curriculum"
            ],

            "grade": state[
                "grade"
            ],

            "subject": state[
                "subject"
            ],

            "chapters": state[
                "chapters"
            ],
        }
    ]

    try:
        SubjectCurriculum.model_validate(output[0])
    except ValidationError as e:
        print(f"Warning: Curriculum schema validation issue: {e}")

    output_file = (
        f"output_"
        f"{state['subject_slug']}"
        f".json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "SUCCESS!"
    )

    print(
        f"Wrote {output_file}"
    )

    try:
        from json_to_html import convert_json_to_html
        convert_json_to_html(output_file, "curriculum.html")
        print("Wrote curriculum.html")
    except Exception as e:
        print(f"Warning: HTML generation failed: {e}")

    if state.get("errors"):

        print(
            "WARNING: unresolved "
            "validation errors remain."
        )

    return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        PipelineState
    )

    graph.add_node(
        "parse",
        parse_node,
    )

    graph.add_node(
        "structure",
        structure_node,
    )

    graph.add_node(
        "validate",
        validate_node,
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    graph.set_entry_point(
        "parse"
    )

    graph.add_edge(
        "parse",
        "structure",
    )

    graph.add_edge(
        "structure",
        "validate",
    )

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "structure": "structure",
            "finalize": "finalize",
        },
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()