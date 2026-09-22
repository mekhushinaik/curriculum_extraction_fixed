from __future__ import annotations

import json
import os
import re

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
# STRUCTURE ONE CHAPTER
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

    # Limit chunk text length to 9000 chars to avoid model context / token limits
    if len(content) > 9000:
        content = content[:9000]

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
    except Exception as e:
        # Fallback to Mode.TOOLS if Mode.JSON failed
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            tools_client = instructor.from_groq(Groq(api_key=groq_api_key), mode=instructor.Mode.TOOLS)
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
        except Exception as e2:
            print(f"\nAI STRUCTURE EXTRACTION FAILED: {e2}")
            raise RuntimeError(
                f"Failed to extract real curriculum structure via Groq AI. Details: {e2}"
            )


    chapter = AIChapter.model_validate(chapter_obj)

    return chapter.model_dump()


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
# STRUCTURE NODE
# ============================================================

def structure_node(
    state: PipelineState,
) -> PipelineState:

    chapters = []
    error_feedback = ""

    if (
        state.get("retry_count", 0) > 0
        and state.get("errors")
    ):
        error_feedback = (
            state["errors"][-1]
        )

    for i, chunk in enumerate(
        state["chunks"],
        start=1,
    ):
        print(
            f"Structuring chapter {i}..."
        )

        # ----------------------------------------------------
        # AI CREATES CHAPTER STRUCTURE
        # ----------------------------------------------------
        chapter = structure_one(
            content=chunk,
            subject_slug=state[
                "subject_slug"
            ],
            chapter_num=i,
            error_feedback=error_feedback,
        )

        # ----------------------------------------------------
        # PYTHON GENERATES IDs AND REMOVES SOURCE
        # ----------------------------------------------------
        chapter = add_ids_to_chapter(
            chapter=chapter,
            subject_slug=state[
                "subject_slug"
            ],
            chapter_num=i,
        )

        chapters.append(
            chapter
        )

    state["chapters"] = chapters

    return state


# ============================================================
# VALIDATE NODE
# ============================================================

def validate_node(
    state: PipelineState,
) -> PipelineState:

    try:
        validated_chapters = []

        for chapter in state["chapters"]:
            validated = FinalChapter.model_validate(
                chapter
            )
            validated_chapters.append(
                validated.model_dump()
            )

        state["chapters"] = validated_chapters
        state["errors"] = []

        print(
            "Validation successful."
        )

    except ValidationError as e:
        print(
            "Validation error:"
        )

        print(e)

        state["errors"] = (
            state.get(
                "errors",
                []
            )
            + [
                str(e)
            ]
        )

        state["retry_count"] = (
            state.get(
                "retry_count",
                0
            )
            + 1
        )

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