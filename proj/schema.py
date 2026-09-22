from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================
# SOURCE
# ============================================================

Source = Literal["ai", "tb"]


# ============================================================
# FORBIDDEN GENERIC PATTERNS & VALIDATION
# ============================================================

import re

FORBIDDEN_PATTERNS = [
    "<!--",
    "-->",
    "<image>",
    "overview of",
    "introduction to",
    "key concepts in",
    "general studies",
    "ai-generated explanation",
    "foundational concepts",
    "template text",
]


def clean_ocr_artifacts(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"-\s+-", "-", text)
    s = re.sub(r"-\s+", "-", s)
    s = re.sub(r"\s+-", "-", s)
    s = re.sub(
        r"\b([A-Za-z])\s+([A-Za-z]{2,})",
        lambda m: m.group(2) if m.group(1).lower() == m.group(2)[0].lower() else m.group(0),
        s,
    )
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def validate_string_content(v: str, field_name: str) -> str:
    if not isinstance(v, str):
        return v
    s = clean_ocr_artifacts(v.strip())
    if not s:
        raise ValueError(f"{field_name} cannot be empty")
    s_lower = s.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in s_lower:
            raise ValueError(
                f"{field_name} contains forbidden generic/placeholder pattern '{pattern}': '{v}'"
            )
    return s


# ============================================================
# AI-GENERATED STRUCTURE
# ============================================================

class AISubTopic(BaseModel):
    title: str = Field(
        alias="title",
        min_length=1,
        description="Subtopic title extracted from textbook subheading or concept discussed. CRITICAL: Use key 'title' (NOT 'subtopicTitle')."
    )
    description: str = Field(
        min_length=1,
        description="Detailed description explaining what the subtopic covers based ONLY on the textbook text. Do NOT use generic template text or write 'AI-generated'."
    )
    source: Source = Field(
        description="Must be 'tb' if directly from textbook heading/text, or 'ai' if generated from concepts. CRITICAL: Use key 'source'."
    )

    @field_validator("title", "description", mode="after")
    @classmethod
    def check_subtopic_fields(cls, v: str, info) -> str:
        return validate_string_content(v, info.field_name)


FORBIDDEN_TOPIC_TITLES = [
    "activity",
    "activities",
    "do you know",
    "do you know?",
    "find out",
    "let's discuss",
    "lets discuss",
]


def validate_topic_title(v: str) -> str:
    s = validate_string_content(v, "topic.title")
    s_clean = s.strip().lower().rstrip("?:!.")
    for forbidden in FORBIDDEN_TOPIC_TITLES:
        forbidden_clean = forbidden.lower().rstrip("?:!.")
        if s_clean == forbidden_clean or s_clean.startswith(forbidden_clean + " "):
            raise ValueError(
                f"Topic title '{v}' is a callout/activity box header and cannot be used as a topic title or topic_id."
            )
    return s


class AITopic(BaseModel):
    title: str = Field(
        alias="title",
        min_length=1,
        description="Topic title extracted directly from a major textbook section heading or concept. CRITICAL: Use key 'title' (NOT 'topicTitle'). Do NOT use callouts like 'Activity' or 'Do you know?'."
    )
    source: Source = Field(
        description="Must be 'tb' if directly from textbook heading, or 'ai' if generated. CRITICAL: Use key 'source'."
    )
    subtopics: List[AISubTopic] = Field(
        min_length=1,
        description="List of subtopics under this topic (at least 1 required)."
    )

    @field_validator("title", mode="after")
    @classmethod
    def check_topic_title(cls, v: str) -> str:
        return validate_topic_title(v)

    @model_validator(mode="after")
    def validate_subtopics(self):
        if not self.subtopics:
            raise ValueError(
                f"Topic '{self.title}' must have at least one subtopic"
            )
        return self


class AIChapter(BaseModel):
    chapterNumber: str = Field(min_length=1, description="Chapter number e.g. '1', '3'")
    chapterTitle: str = Field(min_length=1, description="Chapter title extracted from textbook title e.g. 'Water Resources'")
    theme: str = Field(min_length=1, description="Subject theme derived from chapter content e.g. 'Geography' or 'Social Science'")

    topics: List[AITopic] = Field(
        min_length=1,
        description="List of topics in the chapter (at least 1 required)."
    )

    @field_validator("chapterTitle", "theme", mode="after")
    @classmethod
    def check_chapter_fields(cls, v: str, info) -> str:
        return validate_string_content(v, info.field_name)

    @model_validator(mode="after")
    def validate_chapter(self):
        if not self.topics:
            raise ValueError(
                f"Chapter '{self.chapterTitle}' must have at least one topic"
            )

        chap_lower = self.chapterTitle.strip().lower()
        for topic in self.topics:
            top_lower = topic.title.strip().lower()
            if top_lower == chap_lower:
                raise ValueError(
                    f"Topic title '{topic.title}' cannot be identical to chapter title '{self.chapterTitle}'"
                )
        return self


# ============================================================
# FINAL OUTPUT STRUCTURE
# ============================================================

class FinalSubTopic(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    sub_topic_id: str = Field(min_length=1)

    @field_validator("title", "description", mode="after")
    @classmethod
    def check_final_subtopic_fields(cls, v: str, info) -> str:
        return validate_string_content(v, info.field_name)


class FinalTopic(BaseModel):
    title: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)

    subtopics: List[FinalSubTopic] = Field(
        min_length=1
    )

    @field_validator("title", mode="after")
    @classmethod
    def check_final_topic_title(cls, v: str) -> str:
        return validate_topic_title(v)

    @model_validator(mode="after")
    def validate_subtopics(self):
        if not self.subtopics:
            raise ValueError(
                f"Topic '{self.title}' must have at least one subtopic"
            )
        return self


class FinalChapter(BaseModel):
    id: str = Field(min_length=1)

    chapterNumber: str = Field(min_length=1)
    chapterTitle: str = Field(min_length=1)
    theme: str = Field(min_length=1)

    topics: List[FinalTopic] = Field(
        min_length=1
    )

    @field_validator("chapterTitle", "theme", mode="after")
    @classmethod
    def check_final_chapter_fields(cls, v: str, info) -> str:
        return validate_string_content(v, info.field_name)

    @model_validator(mode="after")
    def validate_chapter(self):
        if not self.topics:
            raise ValueError(
                f"Chapter '{self.chapterTitle}' must have at least one topic"
            )

        chap_lower = self.chapterTitle.strip().lower()
        for topic in self.topics:
            top_lower = topic.title.strip().lower()
            if top_lower == chap_lower:
                raise ValueError(
                    f"Topic title '{topic.title}' cannot be identical to chapter title '{self.chapterTitle}'"
                )
        return self


# ============================================================
# COMPLETE CURRICULUM
# ============================================================

class SubjectCurriculum(BaseModel):
    curriculum: str = Field(min_length=1)
    grade: str = Field(min_length=1)
    subject: str = Field(min_length=1)

    chapters: List[FinalChapter] = Field(
        min_length=1
    )