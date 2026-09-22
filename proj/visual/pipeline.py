from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from visual.models import (
    FigureRecord,
    PageVisualAnalysis,
    VisualAnalysisOutput,
)
from visual.page_renderer import crop_figure, render_all_pages_to_pil
from visual.figure_detector import detect_figures_on_page
from visual.figure_analyzer import analyze_crop


def slugify_subject(subject: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")
    return slug or "subject"


def process_pdf_visuals(
    pdf_path: str,
    subject: str,
    grade: str,
    chapters_curriculum: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for PDF visual analysis pipeline.
    Renders pages -> detects figures -> crops -> classifies & analyzes -> links to curriculum text -> writes JSON.
    Runs isolated from curriculum schema and handles failures gracefully.
    """
    subject_slug = slugify_subject(subject)
    output_filename = f"visual_analysis_{subject_slug}.json"
    result_pages: List[PageVisualAnalysis] = []

    print(f"\n[Visual Pipeline] Starting visual analysis for '{pdf_path}'...")

    try:
        rendered_pages = render_all_pages_to_pil(pdf_path, scale=2.0)
        total_pages = len(rendered_pages)
        print(f"[Visual Pipeline] Successfully rendered {total_pages} page(s).")

        for page_num, page_image, page_text in rendered_pages:
            print(f"[Visual Pipeline] Detecting figures on page {page_num}/{total_pages}...")
            detected_raw = detect_figures_on_page(
                image=page_image,
                page_number=page_num,
                total_pages=total_pages,
                api_key=api_key,
            )

            page_figures: List[FigureRecord] = []

            for raw_fig in detected_raw:
                bbox = raw_fig["bbox"]
                fig_id = raw_fig["figure_id"]

                fig_label = str(raw_fig.get("label") or "")
                # Crop figure region
                crop_img = crop_figure(page_image, bbox, padding=0.0)

                rich_context = f"Page {page_num} textbook content for {subject}.\n\nPage Text Snippet:\n{page_text[:1500]}"

                # Classify and run specialized domain analysis
                analyzed = analyze_crop(
                    crop_img,
                    label=fig_label,
                    page_context=rich_context,
                    api_key=api_key,
                )


                if analyzed is None:
                    # Filtered out (e.g. data table or single line equation)
                    continue


                # Link visual to nearest chapter/topic/subtopic from parsed curriculum
                linked_chap, linked_top, linked_sub = link_figure_to_curriculum(
                    page_num=page_num,
                    chapters=chapters_curriculum or [],
                )

                labels_list = []
                if raw_fig.get("label"):
                    labels_list.append(str(raw_fig["label"]))

                fig_record = FigureRecord(
                    figure_id=fig_id,
                    page_number=page_num,
                    figure_type=analyzed["figure_type"],
                    bbox=bbox,
                    description=analyzed["description"],
                    confidence=float(analyzed["confidence"]),
                    labels=labels_list,
                    details=analyzed.get("details"),
                    related_text=format_belongs_to_text(raw_fig.get("belongs_to")),
                    related_chapter=linked_chap,
                    related_topic=linked_top,
                    related_subtopic=linked_sub,
                )

                # Validate figure record to exclude 'Do you know?', 'Interesting fact', and page banners
                try:
                    from image_extractor import is_valid_figure
                    if not is_valid_figure(fig_record.model_dump()):
                        continue
                except Exception:
                    pass

                # Save extracted image figure to extracted_images/
                try:
                    img_dir = Path("extracted_images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    crop_img.save(img_dir / f"{fig_id}.png", "PNG")
                except Exception as ie:
                    print(f"Warning: Failed to save extracted image {fig_id}: {ie}")

                page_figures.append(fig_record)

            if page_figures or page_num <= total_pages:
                result_pages.append(
                    PageVisualAnalysis(
                        page_number=page_num,
                        figures=page_figures,
                    )
                )

    except Exception as e:
        print(f"\n[Visual Pipeline WARNING] Visual analysis encountered an error: {e}")
        print("[Visual Pipeline] Continuing curriculum extraction without visual sidecar data.")

    # Wrap in VisualAnalysisOutput
    output_obj = VisualAnalysisOutput(
        pdf_path=pdf_path,
        subject=subject,
        grade=grade,
        pages=result_pages,
    )

    out_dict = output_obj.model_dump()

    # Save visual analysis sidecar JSON
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
        print(f"[Visual Pipeline] Wrote visual analysis to '{output_filename}'")
    except Exception as e:
        print(f"[Visual Pipeline WARNING] Failed to write '{output_filename}': {e}")

    # Re-generate curriculum.html to incorporate visual figures & images
    try:
        from json_to_html import convert_json_to_html
        output_json = f"output_{subject_slug}.json"
        if os.path.exists(output_json):
            convert_json_to_html(output_json, "curriculum.html")
            print("[Visual Pipeline] Updated 'curriculum.html' with visual figures & images.")
    except Exception as e:
        print(f"[Visual Pipeline WARNING] Failed to update curriculum.html: {e}")

    return out_dict


def format_belongs_to_text(belongs_to: Optional[Dict[str, Any]]) -> Optional[str]:
    if not belongs_to or not isinstance(belongs_to, dict):
        return None

    q_no = belongs_to.get("q_no")
    sub_q_no = belongs_to.get("sub_q_no")

    parts = []
    if q_no is not None:
        parts.append(f"Question {q_no}")
    if sub_q_no:
        parts.append(f"Part {sub_q_no}")

    return " ".join(parts) if parts else None


def link_figure_to_curriculum(
    page_num: int, chapters: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Match figure page position to nearest curriculum chapter, topic, and subtopic.
    """
    if not chapters:
        return None, None, None

    # Estimate chapter by page distribution or chapter index
    num_chaps = len(chapters)
    chap_idx = min(num_chaps - 1, max(0, (page_num - 1) // max(1, 10)))

    chap = chapters[chap_idx]
    chap_id = chap.get("id") or chap.get("chapterTitle")

    topics = chap.get("topics", [])
    if not topics:
        return chap_id, None, None

    top = topics[0]
    top_id = top.get("topic_id") or top.get("title")

    subtopics = top.get("subtopics", [])
    if not subtopics:
        return chap_id, top_id, None

    sub = subtopics[0]
    sub_id = sub.get("sub_topic_id") or sub.get("title")

    return chap_id, top_id, sub_id
