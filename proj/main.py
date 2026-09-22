import argparse
import os

from graph import (
    build_graph,
    slugify_subject,
)


def run(
    pdf_path: str,
    subject: str,
    grade: str,
    curriculum: str,
    api_key: str | None = None,
    disable_visual: bool = False,
):
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    app = build_graph()

    res = app.invoke(
        {
            "pdf_path": pdf_path,

            "subject": subject,

            "subject_slug": slugify_subject(
                subject
            ),

            "grade": grade,

            "curriculum": curriculum,

            "chunks": [],

            "chapters": [],

            "errors": [],

            "retry_count": 0,
        }
    )

    if not disable_visual:
        try:
            from visual.pipeline import process_pdf_visuals

            process_pdf_visuals(
                pdf_path=pdf_path,
                subject=subject,
                grade=grade,
                chapters_curriculum=res.get("chapters", []),
                api_key=api_key,
            )
        except Exception as e:
            print(f"Warning: Visual analysis pipeline skipped/failed: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf",
        required=True,
    )

    parser.add_argument(
        "--subject",
        required=True,
    )

    parser.add_argument(
        "--grade",
        required=True,
    )

    parser.add_argument(
        "--curriculum",
        required=True,
    )

    parser.add_argument(
        "--api-key",
        required=False,
        default=None,
        help="Optional Groq API key (can also be stored in .env or GROQ_API_KEY env var)",
    )

    parser.add_argument(
        "--disable-visual",
        action="store_true",
        default=False,
        help="Optional flag to disable visual analysis sidecar pipeline",
    )

    args = parser.parse_args()

    run(
        args.pdf,
        args.subject,
        args.grade,
        args.curriculum,
        args.api_key,
        args.disable_visual,
    )