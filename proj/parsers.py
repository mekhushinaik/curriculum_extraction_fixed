from __future__ import annotations

import re
from pathlib import Path
from typing import List


from schema import clean_ocr_artifacts


def clean_markdown_content(markdown: str) -> str:
    """
    Filter out Docling/OCR artifacts, HTML comments (<!-- image -->, <!-- ... -->),
    markdown image syntax (![...](...)), and empty headings (#, ##, ###)
    while preserving actual surrounding textbook text and real headings.
    """
    if not markdown:
        return ""

    # 1. Remove all HTML comments (e.g. <!-- image -->, <!-- PAGE_BREAK -->, <!-- any text -->)
    clean = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)

    # 2. Remove markdown image syntax ![alt](url)
    clean = re.sub(r"!\[.*?\]\(.*?\)", "", clean)

    # 3. Line-by-line cleaning: remove empty headings and standalone image artifact lines
    cleaned_lines = []
    for line in clean.splitlines():
        s = line.strip()

        # Skip lines that are empty headings like '#', '##', '###', '####'
        if re.match(r"^#{1,6}\s*$", s):
            continue

        # Skip lines containing standalone image placeholder artifacts
        if s.lower() in ("image", "<image>", "<!-- image -->", "<!--image-->"):
            continue

        cleaned_lines.append(clean_ocr_artifacts(line))

    result = "\n".join(cleaned_lines)
    # Collapse multiple consecutive blank lines to double newlines
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return clean_ocr_artifacts(result)


def validate_markdown_has_text(markdown: str, min_words: int = 30) -> None:
    """
    Validation check: raise ValueError if parsed Markdown contains insufficient meaningful text.
    """
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", markdown)
    if len(words) < min_words:
        raise ValueError(
            f"Parsed PDF markdown contains insufficient meaningful text ({len(words)} words found, minimum {min_words} required). "
            f"The PDF may be unreadable, corrupted, or missing OCR text. Please check the input file."
        )


def parse_with_docling(pdf_path: str) -> str:
    """
    Parse PDF using fast pypdfium2 text extraction if digital text is present,
    otherwise fallback to docling document converter.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1. Fast digital text extraction via pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        pages_text = []
        for i in range(len(pdf)):
            page_t = pdf[i].get_textpage().get_text_range()
            if page_t and page_t.strip():
                pages_text.append(f"# Page {i+1}\n" + page_t.strip())
        if pages_text:
            fast_md = "\n\n<!-- PAGE_BREAK -->\n\n".join(pages_text)
            cleaned_fast = clean_markdown_content(fast_md)
            words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", cleaned_fast)
            if len(words) >= 100:
                print(f"Successfully extracted {len(words)} words via fast pypdfium2 parser.")
                return cleaned_fast
    except Exception as e:
        print(f"Fast pypdfium2 extraction skipped ({e}), falling back to Docling OCR...")

    # 2. Fallback to Docling OCR
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # disable heavy neural OCR unless needed
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    raw_markdown = result.document.export_to_markdown(
        page_break_placeholder="\n<!-- PAGE_BREAK -->\n"
    )
    cleaned_markdown = clean_markdown_content(raw_markdown)
    validate_markdown_has_text(cleaned_markdown, min_words=30)
    return cleaned_markdown


def chunk_markdown_by_chapter(markdown: str) -> List[str]:
    """
    Split markdown into meaningful chapter chunks containing actual text.
    Filter out any chunk containing only image placeholders or insufficient text.
    """
    cleaned = clean_markdown_content(markdown)
    lines = cleaned.splitlines()
    starts = []

    # Look for chapter/unit headings (e.g. # Chapter 3, ## Unit 1, # 3. Water Resources, etc.)
    heading_pattern = re.compile(
        r"^(?:#{1,3}\s*)?(?:chapter|unit|\d+\.)\b",
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        s = line.strip()
        if heading_pattern.match(s):
            starts.append(i)

    chunks = []
    if starts:
        for j, start in enumerate(starts):
            end = starts[j + 1] if j + 1 < len(starts) else len(lines)
            chunk_text = "\n".join(lines[start:end]).strip()
            chunk_cleaned = clean_markdown_content(chunk_text)

            # Ensure chunk has real content (at least 30 words)
            words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", chunk_cleaned)
            if len(words) >= 30:
                chunks.append(chunk_cleaned)

    # Fallback: if no valid chapter headers were found or split chunks failed, return the full cleaned markdown
    if not chunks:
        cleaned_full = clean_markdown_content(markdown)
        validate_markdown_has_text(cleaned_full, min_words=30)
        chunks = [cleaned_full]

    return chunks
