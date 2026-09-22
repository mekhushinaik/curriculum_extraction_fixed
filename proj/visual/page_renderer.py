from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


def render_all_pages_to_pil(
    pdf_path: str, scale: float = 2.0
) -> List[Tuple[int, Image.Image, str]]:
    """
    Render all pages of a PDF file to PIL Images using pypdfium2, and extract page text.
    Returns a list of (page_number_1indexed, PIL.Image, page_text_str).
    Converts each bitmap immediately to RGB to detach from pdfium C memory buffers.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    if pdfium is None:
        raise ImportError("pypdfium2 is required for PDF page rendering.")

    pdf = pdfium.PdfDocument(pdf_path)
    rendered_pages = []

    try:
        total_pages = len(pdf)
        for i in range(total_pages):
            page_num = i + 1
            try:
                page = pdf[i]
                bitmap = page.render(scale=scale)
                # Convert to RGB immediately to copy C buffer into Python memory
                pil_image = bitmap.to_pil().convert("RGB")
                
                # Extract page text safely
                textpage = page.get_textpage()
                raw_text = textpage.get_text_range() or ""
                clean_text = raw_text.encode("ascii", "ignore").decode("ascii").strip()
                
                rendered_pages.append((page_num, pil_image, clean_text))
            except Exception as pe:
                print(f"Warning: Failed to render PDF page {page_num}: {pe}")
                # Fallback blank page
                blank = Image.new("RGB", (600, 800), color="white")
                rendered_pages.append((page_num, blank, ""))
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return rendered_pages


def render_page_to_pil(
    pdf_path: str, page_number: int, scale: float = 2.0
) -> Image.Image:
    """
    Render a single 1-indexed page of a PDF file to PIL Image.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    if pdfium is None:
        raise ImportError("pypdfium2 is required for PDF page rendering.")

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError(
                f"Invalid page_number {page_number}. PDF has {len(pdf)} pages."
            )

        page = pdf[page_number - 1]
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil().convert("RGB")
        return img
    finally:
        try:
            pdf.close()
        except Exception:
            pass


def crop_figure(
    image: Image.Image, bbox: List[float], padding: float = 0.02
) -> Image.Image:
    """
    Crop a normalized bounding box [ymin, xmin, ymax, xmax] from a PIL Image.
    Applies optional padding so labels/legends/captions are never clipped.
    """
    width, height = image.size
    ymin, xmin, ymax, xmax = bbox

    # Apply padding
    xmin_padded = max(0.0, xmin - padding)
    ymin_padded = max(0.0, ymin - padding)
    xmax_padded = min(1.0, xmax + padding)
    ymax_padded = min(1.0, ymax + padding)

    left = int(xmin_padded * width)
    top = int(ymin_padded * height)
    right = int(xmax_padded * width)
    bottom = int(ymax_padded * height)

    # Ensure minimum 10px dimensions
    if right - left < 10:
        right = min(width, left + 10)
    if bottom - top < 10:
        bottom = min(height, top + 10)

    cropped = image.crop((left, top, right, bottom)).convert("RGB")
    return cropped


def image_to_base64_jpeg(image: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image to a base64 encoded JPEG string.
    """
    import base64

    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
