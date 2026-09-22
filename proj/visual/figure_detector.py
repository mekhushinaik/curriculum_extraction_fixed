from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from groq import Groq
from visual.page_renderer import image_to_base64_jpeg
from tools.dataset_prep.prompts.figure_detection import (
    SYSTEM_PROMPT as DETECTOR_SYSTEM_PROMPT,
    build_user_message as build_detector_user_message,
)

# Load .env automatically if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")



def detect_figures_on_page(
    image: Image.Image,
    page_number: int,
    total_pages: int,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect visually distinct figures on a rendered PDF page.
    Returns a list of dicts containing bbox, label, confidence, belongs_to.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print(f"Warning: GROQ_API_KEY missing. Skipping VLM figure detection for page {page_number}.")
        return []

    client = Groq(api_key=groq_api_key)
    user_text_prompt = build_detector_user_message(page_number, total_pages)
    base64_img = image_to_base64_jpeg(image)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": DETECTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        content = response.choices[0].message.content or ""
        vlm_results = parse_detector_response(content, page_number)
        if vlm_results:
            return vlm_results
        return detect_figures_via_cv(image, page_number)

    except Exception as e:
        # Fallback if model is text-only (expects string content) or vision unavailable
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": DETECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            vlm_results = parse_detector_response(content, page_number)
            if vlm_results:
                return vlm_results
        except Exception:
            pass

        return detect_figures_via_cv(image, page_number)


def detect_figures_via_cv(image: Image.Image, page_number: int) -> List[Dict[str, Any]]:
    """
    OpenCV contour / computer vision layout fallback to detect visual figures, maps, diagrams, graphs.
    """
    try:
        import cv2
        import numpy as np

        img_np = np.array(image.convert("L"))
        h, w = img_np.shape

        _, thresh = cv2.threshold(img_np, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter for substantial visual regions (maps, diagrams, graphs)
            if bw > w * 0.15 and bh > h * 0.15:
                x0 = max(0.0, x / w)
                y0 = max(0.0, y / h)
                x1 = min(1.0, (x + bw) / w)
                y1 = min(1.0, (y + bh) / h)
                boxes.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)])

        boxes.sort(key=lambda b: b[1])
        valid_figures = []

        for idx, bbox in enumerate(boxes):
            valid_figures.append(
                {
                    "figure_id": f"fig-p{page_number}-{idx+1:02d}",
                    "page_number": page_number,
                    "bbox": bbox,
                    "confidence": 0.85,
                    "label": f"Figure {page_number}.{idx+1}",
                    "belongs_to": {"q_no": None, "sub_q_no": None},
                }
            )

        return valid_figures
    except Exception as cve:
        print(f"CV detection error on page {page_number}: {cve}")
        return []




def parse_detector_response(response_text: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Safely parse JSON response from figure detection model.
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except Exception:
        # Fallback regex extraction of json object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    raw_annotations = data.get("figure_bbox_annotations", [])
    valid_figures = []

    for idx, item in enumerate(raw_annotations):
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox
        # Validate coordinates are valid numbers
        try:
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        except (ValueError, TypeError):
            continue

        # Sanity check bbox dimensions & bounds
        if x0 >= x1 or y0 >= y1:
            continue

        # Ignore tiny full-page borders or tiny points
        width_norm = x1 - x0
        height_norm = y1 - y0

        if width_norm < 0.02 and height_norm < 0.02:
            continue

        valid_figures.append(
            {
                "figure_id": f"fig-p{page_number}-{idx+1:02d}",
                "page_number": page_number,
                "bbox": [x0, y0, x1, y1],
                "confidence": float(item.get("confidence", 0.9)),
                "label": item.get("label"),
                "belongs_to": item.get("belongs_to"),
            }
        )

    return valid_figures
