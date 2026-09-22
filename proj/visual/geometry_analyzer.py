from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from PIL import Image

from groq import Groq
from visual.models import GeometryDetails
from visual.page_renderer import image_to_base64_jpeg

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")

GEOMETRY_SYSTEM_PROMPT = """
You are a mathematical geometry visual analysis expert.
Analyze ONLY what is visually present in the supplied geometry diagram image. Do NOT use outside knowledge to invent missing information.

CRITICAL ANTI-HALLUCINATION RULES:
1. Treat labels, dimensions, arrows, tick marks, angle arcs, parallel markers, perpendicular markers, and coordinate values as visual evidence.
2. Do NOT infer unmarked relationships or measurements. If a triangle visually looks equilateral but has no equal-side markings or angle measurements written, do NOT claim it is equilateral.
3. If a measurement or label is unreadable, report null or omit it. Never guess.

Identify:
- Shape type (triangle, square, rectangle, parallelogram, rhombus, trapezium, polygon, circle, sector, chord, tangent, secant, arc, angle, parallel lines, perpendicular lines, coordinate plane, transformation figure, construction figure).
- Vertices and point labels (e.g. A, B, C, O, P, Q).
- Visible written measurements (e.g. "60°", "90°", "5 cm", "r = 7").
- Marked mathematical relationships (e.g. "AB = AC", "angle ABC = 90°", "PQ || RS", "AB ⊥ BC").
- Coordinate values if shown.
- Parallel / perpendicular indicators or tick marks.
- Construction marks / dashed lines.

Return strict JSON matching this schema:
{
  "shape_type": "<shape name>",
  "description": "<detailed factual description of the geometry visual>",
  "vertices": ["A", "B", "C"],
  "point_labels": ["A", "B", "C"],
  "side_labels": ["AB", "BC", "AC"],
  "angle_labels": ["angle A", "angle B"],
  "measurements": ["60°"],
  "relationships": ["AB = AC"],
  "coordinate_values": [],
  "parallel_perpendicular_indicators": [],
  "construction_marks": [],
  "confidence": 0.95
}
"""


def analyze_geometry_figure(
    crop_image: Image.Image,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a cropped geometry diagram image.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {
            "description": "Geometry diagram",
            "details": GeometryDetails(
                shape_type="geometry_figure", confidence=0.5
            ).model_dump(),
        }

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": GEOMETRY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this geometry diagram crop. Return JSON only.",
                        },
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
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        return parse_geometry_response(content)

    except Exception as e:
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": GEOMETRY_SYSTEM_PROMPT},
                    {"role": "user", "content": "Analyze this geometry diagram. Return JSON only."},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            return parse_geometry_response(content)
        except Exception:
            return {
                "description": "Geometry diagram figure",
                "details": GeometryDetails(shape_type="geometry", confidence=0.5).model_dump(),
            }



def parse_geometry_response(response_text: str) -> Dict[str, Any]:
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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}
        else:
            data = {}

    description = data.get("description", "Geometry diagram showing shape and labels.")
    shape_type = data.get("shape_type", "geometry")

    details = GeometryDetails(
        shape_type=shape_type,
        vertices=data.get("vertices", []),
        point_labels=data.get("point_labels", []),
        side_labels=data.get("side_labels", []),
        angle_labels=data.get("angle_labels", []),
        measurements=data.get("measurements", []),
        relationships=data.get("relationships", []),
        coordinate_values=data.get("coordinate_values", []),
        parallel_perpendicular_indicators=data.get(
            "parallel_perpendicular_indicators", []
        ),
        construction_marks=data.get("construction_marks", []),
        confidence=float(data.get("confidence", 0.9)),
    )

    return {
        "description": description,
        "details": details.model_dump(),
    }
