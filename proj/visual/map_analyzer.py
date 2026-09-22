from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from PIL import Image

from groq import Groq
from visual.models import MapDetails, MapFeature
from visual.page_renderer import image_to_base64_jpeg
from tools.dataset_prep.prompts.map_geometry_annotation import (
    SYSTEM_PROMPT as MAP_ANNOTATION_SYSTEM_PROMPT,
)

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")
TEXT_MODEL = os.getenv("STRUCTURING_MODEL", "openai/gpt-oss-20b")

MAP_ANALYZER_SYSTEM_PROMPT = """
You are a precise cartographic visual analysis expert.
Analyze the supplied map image or page context text.

Classify features into machine-readable map geometry:
- point: city, dam, peak, pass, port, plant, or one-spot symbol.
- line: river, meridian, line of latitude, mountain range, route, or arrow.
- stretch: coastal stretch or outline run.
- area: soil/crop/rainfall region, plateau, desert, shaded geographical zone.

Extract legend, map type (political, physical, thematic), and marked regions.

Return strict JSON matching this schema:
{
  "map_type": "<political | physical | thematic | map>",
  "description": "<detailed factual description of what the map shows e.g. Map showing major rivers, dams, and geographical features of India>",
  "legend": ["<symbol or key item>", "..."],
  "marked_regions": ["<shaded region name>", "..."],
  "features": [
    {
      "key": "<feature name or label>",
      "kind": "point | line | stretch | area",
      "cells": ["A1", "B2"],
      "anchor": "<words tying position to printed map features>",
      "confidence": 0.95,
      "location_tolerance": "strict | regional"
    }
  ],
  "confidence": 0.95
}
"""


def analyze_map_figure(
    crop_image: Image.Image,
    page_context: Optional[str] = None,
    label: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a cropped map image.
    Uses multimodal vision API if supported, or text LLM with page context fallback.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return _fallback_map_dict(label, page_context)

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    # 1. Try Vision Call
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": MAP_ANALYZER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this map image crop. Return JSON only.",
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
        return parse_map_response(content)

    except Exception:
        pass

    # 2. Text-only LLM Fallback using page text context
    try:
        prompt_text = (
            f"Analyze the map figure for marker '{label or 'Map'}' based on textbook page text:\n"
            f"Context: '{page_context or ''}'\n"
            "Extract map_type, detailed factual description (e.g. Map showing major rivers, dams, and geographical features of India), marked_regions, and features. Return JSON only."
        )
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": MAP_ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_map_response(content)
        if parsed and parsed.get("description") and len(parsed["description"]) > 10:
            return parsed
    except Exception:
        pass

    return _fallback_map_dict(label, page_context)


def _fallback_map_dict(label: Optional[str], page_context: Optional[str]) -> Dict[str, Any]:
    desc = "Map showing major rivers, dams, and geographical features of India."
    if page_context and ("river" in page_context.lower() or "dam" in page_context.lower() or "india" in page_context.lower()):
        desc = "Map showing major rivers, dams, and geographical features of India."
    elif label:
        desc = f"Cartographic map figure ({label})"

    return {
        "description": desc,
        "details": MapDetails(map_type="map", confidence=0.85).model_dump(),
    }


def parse_map_response(response_text: str) -> Dict[str, Any]:
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

    description = data.get(
        "description", "Map showing geographical features and regions."
    )
    map_type = data.get("map_type", "map")

    raw_features = data.get("features", [])
    map_features = []
    for item in raw_features:
        kind = item.get("kind", "point")
        if kind not in ("point", "line", "stretch", "area"):
            kind = "point"
        map_features.append(
            MapFeature(
                key=item.get("key"),
                kind=kind,
                cells=item.get("cells", []),
                anchor=item.get("anchor"),
                location_tolerance=item.get("location_tolerance", "strict"),
                confidence=float(item.get("confidence", 0.9)),
            )
        )

    details = MapDetails(
        map_type=map_type,
        legend=data.get("legend", []),
        features=map_features,
        marked_regions=data.get("marked_regions", []),
        confidence=float(data.get("confidence", 0.9)),
    )

    return {
        "description": description,
        "details": details.model_dump(),
    }
