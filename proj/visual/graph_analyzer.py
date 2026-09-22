from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from PIL import Image

from groq import Groq
from visual.models import GraphDetails
from visual.page_renderer import image_to_base64_jpeg

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")
TEXT_MODEL = os.getenv("STRUCTURING_MODEL", "openai/gpt-oss-20b")

GRAPH_SYSTEM_PROMPT = """
You are a data visualization visual analysis expert.
Analyze the supplied graph or chart image or text context.

Identify:
- Graph type (line_graph, bar_chart, pie_chart, histogram, scatter_plot).
- X-axis label and Y-axis label.
- Units of measurement visible on axes or legend.
- Legend entries.
- Plotted values if clearly readable.
- Trend or pattern shown.

Return strict JSON matching this schema:
{
  "graph_type": "<line_graph | bar_chart | pie_chart | histogram | scatter_plot>",
  "description": "<detailed description e.g. Line graph showing temperature variation over time>",
  "x_axis_label": "<x axis label>",
  "y_axis_label": "<y axis label>",
  "units": ["<unit 1>", "..."],
  "legend": ["<legend 1>", "..."],
  "plotted_values": ["<value 1>", "..."],
  "trend_pattern": "<summary of visible trend>",
  "confidence": 0.95
}
"""


def analyze_graph_figure(
    crop_image: Image.Image,
    page_context: Optional[str] = None,
    label: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a cropped graph or chart image.
    Uses multimodal vision API if supported, or text LLM with page context fallback.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return _fallback_graph_dict(label, page_context)

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    # 1. Try Vision Call
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this graph/chart image crop. Return JSON only.",
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
        return parse_graph_response(content)

    except Exception:
        pass

    # 2. Text-only LLM Fallback using page text context
    try:
        prompt_text = (
            f"Analyze the graph/chart figure for marker '{label or 'Figure'}' based on page text context:\n"
            f"Context: '{page_context or ''}'\n"
            "Identify graph_type (line_graph, bar_chart, pie_chart, histogram, scatter_plot), description, x_axis_label, y_axis_label, and trend_pattern. Return JSON only."
        )
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_graph_response(content)
        if parsed and parsed.get("description") and "Graph" in parsed["description"]:
            return parsed
    except Exception:
        pass

    return _fallback_graph_dict(label, page_context)


def _fallback_graph_dict(label: Optional[str], page_context: Optional[str]) -> Dict[str, Any]:
    desc = f"Graph figure ({label})" if label else "Graph or data chart"
    if page_context:
        ctx_sub = page_context[:100].strip()
        desc = f"Graph figure for {label or 'data'}: {ctx_sub}"

    return {
        "description": desc,
        "details": GraphDetails(
            graph_type="graph",
            confidence=0.8,
            trend_pattern="Data trends depicted in chart",
        ).model_dump(),
    }


def parse_graph_response(response_text: str) -> Dict[str, Any]:
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
        "description", "Graph displaying data and relationships."
    )
    graph_type = data.get("graph_type", "graph")

    details = GraphDetails(
        graph_type=graph_type,
        x_axis_label=data.get("x_axis_label"),
        y_axis_label=data.get("y_axis_label"),
        units=data.get("units", []),
        legend=data.get("legend", []),
        plotted_values=data.get("plotted_values", []),
        trend_pattern=data.get("trend_pattern"),
        confidence=float(data.get("confidence", 0.9)),
    )

    return {
        "description": description,
        "details": details.model_dump(),
    }
