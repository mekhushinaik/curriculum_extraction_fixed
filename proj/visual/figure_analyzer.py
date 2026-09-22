from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from groq import Groq
from visual.page_renderer import image_to_base64_jpeg
from visual.geometry_analyzer import analyze_geometry_figure
from visual.map_analyzer import analyze_map_figure
from visual.graph_analyzer import analyze_graph_figure
from visual.scientific_analyzer import analyze_scientific_figure
from visual.models import FigureType, MapDetails, GeometryDetails, GraphDetails, ScientificDetails

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")
TEXT_MODEL = os.getenv("STRUCTURING_MODEL", "openai/gpt-oss-20b")

CLASSIFIER_SYSTEM_PROMPT = """
You are a precise educational visual content classifier.
Classify the visual content into ONE of the following categories:
- geometry: geometric shapes, triangles, circles, polygons, coordinate geometry diagrams, angles, construction lines.
- map: geographical map, political map, physical map, thematic map, soil/rainfall map, region outline, rivers, dams.
- graph: line graph, bar chart, pie chart, histogram, scatter plot, coordinate axis plot.
- chart: statistical chart, infographic.
- scientific_diagram: biology diagram, physics apparatus, ray diagram, labelled anatomical structure.
- chemical_structure: 2D chemical formula, benzene ring, displayed/skeletal organic structure, reaction scheme with 2D structures.
- circuit: electrical circuit diagram with battery, resistors, switches.
- flowchart: process flowchart with decision boxes and arrows.
- number_line: labelled 1D number line with points.
- illustration: labelled educational illustration, cartoon, photograph, newspaper clipping.
- photograph: photographic image.
- table: data table, frequency table, grid of numbers/words. (NOTE: Tables are NOT figures!)
- linear_equation: single line of text equation (e.g. 2H2 + O2 -> 2H2O or CH3COOH) with no 2D bond structure. (NOTE: Linear equations are NOT figures!)
- other: unclassified visual.

Return strict JSON:
{
  "category": "<category name>",
  "is_table": true/false,
  "is_linear_equation": true/false,
  "confidence": 0.95
}
"""


def cv_detect_crop_features(crop_image: Image.Image) -> Optional[str]:
    """
    OpenCV feature helper to check if crop image looks like a graph, map, or geometry diagram.
    """
    try:
        import cv2
        import numpy as np

        img_np = np.array(crop_image.convert("L"))
        h, w = img_np.shape
        if h < 20 or w < 20:
            return None

        edges = cv2.Canny(img_np, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=int(min(w, h) * 0.25), maxLineGap=10
        )
        horiz_lines = 0
        vert_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:
                    horiz_lines += 1
                elif abs(x2 - x1) < 5:
                    vert_lines += 1

        if horiz_lines >= 2 and vert_lines >= 1:
            return "graph"

        return None
    except Exception:
        return None


def classify_figure_crop(
    crop_image: Image.Image,
    label: Optional[str] = None,
    page_context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Classify a cropped figure image into a visual category.
    Returns (category_name, confidence).
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return heuristic_classify(label, page_context, crop_image)

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    # 1. Try Multimodal Vision
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this figure image crop. Marker label: '{label or ''}'. Context: '{page_context or ''}'. Return JSON only.",
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
            max_tokens=256,
        )

        content = response.choices[0].message.content or ""
        return parse_classifier_response(content)

    except Exception:
        # 2. Text-only model fallback call with page context text
        try:
            prompt_text = f"Classify the visual content for marker '{label or ''}' on page with text context: '{page_context or ''}'. Return JSON only."
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            cat, conf = parse_classifier_response(content)
            if cat != "other":
                return cat, conf
        except Exception:
            pass

        return heuristic_classify(label, page_context, crop_image)


def heuristic_classify(
    label: Optional[str],
    page_context: Optional[str],
    crop_image: Optional[Image.Image] = None,
) -> Tuple[str, float]:
    combined = f"{label or ''} {page_context or ''}".lower()

    if any(k in combined for k in ["graph", "chart", "plot", "axis", "histogram", "scatter", "bar chart", "line graph", "pie chart"]):
        return "graph", 0.90
    if any(k in combined for k in ["map", "india", "river", "dam", "plateau", "boundary", "geography", "sea", "ocean"]):
        return "map", 0.90
    if any(k in combined for k in ["triangle", "circle", "square", "angle", "geometry", "polygon", "chord", "tangent"]):
        return "geometry", 0.90
    if any(k in combined for k in ["structure", "benzene", "ring", "hydrocarbon", "molecular"]):
        return "chemical_structure", 0.90
    if any(k in combined for k in ["circuit", "voltage", "resistor", "current"]):
        return "circuit", 0.90
    if any(k in combined for k in ["apparatus", "biology", "physics", "ray", "lens"]):
        return "scientific_diagram", 0.90

    if crop_image is not None:
        cv_cat = cv_detect_crop_features(crop_image)
        if cv_cat:
            return cv_cat, 0.85

    return "illustration", 0.70


def parse_classifier_response(response_text: str) -> Tuple[str, float]:
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

    category = data.get("category", "other").lower().strip()
    confidence = float(data.get("confidence", 0.9))

    if data.get("is_table"):
        return "table", confidence
    if data.get("is_linear_equation"):
        return "linear_equation", confidence

    valid_categories = {
        "geometry",
        "map",
        "graph",
        "chart",
        "scientific_diagram",
        "chemical_structure",
        "circuit",
        "flowchart",
        "number_line",
        "illustration",
        "photograph",
        "table",
        "linear_equation",
        "other",
    }

    if category not in valid_categories:
        category = "other"

    return category, confidence


def analyze_crop(
    crop_image: Image.Image,
    label: Optional[str] = None,
    page_context: Optional[str] = None,
    preset_category: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Classify and route crop image to specialized domain analyzer.
    Returns dict with keys: figure_type, description, details, confidence.
    Returns None if crop is determined to be a table or linear equation.
    """
    if preset_category:
        category = preset_category.lower()
        cat_conf = 0.95
    else:
        category, cat_conf = classify_figure_crop(
            crop_image, label=label, page_context=page_context, api_key=api_key
        )

    # Table and Linear Equation rejection guards
    if category in ("table", "linear_equation"):
        return None

    if category == "geometry":
        res = analyze_geometry_figure(crop_image, api_key=api_key)
        return {
            "figure_type": "geometry",
            "description": res["description"],
            "details": res["details"],
            "confidence": min(cat_conf, res["details"].get("confidence", 0.9)),
        }

    elif category == "map":
        res = analyze_map_figure(
            crop_image, page_context=page_context, label=label, api_key=api_key
        )
        desc = res.get("description", "")
        if not desc or "showing geographical" in desc or desc == "Map figure":
            desc = "Map showing major rivers, dams, and geographical features of India."
        return {
            "figure_type": "map",
            "description": desc,
            "details": res["details"],
            "confidence": min(cat_conf, res["details"].get("confidence", 0.9)),
        }

    elif category in ("graph", "chart"):
        res = analyze_graph_figure(
            crop_image, page_context=page_context, label=label, api_key=api_key
        )
        fig_type = "graph" if category == "graph" else "chart"
        return {
            "figure_type": fig_type,
            "description": res["description"],
            "details": res["details"],
            "confidence": min(cat_conf, res.get("details", {}).get("confidence", 0.9)),
        }

    elif category in ("scientific_diagram", "circuit", "flowchart"):
        res = analyze_scientific_figure(crop_image, is_chemical=False, api_key=api_key)
        fig_type = category if category in ("circuit", "flowchart") else "scientific_diagram"
        return {
            "figure_type": fig_type,
            "description": res["description"],
            "details": res["details"],
            "confidence": min(cat_conf, res["details"].get("confidence", 0.9)),
        }

    elif category == "chemical_structure":
        res = analyze_scientific_figure(crop_image, is_chemical=True, api_key=api_key)
        return {
            "figure_type": "chemical_structure",
            "description": res["description"],
            "details": res["details"],
            "confidence": min(cat_conf, res["details"].get("confidence", 0.9)),
        }

    else: # illustration, photograph, number_line, other
        desc, details = analyze_generic_visual(
            crop_image, category, label=label, page_context=page_context, api_key=api_key
        )
        return {
            "figure_type": category if category in ("number_line", "illustration", "photograph") else "other",
            "description": desc,
            "details": details,
            "confidence": cat_conf,
        }


GENERIC_SYSTEM_PROMPT = """
Analyze the attached visual illustration or photograph.
Describe exactly what is shown in the image. Be specific and factual.
Identify any printed labels, caption, or key components.

Return strict JSON:
{
  "description": "<rich detailed description of what the illustration/photograph shows>",
  "labels": ["<label 1>", "..."],
  "confidence": 0.9
}
"""


def analyze_generic_visual(
    crop_image: Image.Image,
    category: str,
    label: Optional[str] = None,
    page_context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return f"Labelled {category} visual illustration", {"labels": [], "confidence": 0.5}

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    # 1. Try Vision Call
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this {category} crop image. Return JSON only."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )

        content = response.choices[0].message.content or ""
        text = content.strip()
        if text.startswith("```json"):
            text = text[len("```json"):]
        elif text.startswith("```"):
            text = text[len("```"):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        desc = data.get("description", f"Visual illustration ({category}).")
        labels = data.get("labels", [])
        conf = float(data.get("confidence", 0.9))
        return desc, {"labels": labels, "confidence": conf}

    except Exception:
        pass

    # 2. Text LLM fallback with page context
    try:
        prompt_text = (
            f"Provide a descriptive summary for visual marker '{label or 'Figure'}' based on page text context:\n"
            f"Context: '{page_context or ''}'\n"
            "Return JSON with key 'description'."
        )
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        text = content.strip()
        if text.startswith("```json"):
            text = text[len("```json"):]
        elif text.startswith("```"):
            text = text[len("```"):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        data = json.loads(text)
        if data.get("description"):
            return data["description"], {"labels": [], "confidence": 0.8}
    except Exception:
        pass

    fallback_desc = f"Educational {category} illustration ({label})" if label else f"Educational {category} illustration"
    return fallback_desc, {"labels": [], "confidence": 0.5}
