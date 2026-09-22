from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from PIL import Image

from groq import Groq
from visual.models import ChemicalStructureDetails, ScientificDetails
from visual.page_renderer import image_to_base64_jpeg

VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-90b-vision-preview")

SCIENTIFIC_SYSTEM_PROMPT = """
You are a scientific visual analysis expert (biology, physics, chemistry, apparatus, 2D structures).
Analyze ONLY what is visually present in the supplied scientific diagram image crop.

For general scientific diagrams (biology, physics, apparatus, circuits, flowcharts):
- Extract diagram_type, labeled components, arrows, and component relationships.

For 2D chemical structures (benzene ring, displayed/skeletal formula, reaction scheme):
- Extract structure_type, atoms/labels, bonds, rings, reaction_arrows, and conditions written above/below arrows.

Return strict JSON matching this schema:
{
  "category": "<scientific_diagram | chemical_structure>",
  "description": "<detailed factual description of the scientific visual content>",
  "scientific_details": {
    "diagram_type": "<biology | physics | apparatus | circuit | flowchart>",
    "components": ["<label 1>", "..."],
    "arrows": ["<arrow 1>", "..."],
    "relationships": ["<relation 1>", "..."],
    "confidence": 0.95
  },
  "chemical_details": {
    "structure_type": "<skeletal | displayed | benzene_ring | reaction_scheme>",
    "atoms_labels": ["C", "H", "O"],
    "bonds": ["single", "double", "aromatic"],
    "rings": ["benzene"],
    "reaction_arrows": ["-->"],
    "conditions": ["heat", "catalyst"],
    "confidence": 0.95
  }
}
"""


def analyze_scientific_figure(
    crop_image: Image.Image,
    is_chemical: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a cropped scientific diagram or 2D chemical structure image.
    """
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        if is_chemical:
            return {
                "description": "2D Chemical structure diagram",
                "details": ChemicalStructureDetails(
                    structure_type="chemical_structure", confidence=0.5
                ).model_dump(),
            }
        else:
            return {
                "description": "Scientific diagram",
                "details": ScientificDetails(
                    diagram_type="scientific_diagram", confidence=0.5
                ).model_dump(),
            }

    client = Groq(api_key=groq_api_key)
    base64_img = image_to_base64_jpeg(crop_image)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": SCIENTIFIC_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this scientific/chemical image crop. Return JSON only.",
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
        return parse_scientific_response(content, is_chemical)

    except Exception as e:
        print(f"Scientific visual analysis failed: {e}")
        if is_chemical:
            return {
                "description": "2D Chemical structure diagram",
                "details": ChemicalStructureDetails(
                    structure_type="chemical_structure", confidence=0.5
                ).model_dump(),
            }
        else:
            return {
                "description": "Scientific diagram",
                "details": ScientificDetails(
                    diagram_type="scientific_diagram", confidence=0.5
                ).model_dump(),
            }


def parse_scientific_response(
    response_text: str, default_chemical: bool = False
) -> Dict[str, Any]:
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

    description = data.get("description", "Scientific diagram visual.")
    category = data.get("category", "chemical_structure" if default_chemical else "scientific_diagram")

    if category == "chemical_structure" or default_chemical:
        chem_raw = data.get("chemical_details", {})
        details = ChemicalStructureDetails(
            structure_type=chem_raw.get("structure_type", "chemical_structure"),
            atoms_labels=chem_raw.get("atoms_labels", []),
            bonds=chem_raw.get("bonds", []),
            rings=chem_raw.get("rings", []),
            reaction_arrows=chem_raw.get("reaction_arrows", []),
            conditions=chem_raw.get("conditions", []),
            confidence=float(chem_raw.get("confidence", 0.9)),
        )
    else:
        sci_raw = data.get("scientific_details", {})
        details = ScientificDetails(
            diagram_type=sci_raw.get("diagram_type", "scientific_diagram"),
            components=sci_raw.get("components", []),
            arrows=sci_raw.get("arrows", []),
            relationships=sci_raw.get("relationships", []),
            confidence=float(sci_raw.get("confidence", 0.9)),
        )

    return {
        "description": description,
        "details": details.model_dump(),
    }
