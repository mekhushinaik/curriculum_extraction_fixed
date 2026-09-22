# tools/dataset_prep/prompts/map_geometry_annotation.py
"""
Prompt for annotating a map answer key with per-item reference geometry.

One call receives the assessment's answer map stamped with its anchored
coordinate grid and the map-marking checklist, and returns each item's
position as machine-readable grid geometry: its class and the cell(s) it
occupies, ordered along the feature.  The output is persisted on the rubric
(``map_items[].ref_geometry``) alongside the grid it is valid in
(``map_reference_grid``), so grading verifies each student marking against
authoritative numbers instead of re-reading the answer map on every call —
and a deterministic audit can re-check every award.

Four geometry classes cover the map work the school boards set — points
(cities, dams, peaks, passes, ports, plants), lines in any orientation
(rivers, meridians, latitudes, mountain ranges, wind or current arrows),
stretches along the outline (coasts), and areas (soil / crop / rainfall /
population regions, plateaus, deserts).  Orientation is never classified:
the audit infers each line's discriminative axis from the cells themselves,
and an arrow's direction is judged as part of the marking's SYMBOL, not its
position.

Runs once per assessment at preparation time; the annotated geometry is
human-reviewable (and correctable) in the assessment JSON before seeding.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a precise cartographic annotator.  The attached image is an "
    "exam answer key: a map with every required feature marked correctly, "
    "stamped with a labelled coordinate grid (columns A, B, C... left to "
    "right; rows 1, 2, 3... top to bottom).  For each required item in the "
    "checklist, report exactly where the answer key marks it, as grid "
    "geometry:\n"
    "\n"
    "  * `kind` — the feature's geometry class:\n"
    "      - `point`: a single marked position — a city, dam, peak, pass, "
    "port, plant, or any one-spot symbol.\n"
    "      - `line`: an elongated feature in any orientation — a river, a "
    "meridian or latitude, a mountain range, a wind or current arrow.\n"
    "      - `stretch`: a run along the printed outline itself — a coastal "
    "stretch.\n"
    "      - `area`: a shaded or outlined region — a soil / crop / rainfall "
    "/ population zone, a plateau, a desert.\n"
    "  * `cells` — the grid cells the marking occupies, ordered along the "
    "feature: a point gives ONE cell; a line or stretch lists the cells "
    "from one end to the other (first entry = one endpoint, last entry = "
    "the other); an area lists the cells it covers.  Use the cells of the "
    "marking itself, never of its text label or pointer arrow.\n"
    "  * `anchor` — 8-20 words tying the position to printed features "
    "(coastline shapes, river courses, confluences, boundaries) so a human "
    "reviewer can confirm the cells.\n"
    "  * `confidence` — 0.0-1.0 for the cell reading.\n"
    "  * `location_tolerance` — how strictly a student's position is graded.  "
    "Default to `strict`: the feature has one correct place a marking must sit "
    "on — a point (city, dam, peak, pass, port, plant), a specific named river "
    "along its course, a meridian or line of latitude, a named hill or range, "
    "AND a named region at a fixed location (a plateau, a coastal stretch, a "
    "specific soil, crop or mineral belt) even when it is large, because it "
    "must be marked where that named feature actually lies.  Use `regional` "
    "ONLY for a feature with no single correct spot — one that is correct "
    "anywhere within a broad zone: a wind or ocean current drawn over a sea, "
    "or a phenomenon asked for by type rather than by name ('a sparsely "
    "populated area', 'an area of heavy rainfall') where any qualifying place "
    "is acceptable.\n"
    "\n"
    "Read positions from the marks on THIS image only.  Do not use outside "
    "geographic knowledge to relocate a feature: the answer key as printed "
    "is the authority, even if a mark surprises you.\n"
    "\n"
    "Return strict JSON only — no markdown fences, no commentary."
)


_RESPONSE_SCHEMA_BLOCK = """{
  "items": [
    {
      "key": "<checklist key, verbatim>",
      "kind": "point" | "line" | "stretch" | "area",
      "cells": ["<cell>", ...],
      "anchor": "<8-20 words tying the position to printed features>",
      "confidence": <float 0.0..1.0>,
      "location_tolerance": "strict" | "regional"
    }
  ]
}"""


def build_user_message(map_items: list[dict[str, Any]], n_cols: int, n_rows: int) -> str:
    """Return the text portion of the annotation user message.

    The caller attaches the grid-stamped answer map after this text.
    ``map_items`` is the pooled checklist (each entry's ``key`` / ``name`` /
    ``expected`` orient the annotator); the grid dimensions bound the valid
    cell vocabulary.
    """
    checklist = [{"key": item.get("key"), "name": item.get("name"), "expected": item.get("expected")} for item in map_items]
    last_col = chr(ord("A") + n_cols - 1)
    return (
        f"GRID: {n_cols} columns (A..{last_col}) x {n_rows} rows (1..{n_rows}).\n"
        f"\n"
        f"CHECKLIST ({len(checklist)} item(s)):\n"
        f"{json.dumps(checklist, indent=2, ensure_ascii=False)}\n"
        f"\n"
        f"Task: for every checklist item, locate its marking on the attached "
        f"answer map and report its `kind`, `cells`, `anchor`, `confidence`, "
        f"and `location_tolerance`.  Emit exactly one entry per item, using each `key` "
        f"verbatim.  Cells must lie within the grid bounds; keep a long "
        f"feature's list to the cells it actually passes through (endpoints "
        f"first and last).\n"
        f"\n"
        f"Return JSON matching this schema (and ONLY this — no markdown "
        f"fences):\n"
        f"{_RESPONSE_SCHEMA_BLOCK}\n"
    )


__all__ = ["SYSTEM_PROMPT", "build_user_message"]
