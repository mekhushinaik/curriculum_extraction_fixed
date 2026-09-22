from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

FigureType = Literal[
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
    "other",
]


class GeometryDetails(BaseModel):
    shape_type: Optional[str] = Field(
        default=None,
        description="Type of geometric shape e.g. triangle, circle, square, parallelogram, coordinate_plane, etc.",
    )
    vertices: List[str] = Field(
        default_factory=list,
        description="Vertices or point labels e.g. ['A', 'B', 'C']",
    )
    point_labels: List[str] = Field(
        default_factory=list, description="Labels for points in the diagram"
    )
    side_labels: List[str] = Field(
        default_factory=list, description="Labels or names for sides/edges"
    )
    angle_labels: List[str] = Field(
        default_factory=list, description="Labels or names for angles"
    )
    measurements: List[str] = Field(
        default_factory=list,
        description="Visibly written measurements e.g. ['60°', '5 cm']",
    )
    relationships: List[str] = Field(
        default_factory=list,
        description="Mathematical relationships explicitly marked e.g. ['AB = AC', 'angle ABC = 90°']",
    )
    coordinate_values: List[str] = Field(
        default_factory=list,
        description="Coordinate values e.g. ['(0, 0)', '(3, 4)']",
    )
    parallel_perpendicular_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators for parallel or perpendicular lines",
    )
    construction_marks: List[str] = Field(
        default_factory=list, description="Construction lines or marks"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MapFeature(BaseModel):
    key: Optional[str] = Field(default=None, description="Feature checklist key or name")
    kind: Literal["point", "line", "stretch", "area"] = Field(
        default="point", description="Feature geometry class"
    )
    cells: List[str] = Field(
        default_factory=list, description="Grid cells occupied by the feature e.g. ['B2', 'B3']"
    )
    anchor: Optional[str] = Field(
        default=None, description="Tying position to printed geographic features"
    )
    location_tolerance: Literal["strict", "regional"] = Field(default="strict")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MapDetails(BaseModel):
    map_type: Optional[str] = Field(
        default=None, description="Type of map e.g. political, physical, thematic"
    )
    legend: List[str] = Field(
        default_factory=list, description="Legend/key symbols or entries"
    )
    features: List[MapFeature] = Field(
        default_factory=list, description="Extracted point, line, stretch, area features"
    )
    marked_regions: List[str] = Field(
        default_factory=list, description="Named or shaded geographical regions"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphDetails(BaseModel):
    graph_type: Optional[str] = Field(
        default=None, description="e.g. line_graph, bar_chart, pie_chart, histogram, scatter_plot"
    )
    x_axis_label: Optional[str] = Field(default=None, description="Label on X axis")
    y_axis_label: Optional[str] = Field(default=None, description="Label on Y axis")
    units: List[str] = Field(default_factory=list, description="Units of measurement on axes")
    legend: List[str] = Field(default_factory=list, description="Graph legend items")
    plotted_values: List[str] = Field(
        default_factory=list, description="Readable data points or values"
    )
    trend_pattern: Optional[str] = Field(
        default=None, description="Overall trend or pattern shown"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ScientificDetails(BaseModel):
    diagram_type: Optional[str] = Field(
        default=None, description="e.g. biology_diagram, physics_apparatus, circuit, flowchart"
    )
    components: List[str] = Field(
        default_factory=list, description="Labeled components or apparatus parts"
    )
    arrows: List[str] = Field(
        default_factory=list, description="Flow or reaction arrows"
    )
    relationships: List[str] = Field(
        default_factory=list, description="Component interactions or flow steps"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChemicalStructureDetails(BaseModel):
    structure_type: Optional[str] = Field(
        default=None, description="e.g. skeletal, displayed, benzene_ring, reaction_scheme"
    )
    atoms_labels: List[str] = Field(
        default_factory=list, description="Atoms or functional groups shown"
    )
    bonds: List[str] = Field(
        default_factory=list, description="Types of bonds e.g. single, double, aromatic ring"
    )
    rings: List[str] = Field(
        default_factory=list, description="Ring structures present e.g. benzene ring"
    )
    reaction_arrows: List[str] = Field(
        default_factory=list, description="Reaction arrows connecting structures"
    )
    conditions: List[str] = Field(
        default_factory=list, description="Reaction conditions written above/below arrows"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FigureRecord(BaseModel):
    figure_id: str = Field(..., description="Unique figure ID e.g. fig-p5-01")
    page_number: int = Field(..., ge=1, description="1-indexed PDF page number")
    figure_type: FigureType = Field(..., description="Classified figure category")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalized bounding box [x0, y0, x1, y1] with values in [0.0, 1.0]",
    )
    description: str = Field(
        ..., min_length=1, description="Rich detailed description of visual content"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    labels: List[str] = Field(
        default_factory=list, description="Printed markers or labels associated with figure"
    )
    details: Optional[
        Union[
            GeometryDetails,
            MapDetails,
            GraphDetails,
            ScientificDetails,
            ChemicalStructureDetails,
            Dict[str, Any],
        ]
    ] = Field(default=None, description="Domain-specific structured analysis")
    related_text: Optional[str] = Field(
        default=None, description="Nearby text or stem heading linked to this figure"
    )
    related_chapter: Optional[str] = Field(
        default=None, description="ID or title of related chapter"
    )
    related_topic: Optional[str] = Field(
        default=None, description="ID or title of related topic"
    )
    related_subtopic: Optional[str] = Field(
        default=None, description="ID or title of related subtopic"
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must contain exactly 4 numbers [x0, y0, x1, y1]")
        x0, y0, x1, y1 = v
        if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
            # Clamp to [0, 1] if slightly out of bounds
            x0 = max(0.0, min(1.0, x0))
            y0 = max(0.0, min(1.0, y0))
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
        if x0 >= x1:
            x1 = min(1.0, x0 + 0.05)
        if y0 >= y1:
            y1 = min(1.0, y0 + 0.05)
        return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


class PageVisualAnalysis(BaseModel):
    page_number: int = Field(..., ge=1)
    figures: List[FigureRecord] = Field(default_factory=list)


class VisualAnalysisOutput(BaseModel):
    pdf_path: str
    subject: str
    grade: str
    pages: List[PageVisualAnalysis] = Field(default_factory=list)
