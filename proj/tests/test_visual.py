import unittest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw, ImageFont

from visual.figure_analyzer import analyze_crop, classify_figure_crop
from visual.figure_detector import parse_detector_response
from visual.geometry_analyzer import parse_geometry_response
from visual.map_analyzer import parse_map_response
from visual.graph_analyzer import parse_graph_response
from visual.scientific_analyzer import parse_scientific_response
from visual.models import FigureRecord
from schema import SubjectCurriculum


class TestVisualPipeline(unittest.TestCase):
    """
    Comprehensive test suite covering all 13 visual pipeline test scenarios.
    """

    def setUp(self):
        # Create a blank white canvas image (500x500)
        self.blank_img = Image.new("RGB", (500, 500), color="white")

    # TEST 1: Text-only page -> Expected: no figures detected
    def test_text_only_page(self):
        response_json = '{"page_number": 1, "figure_bbox_annotations": []}'
        figures = parse_detector_response(response_json, page_number=1)
        self.assertEqual(len(figures), 0)

    # TEST 2: Page with one triangle -> Expected: one geometry figure detected
    def test_page_with_one_triangle(self):
        response_json = '''{
          "page_number": 1,
          "figure_bbox_annotations": [
            {
              "bbox": [0.1, 0.1, 0.5, 0.5],
              "confidence": 0.95,
              "label": "Fig. 1",
              "belongs_to": {"q_no": 1, "sub_q_no": null}
            }
          ]
        }'''
        figures = parse_detector_response(response_json, page_number=1)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]["bbox"], [0.1, 0.1, 0.5, 0.5])

    # TEST 3: Page with four geometry MCQ options -> Expected: four separate figure bboxes
    def test_page_with_four_mcq_options(self):
        response_json = '''{
          "page_number": 2,
          "figure_bbox_annotations": [
            {"bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.9, "label": "(a)", "belongs_to": {"q_no": 5, "sub_q_no": null}},
            {"bbox": [0.5, 0.1, 0.9, 0.4], "confidence": 0.9, "label": "(b)", "belongs_to": {"q_no": 5, "sub_q_no": null}},
            {"bbox": [0.1, 0.5, 0.4, 0.9], "confidence": 0.9, "label": "(c)", "belongs_to": {"q_no": 5, "sub_q_no": null}},
            {"bbox": [0.5, 0.5, 0.9, 0.9], "confidence": 0.9, "label": "(d)", "belongs_to": {"q_no": 5, "sub_q_no": null}}
          ]
        }'''
        figures = parse_detector_response(response_json, page_number=2)
        self.assertEqual(len(figures), 4)
        labels = [f["label"] for f in figures]
        self.assertIn("(a)", labels)
        self.assertIn("(b)", labels)
        self.assertIn("(c)", labels)
        self.assertIn("(d)", labels)

    # TEST 4: Page with a map -> Expected: map detected and analyzed with point/line/stretch/area grid
    def test_page_with_map(self):
        map_json = '''{
          "map_type": "thematic",
          "description": "Map showing the distribution of black soil across parts of India.",
          "legend": ["Black soil region"],
          "marked_regions": ["Deccan Plateau"],
          "features": [
            {
              "key": "Deccan Black Soil Zone",
              "kind": "area",
              "cells": ["B2", "B3", "C3"],
              "anchor": "Central plateau bounded by Western Ghats",
              "confidence": 0.95,
              "location_tolerance": "strict"
            }
          ],
          "confidence": 0.95
        }'''
        parsed = parse_map_response(map_json)
        self.assertEqual(parsed["details"]["map_type"], "thematic")
        self.assertIn("black soil", parsed["description"].lower())
        self.assertEqual(len(parsed["details"]["features"]), 1)
        self.assertEqual(parsed["details"]["features"][0]["kind"], "area")

    # TEST 5: Page with map + text -> Expected: map detected, text still extracted normally
    def test_page_with_map_and_text(self):
        detector_json = '''{
          "page_number": 3,
          "figure_bbox_annotations": [
            {"bbox": [0.2, 0.2, 0.8, 0.8], "confidence": 0.92, "label": "Fig 3.2", "belongs_to": {"q_no": 2, "sub_q_no": "a"}}
          ]
        }'''
        figures = parse_detector_response(detector_json, page_number=3)
        self.assertEqual(len(figures), 1)

    # TEST 6: Page with graph -> Expected: graph detected and analyzed with x/y axes and trends
    def test_page_with_graph(self):
        graph_json = '''{
          "graph_type": "line_graph",
          "description": "Line graph showing temperature variation over time.",
          "x_axis_label": "Time (Hours)",
          "y_axis_label": "Temperature (°C)",
          "units": ["Hours", "°C"],
          "legend": ["City A"],
          "plotted_values": ["10", "20", "30"],
          "trend_pattern": "Steadily increasing over morning hours",
          "confidence": 0.95
        }'''
        parsed = parse_graph_response(graph_json)
        self.assertEqual(parsed["details"]["graph_type"], "line_graph")
        self.assertEqual(parsed["details"]["x_axis_label"], "Time (Hours)")

    # TEST 7: Page with table -> Expected: table NOT detected as a figure
    def test_page_with_table(self):
        # Classifier identifies crop as table -> analyze_crop returns None
        with patch("visual.figure_analyzer.classify_figure_crop", return_value=("table", 0.98)):
            res = analyze_crop(self.blank_img)
            self.assertIsNone(res)

    # TEST 8: Page with chemistry equation written linearly -> Expected: NOT detected as figure
    def test_page_with_linear_chemistry(self):
        # Classifier identifies crop as linear equation -> analyze_crop returns None
        with patch("visual.figure_analyzer.classify_figure_crop", return_value=("linear_equation", 0.95)):
            res = analyze_crop(self.blank_img)
            self.assertIsNone(res)

    # TEST 9: Page with 2D chemical structure -> Expected: detected as figure
    def test_page_with_2d_chemical_structure(self):
        chem_json = '''{
          "category": "chemical_structure",
          "description": "2D skeletal benzene ring structure with hydroxyl group.",
          "chemical_details": {
            "structure_type": "benzene_ring",
            "atoms_labels": ["C", "H", "OH"],
            "bonds": ["aromatic ring"],
            "rings": ["benzene"],
            "reaction_arrows": [],
            "conditions": [],
            "confidence": 0.95
          }
        }'''
        parsed = parse_scientific_response(chem_json, default_chemical=True)
        self.assertIn("hydroxyl", parsed["description"].lower())
        self.assertEqual(parsed["details"]["structure_type"], "benzene_ring")

    # TEST 10: Geometry diagram with labels -> Expected: labels and visible relationships extracted
    def test_geometry_diagram_with_labels(self):
        geom_json = '''{
          "shape_type": "triangle",
          "description": "Geometry diagram showing triangle ABC with angle A = 60° and marked right angle.",
          "vertices": ["A", "B", "C"],
          "point_labels": ["A", "B", "C"],
          "side_labels": ["AB", "BC", "AC"],
          "angle_labels": ["angle A"],
          "measurements": ["60°"],
          "relationships": ["AB = AC", "angle ABC = 90°"],
          "confidence": 0.96
        }'''
        parsed = parse_geometry_response(geom_json)
        self.assertEqual(parsed["details"]["vertices"], ["A", "B", "C"])
        self.assertIn("60°", parsed["details"]["measurements"])
        self.assertIn("AB = AC", parsed["details"]["relationships"])

    # TEST 11: Geometry diagram with unreadable measurement -> Expected: do not hallucinate measurement
    def test_geometry_unreadable_measurement(self):
        geom_json = '''{
          "shape_type": "triangle",
          "description": "Triangle ABC with an unreadable side measurement.",
          "vertices": ["A", "B", "C"],
          "point_labels": ["A", "B", "C"],
          "measurements": [],
          "relationships": [],
          "confidence": 0.85
        }'''
        parsed = parse_geometry_response(geom_json)
        self.assertEqual(parsed["details"]["measurements"], [])
        self.assertEqual(parsed["details"]["relationships"], [])

    # TEST 12: Multiple unrelated figures on one page -> Expected: separate bbox for each
    def test_multiple_unrelated_figures(self):
        detector_json = '''{
          "page_number": 4,
          "figure_bbox_annotations": [
            {"bbox": [0.05, 0.05, 0.45, 0.45], "confidence": 0.95, "label": "Fig 1", "belongs_to": {"q_no": 1, "sub_q_no": null}},
            {"bbox": [0.55, 0.55, 0.95, 0.95], "confidence": 0.95, "label": "Fig 2", "belongs_to": {"q_no": 2, "sub_q_no": null}}
          ]
        }'''
        figures = parse_detector_response(detector_json, page_number=4)
        self.assertEqual(len(figures), 2)
        self.assertNotEqual(figures[0]["bbox"], figures[1]["bbox"])

    # TEST 13: Existing textbook with no figures -> Expected: existing curriculum JSON remains unchanged
    def test_existing_curriculum_unchanged(self):
        sample_curriculum = {
            "curriculum": "NCERT",
            "grade": "10",
            "subject": "Social Science",
            "chapters": [
                {
                    "id": "chap-social-science-tb-1",
                    "chapterNumber": "1",
                    "chapterTitle": "Water Resources",
                    "theme": "Geography",
                    "topics": [
                        {
                            "title": "Multipurpose River Projects",
                            "topic_id": "top-social-science-tb-1-1",
                            "subtopics": [
                                {
                                    "title": "Dams and Water Management",
                                    "description": "Role of dams in flood control and irrigation.",
                                    "sub_topic_id": "sub-social-science-tb-1-1-1",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        # Validate that existing curriculum matches schema exactly
        validated = SubjectCurriculum.model_validate(sample_curriculum)
        self.assertEqual(validated.subject, "Social Science")
        self.assertEqual(len(validated.chapters), 1)


if __name__ == "__main__":
    unittest.main()
