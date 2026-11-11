"""Implementation and data-privacy invariants for input pathway diagrams."""
import re
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from src.evidence.input_representations import (
    ARCHITECTURE_LABELS, ROUTES, SOURCE_COMMIT, diagram_svg, original_link,
)


class InputRepresentationTests(unittest.TestCase):
    def test_five_distinct_archived_pathways(self):
        self.assertEqual(len(ROUTES), 5)
        self.assertEqual(len({route.heading for route in ROUTES.values()}), 5)
        self.assertEqual(set(ARCHITECTURE_LABELS.values()), set(ROUTES))
        for route in ROUTES.values():
            self.assertEqual(len(route.steps), 4)
            self.assertTrue(route.context)
            self.assertTrue(route.result)
            self.assertTrue(all(s.title and s.shape and s.detail for s in route.steps))
            self.assertTrue(route.source_paths)

    def test_historical_binary_is_not_six_class_or_roi(self):
        binary = ROUTES["Phase 6 · Binary quantum QuCNet"]
        self.assertIn("128 × 128", binary.steps[0].shape)
        self.assertIn("64 × 64 × 4", binary.steps[2].shape)
        self.assertIn("54.29%", binary.result)
        roi = ROUTES["Phase 14 · Strict ROI CNN"]
        self.assertIn("4/71", roi.result)
        self.assertIn("2/44", roi.result)
        extension = ROUTES["Extension · Fixed 64-D representations"]
        self.assertIn("311 training ROIs", extension.context)
        self.assertIn("not the original", extension.context)

    def test_diagram_is_xml_schematic_with_no_external_media(self):
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for key in ROUTES:
            source = diagram_svg(key)
            root = ET.fromstring(source)
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(len(root.findall("svg:title", namespace)), 1)
            self.assertEqual(len(root.findall("svg:desc", namespace)), 1)
            self.assertEqual(len(root.findall("svg:rect", namespace)), 9)
            self.assertNotIn("<image", source.lower())
            self.assertNotIn("<script", source.lower())
            self.assertNotIn("data:image", source.lower())
            self.assertFalse(re.search(r'\b(?:href|src)\s*=', source))

    def test_no_nested_expander_in_input_renderer(self):
        import inspect
        from src.evidence.input_representations import render_route
        self.assertNotIn("st.expander(", inspect.getsource(render_route))

    def test_source_links_must_be_pinned_and_relative(self):
        for route in ROUTES.values():
            for path in route.source_paths:
                self.assertIn(SOURCE_COMMIT, original_link(SOURCE_COMMIT, path))
        with self.assertRaises(ValueError):
            original_link("a" * 40, "README.md")
        with self.assertRaises(ValueError):
            original_link(SOURCE_COMMIT, "../private.csv")

    def test_no_deprecated_width_arguments_in_public_pages(self):
        root = Path(__file__).resolve().parents[1]
        for p in (root / "pages").glob("*.py"):
            self.assertNotIn("use_container_width=", p.read_text(encoding="utf-8-sig"), str(p))
        enhancements = root / "src/evidence/visual_enhancements.py"
        self.assertNotIn("use_container_width=", enhancements.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
