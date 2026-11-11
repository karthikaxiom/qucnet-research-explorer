"""Regression tests for newly added source-pinned visual-evidence integrity."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evidence.visual_enhancements import (
    ADDITIONAL_CHARTS, EVIDENCE_REGISTER, pinned_link, verify_visual,
)
from src.evidence.import_extra_public_figures import verify_png, verify_svg


class EvidenceEnhancementTests(unittest.TestCase):
    def test_archived_binary_metric_arithmetic(self):
        self.assertAlmostEqual((6 + 32) / 70, .5428571428571428)
        self.assertAlmostEqual(32 / 57, .5614035087719298)
        self.assertAlmostEqual(32 / 39, .8205128205128205)
        self.assertAlmostEqual(6 / 31, .1935483870967742)
        self.assertAlmostEqual(64 / 96, 2/3)

    def test_figure_whitelist_has_no_medical_image_previews(self):
        paths = [r[3] for r in ADDITIONAL_CHARTS]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(p.endswith(('.png', '.svg')) for p in paths))
        for path in paths:
            self.assertNotIn('roi_preview', path)
            self.assertNotIn('false_positive_gallery', path)
            self.assertNotIn('false_negative_gallery', path)
            self.assertNotIn('Archive_v1.1_outdated', path)
            self.assertNotIn('03_Dataset', path)

    def test_pinned_url_validation(self):
        sha = 'e848013b9ff943d816a8c90840255233a5ae7b8d'
        self.assertIn('/blob/'+sha+'/', pinned_link(sha, '07_Results/figure.png'))
        with self.assertRaises(ValueError):
            pinned_link('latest', 'README.md')
        with self.assertRaises(ValueError):
            pinned_link(sha, '../../private.jpg')

    def test_sha256_image_gate(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'assets' / 'figures'
            folder.mkdir(parents=True)
            asset = folder / 'example.png'
            asset.write_bytes(b'example file for hash verification')
            (folder / 'extra_figure_manifest.json').write_text(json.dumps({'figures': [{
                'file': asset.name,
                'sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
            }]}), encoding='utf-8')
            self.assertEqual(verify_visual(Path(d), asset.name, manifest_name='extra_figure_manifest.json')[0], True)
            asset.write_bytes(b'changed')
            self.assertEqual(verify_visual(Path(d), asset.name, manifest_name='extra_figure_manifest.json')[0], False)

    def test_svg_filters_active_elements(self):
        verify_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" /></svg>')
        for bad in [b'<svg><image href="x.png"/></svg>', b'<svg><script/></svg>',
                    b'<!DOCTYPE svg><svg/>', b'<svg onload="alert(1)"/>']:
            with self.assertRaises(ValueError):
                verify_svg(bad)

    def test_png_signature(self):
        with self.assertRaises(ValueError):
            verify_png(b'not a PNG file')

    def test_coverage_register_has_real_paths_and_protocols(self):
        self.assertGreaterEqual(len(EVIDENCE_REGISTER), 12)
        self.assertTrue(all(entry[4].startswith(('07_Results/', 'Research_Extension/')) for entry in EVIDENCE_REGISTER))
        self.assertTrue(any('BatchNorm' in entry[1] for entry in EVIDENCE_REGISTER))
        self.assertTrue(any('calibration' in entry[1].lower() for entry in EVIDENCE_REGISTER))


if __name__ == '__main__':
    unittest.main()
