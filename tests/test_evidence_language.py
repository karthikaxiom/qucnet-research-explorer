"""Regression checks for consistent research terminology and accuracy boundaries."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceLanguageTests(unittest.TestCase):
    def test_deprecated_interface_labels_are_absent(self):
        files = (
            'pages/3_Experimental_Results_Lab.py',
            'pages/4_Reliability_and_Accuracy_Gap.py',
            'pages/5_Presenter_Mode_and_Evidence.py',
            'src/evidence/visual_enhancements.py',
            'src/evidence/presenter_flow.py',
        )
        forbidden = (
            'Panel experiment', 'Panel question:', 'Panel demonstration notes',
            'Presenter Mode cannot display', 'A guided, source-linked',
            'When asked whether quantum won', 'Start with the research question.',
            'owner-supplied crop', 'for a panel member',
        )
        for relative in files:
            content = (ROOT / relative).read_text(encoding='utf-8-sig').lower()
            for phrase in forbidden:
                with self.subTest(file=relative, phrase=phrase):
                    self.assertNotIn(phrase.lower(), content)

    def test_proven_metrics_and_boundaries_remain(self):
        narrative = (ROOT / 'src/evidence/presenter_flow.py').read_text(encoding='utf-8-sig')
        for fact in ('54.29%', '56.14%', '5.63%', '4.55%', 'TI-RADS', '70-image',
                     '203', 'not pathology-confirmed', 'does not retrain'):
            self.assertIn(fact, narrative)
        self.assertIn('RESEARCH EVIDENCE SUMMARY', narrative)
        self.assertIn('RESEARCHER COMMENTARY', narrative)

    def test_app_labels_research_focused(self):
        results = (ROOT / 'pages/3_Experimental_Results_Lab.py').read_text(encoding='utf-8-sig')
        self.assertIn('Metric sensitivity analysis', results)
        reliability = (ROOT / 'pages/4_Reliability_and_Accuracy_Gap.py').read_text(encoding='utf-8-sig')
        self.assertIn('Evidence-supported conclusions', reliability)
        evidence = (ROOT / 'pages/5_Presenter_Mode_and_Evidence.py').read_text(encoding='utf-8-sig')
        self.assertIn('Methodological questions', evidence)
        self.assertIn('Researcher’s commentary', evidence)


if __name__ == '__main__':
    unittest.main()
