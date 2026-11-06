"""Structural checks for the unified Streamlit multipage navigation."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / 'app.py'
PAGE_FILES = [
    'pages/0_Research_Overview.py',
    'pages/1_Interactive_Architecture_Lab.py',
    'pages/2_Dataset_and_Integrity.py',
    'pages/3_Experimental_Results_Lab.py',
    'pages/4_Reliability_and_Accuracy_Gap.py',
    'pages/5_Presenter_Mode_and_Evidence.py',
]

class NavigationContractTests(unittest.TestCase):
    def test_all_six_pages_exist(self):
        for relative in PAGE_FILES:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_router_and_pages_parse(self):
        for relative in ['app.py'] + PAGE_FILES:
            with self.subTest(path=relative):
                ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))

    def test_router_registers_all_six_pages(self):
        source = ROUTER.read_text(encoding='utf-8-sig')
        self.assertIn('st.navigation(', source)
        for relative in PAGE_FILES:
            self.assertIn(relative, source)

    def test_overview_and_entrypoint_are_separate(self):
        source = ROUTER.read_text(encoding='utf-8-sig')
        self.assertNotIn('st.title("QuCNet Research Explorer")', source)
        overview = (ROOT / PAGE_FILES[0]).read_text(encoding='utf-8-sig')
        self.assertIn('st.title("QuCNet Research Explorer")', overview)
        self.assertNotIn('Additional modules will be added', overview)

    def test_source_repo_not_modified_by_router(self):
        source = ROUTER.read_text(encoding='utf-8-sig')
        for forbidden in ('git push', 'model.fit(', 'pennylane', 'tensorflow', 'open("D:\\QCNN'):
            self.assertNotIn(forbidden, source)

if __name__ == '__main__':
    unittest.main()
