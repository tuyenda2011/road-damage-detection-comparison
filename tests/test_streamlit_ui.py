from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.demo.app_streamlit import hex_to_rgba, model_color
from src.demo.streamlit_styles import APP_CSS


ROOT = Path(__file__).resolve().parents[1]


class StreamlitUiTests(unittest.TestCase):
    def test_dashboard_renders_empty_state_without_exception(self) -> None:
        app = AppTest.from_file(ROOT / "src/demo/app_streamlit.py", default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Workbench", "Model benchmark", "Visual analytics", "Hướng dẫn"],
        )
        self.assertEqual(len(app.get("file_uploader")), 1)

    def test_video_mode_renders_without_exception(self) -> None:
        app = AppTest.from_file(ROOT / "src/demo/app_streamlit.py", default_timeout=30).run()
        source_selector = next(radio for radio in app.radio if "Video" in radio.options)
        source_selector.set_value("Video").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.get("file_uploader")), 1)

    def test_populated_metrics_and_charts_render_without_exception(self) -> None:
        fixture = ROOT / "tests/fixtures/streamlit_metrics_fixture.py"
        app = AppTest.from_file(fixture, default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.metric), 4)

    def test_chart_color_helpers(self) -> None:
        self.assertEqual(model_color("RT-DETR"), "#7C3AED")
        self.assertEqual(hex_to_rgba("#2DD4BF", 0.13), "rgba(45,212,191,0.13)")

    def test_light_theme_is_explicit(self) -> None:
        config = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
        normalized_config = config.lower()
        self.assertIn('base               = "light"', normalized_config)
        self.assertIn('backgroundcolor    = "#ffffff"', normalized_config)
        self.assertIn("--canvas: #ffffff", APP_CSS)
        self.assertNotIn("--canvas: #07111f", APP_CSS.lower())


if __name__ == "__main__":
    unittest.main()
