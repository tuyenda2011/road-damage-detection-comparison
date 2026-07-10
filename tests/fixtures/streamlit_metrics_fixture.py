"""AppTest fixture that renders populated benchmark views."""
from pathlib import Path

import pandas as pd

import src.demo.app_streamlit as dashboard


def fixture_metrics():
    return (
        pd.DataFrame(
            [
                {"model": "yolo", "precision": 0.72, "recall": 0.68, "map50": 0.70, "fps": 32.0},
                {"model": "faster_rcnn", "precision": 0.78, "recall": 0.65, "map50": 0.73, "fps": 7.0},
                {"model": "rtdetr", "precision": 0.75, "recall": 0.71, "map50": 0.74, "fps": 19.0},
            ]
        ),
        Path(__file__),
    )


dashboard.load_metrics = fixture_metrics
dashboard.inject_css()
dashboard.render_comparison_tab()
dashboard.render_charts_tab()
