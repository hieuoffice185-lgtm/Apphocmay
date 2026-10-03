"""
Package tabs: Chứa các module giao diện và thuật toán cho từng Tab.
"""

from .tab1_clustering import render_tab1
from .tab2_pca import render_tab2
from .tab3_classification import render_tab3
from .tab4_detection import render_tab4

__all__ = [
    "render_tab1",
    "render_tab2",
    "render_tab3",
    "render_tab4",
]
