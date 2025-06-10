"""
Analytics sub-package for the Keisei Shogi DRL evaluation system.

This package contains modules for:
- Performance analysis: Detailed metrics beyond basic win/loss/draw.
- Elo tracking: Managing and updating Elo ratings.
- Report generation: Creating human-readable and machine-parseable reports.
"""

from .elo_tracker import EloTracker
from .performance_analyzer import PerformanceAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    "PerformanceAnalyzer",
    "EloTracker",
    "ReportGenerator",
]

# Enhanced features (optional)
try:
    from .advanced_analytics import (
        AdvancedAnalytics,
        PerformanceComparison,
        StatisticalTest,
        TrendAnalysis,
    )

    __all__.extend(
        [
            "AdvancedAnalytics",
            "StatisticalTest",
            "TrendAnalysis",
            "PerformanceComparison",
        ]
    )
except ImportError:
    pass
