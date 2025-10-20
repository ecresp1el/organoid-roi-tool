"""Named analysis objects that operate on simple projection exports.

Each analysis encapsulates the biological question it answers, including how to
locate projection data, what statistics to calculate, and how to present the
results.  The registry below allows the command-line helper to instantiate an
analysis by name.
"""

from .base import ProjectionAnalysis
from .pcdh_vs_lhx6 import PCDHvsLHX6_WTvsKOIHCAnalysis

ANALYSIS_REGISTRY = {
    "PCDHvsLHX6_WTvsKO_IHC": PCDHvsLHX6_WTvsKOIHCAnalysis,
    # Future analyses can be added here, e.g. "NewQuestion": NewAnalysisClass,
}

__all__ = [
    "ProjectionAnalysis",
    "PCDHvsLHX6_WTvsKOIHCAnalysis",
    "ANALYSIS_REGISTRY",
]
