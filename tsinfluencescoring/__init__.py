"""
TSInfluenceScoring: A framework for selecting influential timestamps from time-series.

This package provides modular components for:
- Attention-based timestamp selection
- Sparse selection mechanisms
- Loss functions for task consistency, diversity, and mutual information
- Counterfactual generation from selected timestamps
"""

__version__ = "0.1.0"

from .selector import TimestampSelector
from .losses import (
    TaskConsistencyLoss,
    DiversityLoss,
    MutualInformationLoss,
    CombinedLoss
)
from .counterfactual import CounterfactualGenerator, CounterfactualExplainer
from .utils import (
    InfluenceFramework,
    ModelAgnosticWrapper,
    create_simple_framework,
    visualize_selection
)

__all__ = [
    "TimestampSelector",
    "TaskConsistencyLoss",
    "DiversityLoss",
    "MutualInformationLoss",
    "CombinedLoss",
    "CounterfactualGenerator",
    "CounterfactualExplainer",
    "InfluenceFramework",
    "ModelAgnosticWrapper",
    "create_simple_framework",
    "visualize_selection",
]
