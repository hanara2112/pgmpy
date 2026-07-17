"""Backward-compatible imports for ANM scores.

New code should import these classes from :mod:`pgmpy.causal_discovery.bivariate_scores`.
"""

from pgmpy.causal_discovery.bivariate_scores import (
    BaseANMScore,
    EntropyScore,
    GaussScore,
    IndependenceScore,
    get_anm_score,
)

__all__ = [
    "BaseANMScore",
    "EntropyScore",
    "GaussScore",
    "IndependenceScore",
    "get_anm_score",
]
