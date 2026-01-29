"""
Leakage detection metrics for synthetic data governance.

Modules:
- statistical_fidelity: Distribution comparison, KL divergence, Wasserstein distance
- privacy_risk: Near-duplicates, membership inference, privacy scoring
- semantic_invariants: Field constraints, cross-field logic, domain rules
- utility_preservation: Task-based utility, predictive similarity
"""

from .statistical_fidelity import StatisticalFidelityMetrics
from .privacy_risk import PrivacyRiskMetrics
from .semantic_invariants import SemanticInvariantMetrics
from .utility_preservation import UtilityPreservationMetrics

__all__ = [
    "StatisticalFidelityMetrics",
    "PrivacyRiskMetrics",
    "SemanticInvariantMetrics",
    "UtilityPreservationMetrics",
]
