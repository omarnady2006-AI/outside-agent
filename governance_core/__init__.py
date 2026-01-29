"""
Hybrid Data Governance Agent for Synthetic Data

Production-grade system for evaluating synthetic datasets against:
- Privacy leakage risk
- Statistical fidelity
- Semantic correctness
- Utility preservation

Architecture:
- RuleEngine: Deterministic metric computation
- GovernanceAgent: LLM-powered interpretation (Ollama default)
- AuditLogger: Complete audit trails
- DataProfiler: Privacy-preserving statistical profiles

Example:
    >>> from governance_core import RuleEngine, GovernanceAgent, DataProfiler
    >>> 
    >>> # Create profile from original data
    >>> profiler = DataProfiler()
    >>> profile = profiler.create_profile(original_df, profile_id="orig_001")
    >>> profile.save("profiles/original.json")
    >>> 
    >>> # Evaluate synthetic data
    >>> engine = RuleEngine()
    >>> result = engine.evaluate_synthetic_data(
    ...     synthetic_df=syn_df,
    ...     original_profile=profile
    ... )
    >>> 
    >>> # Get LLM interpretation
    >>> agent = GovernanceAgent(provider_type="ollama")
    >>> interpretation = agent.interpret_metrics(result)
    >>> print(interpretation['recommendation'])
"""

from .rule_engine import RuleEngine
from .governance_agent import GovernanceAgent
from .audit_logger import AuditLogger, AuditEntry
from .data_profiles import DataProfiler, DatasetProfile, FieldProfile
from .llm_provider import (
    LLMProvider,
    OllamaProvider,
    AnthropicProvider,
    OpenAIProvider,
    create_provider
)
from .metrics import (
    StatisticalFidelityMetrics,
    PrivacyRiskMetrics,
    SemanticInvariantMetrics,
    UtilityPreservationMetrics
)

__version__ = "2.0.0"

__all__ = [
    # Core components
    "RuleEngine",
    "GovernanceAgent",
    "AuditLogger",
    "AuditEntry",
    "DataProfiler",
    "DatasetProfile",
    "FieldProfile",
    
    # LLM providers
    "LLMProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_provider",
    
    # Metrics
    "StatisticalFidelityMetrics",
    "PrivacyRiskMetrics",
    "SemanticInvariantMetrics",
    "UtilityPreservationMetrics",
]
