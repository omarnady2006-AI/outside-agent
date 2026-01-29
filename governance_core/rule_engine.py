"""
Rule Engine - Deterministic execution engine for governance

Orchestrates all rule-based computations:
- Statistical fidelity metrics
- Privacy risk assessment
- Semantic validation
- Utility preservation

CRITICAL: This engine is DETERMINISTIC ONLY.
- NO LLM calls
- NO AI-based decisions
- Pure metric computation and rule execution
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import logging
import time

from .metrics import (
    StatisticalFidelityMetrics,
    PrivacyRiskMetrics,
    SemanticInvariantMetrics,
    UtilityPreservationMetrics
)
from .data_profiles import DataProfiler, DatasetProfile
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Deterministic rule engine for synthetic data governance.
    
    Computes all metrics without any AI/LLM involvement.
    Produces structured, score-based outputs for GovernanceAgent interpretation.
    
    Example:
        >>> engine = RuleEngine(config=config, audit_logger=logger)
        >>> result = engine.evaluate_synthetic_data(
        ...     synthetic_df=syn_df,
        ...     original_profile=profile
        ... )
        >>> print(result['privacy_score'])
        >>> print(result['utility_score'])
    """
    
    def __init__(
        self,
        config: Optional[object] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        """
        Initialize rule engine.
        
        Args:
            config: Configuration object with thresholds and policies
            audit_logger: AuditLogger instance for recording metrics
        """
        self.config = config
        self.audit_logger = audit_logger or AuditLogger()
        
        # Initialize metric calculators
        self.stat_fidelity = StatisticalFidelityMetrics()
        self.privacy_risk = PrivacyRiskMetrics()
        self.semantic_inv = SemanticInvariantMetrics(config=config)
        self.utility_metrics = UtilityPreservationMetrics()
        
        self.profiler = DataProfiler()
    
    def compute_statistical_fidelity(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[DatasetProfile] = None
    ) -> Dict[str, Any]:
        """
        Compute all statistical fidelity metrics.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (if available)
            original_profile: DatasetProfile (if original_df unavailable)
            
        Returns:
            Dictionary of statistical fidelity metrics
        """
        start_time = time.time()
        
        metrics = self.stat_fidelity.compute_all(
            synthetic_df, original_df, original_profile
        )
        
        computation_time = (time.time() - start_time) * 1000  # ms
        
        # Log to audit trail
        if self.audit_logger:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, str)):
                    self.audit_logger.log_metric_computation(
                        eval_id="",  # Will be set by caller
                        metric_name=f"stat_fidelity.{metric_name}",
                        value=value,
                        computation_time_ms=computation_time
                    )
        
        return metrics
    
    def compute_semantic_violations(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute semantic invariant violations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of semantic violations
        """
        start_time = time.time()
        
        metrics = self.semantic_inv.compute_all(df)
        
        computation_time = (time.time() - start_time) * 1000
        
        if self.audit_logger:
            self.audit_logger.log_metric_computation(
                eval_id="",
                metric_name="semantic.total_violations",
                value=metrics["total_semantic_violations"],
                computation_time_ms=computation_time
            )
        
        return metrics
    
    def compute_privacy_risk(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[DatasetProfile] = None
    ) -> Dict[str, Any]:
        """
        Compute privacy risk metrics.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (if available)
            original_profile: DatasetProfile (if original_df unavailable)
            
        Returns:
            Dictionary of privacy metrics including privacy_score (0.0-1.0)
        """
        start_time = time.time()
        
        metrics = self.privacy_risk.compute_all(
            synthetic_df, original_df, original_profile
        )
        
        computation_time = (time.time() - start_time) * 1000
        
        if self.audit_logger:
            self.audit_logger.log_metric_computation(
                eval_id="",
                metric_name="privacy.privacy_score",
                value=metrics["privacy_score"],
                computation_time_ms=computation_time
            )
        
        return metrics
    
    def compute_utility_preservation(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute utility preservation metrics.
        
        Note: Requires original DataFrame (cannot be computed from profile).
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            target_column: Target variable for supervised learning
            
        Returns:
            Dictionary of utility metrics including utility_score (0.0-1.0)
        """
        start_time = time.time()
        
        # Detect target column if not specified
        if not target_column and "label" in synthetic_df.columns:
            target_column = "label"
        
        if target_column:
            self.utility_metrics.target_column = target_column
            metrics = self.utility_metrics.compute_all(synthetic_df, original_df)
        else:
            logger.warning("No target column specified, utility metrics unavailable")
            metrics = {
                "utility_score": 0.5,
                "utility_assessment": "unknown",
                "synthetic_train_real_test_accuracy": None,
                "real_train_real_test_accuracy": None
            }
        
        computation_time = (time.time() - start_time) * 1000
        
        if self.audit_logger:
            self.audit_logger.log_metric_computation(
                eval_id="",
                metric_name="utility.utility_score",
                value=metrics["utility_score"],
                computation_time_ms=computation_time
            )
        
        return metrics
    
    def evaluate_synthetic_data(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[DatasetProfile] = None,
        eval_id: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of synthetic data.
        
        Computes ALL metrics and returns structured result.
        
        Args:
            synthetic_df: Synthetic DataFrame to evaluate
            original_df: Original DataFrame (preferred)
            original_profile: DatasetProfile (if original_df unavailable)
            eval_id: Unique evaluation identifier
            target_column: Target variable for utility metrics
            
        Returns:
            Comprehensive evaluation result with all metrics and scores
        """
        import uuid
        from datetime import datetime
        
        eval_id = eval_id or f"eval_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting evaluation {eval_id}")
        
        # Update audit logger with eval_id (hacky but works for this pattern)
        original_log_metric = self.audit_logger.log_metric_computation
        def log_with_id(*args, **kwargs):
            kwargs['eval_id'] = eval_id
            return original_log_metric(*args, **kwargs)
        self.audit_logger.log_metric_computation = log_with_id
        
        result = {
            "eval_id": eval_id,
            "timestamp": datetime.now().isoformat(),
            "synthetic_rows": len(synthetic_df),
            "synthetic_columns": len(synthetic_df.columns)
        }
        
        # 1. Statistical Fidelity
        logger.info("Computing statistical fidelity...")
        result["statistical_fidelity"] = self.compute_statistical_fidelity(
            synthetic_df, original_df, original_profile
        )
        
        # 2. Privacy Risk
        logger.info("Computing privacy risk...")
        result["privacy_risk"] = self.compute_privacy_risk(
            synthetic_df, original_df, original_profile
        )
        
        # Extract top-level scores
        result["privacy_score"] = result["privacy_risk"]["privacy_score"]
        result["leakage_risk_level"] = result["privacy_risk"]["leakage_risk_level"]
        
        # 3. Semantic Invariants (on synthetic data only)
        logger.info("Computing semantic violations...")
        result["semantic_validation"] = self.compute_semantic_violations(synthetic_df)
        result["semantic_violations"] = result["semantic_validation"]["total_semantic_violations"]
        
        # 4. Utility Preservation (requires original DataFrame)
        if original_df is not None:
            logger.info("Computing utility preservation...")
            result["utility_preservation"] = self.compute_utility_preservation(
                synthetic_df, original_df, target_column
            )
            result["utility_score"] = result["utility_preservation"]["utility_score"]
            result["utility_assessment"] = result["utility_preservation"]["utility_assessment"]
        else:
            logger.warning("Original DataFrame not available, skipping utility metrics")
            result["utility_score"] = None
            result["utility_assessment"] = "unknown"
        
        # 5. Statistical drift classification
        result["statistical_drift"] = result["statistical_fidelity"].get(
            "drift_classification", "unknown"
        )
        
        # Restore original logging function
        self.audit_logger.log_metric_computation = original_log_metric
        
        logger.info(
            f"Evaluation {eval_id} complete: "
            f"privacy={result['privacy_score']:.2f}, "
            f"utility={result.get('utility_score', 'N/A')}, "
            f"drift={result['statistical_drift']}"
        )
        
        return result
