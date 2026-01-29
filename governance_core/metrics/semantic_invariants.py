"""
Semantic Invariant Metrics

Validates business logic, domain rules, and cross-field constraints.

Metrics:
- Field-level constraints (age >= 0, etc.)
- Cross-field logic (salary correlates with experience)
- Domain-specific rules from policy
- Temporal consistency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SemanticInvariantMetrics:
    """
    Validate semantic constraints and domain rules.
    
    Checks:
    - Field-level constraints from column_dictionary
    - Custom cross-field rules
    - Domain-specific business logic
    
    Example:
        >>> metrics = SemanticInvariantMetrics(config)
        >>> result = metrics.compute_all(synthetic_df)
        >>> print(result['total_semantic_violations'])
    """
    
    def __init__(self, config: Optional[object] = None):
        """
        Initialize semantic validator.
        
        Args:
            config: Configuration object with column_dictionary and thresholds
        """
        self.config = config
    
    def validate_field_constraints(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate field-level constraints from column_dictionary.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary mapping field names to violation counts
        """
        violations = {}
        
        if not self.config:
            return violations
        
        for col in df.columns:
            constraints = self.config.get_field_constraints(col)
            if not constraints:
                continue
            
            # Range constraints
            range_min = constraints.get("range_min")
            range_max = constraints.get("range_max")
            
            if range_min is not None or range_max is not None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].dropna()
                    count = 0
                    
                    if range_min is not None:
                        count += (non_null < range_min).sum()
                    if range_max is not None:
                        count += (non_null > range_max).sum()
                    
                    if count > 0:
                        violations[col] = int(count)
            
            # Enum constraints
            allowed_values = constraints.get("allowed_values")
            if allowed_values:
                non_null = df[col].dropna()
                invalid = ~non_null.isin(allowed_values)
                count = invalid.sum()
                
                if count > 0:
                    violations[col] = violations.get(col, 0) + int(count)
            
            # Nullability constraints
            nullable = constraints.get("nullable", True)
            if not nullable:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    violations[col] = violations.get(col, 0) + int(null_count)
        
        return violations
    
    def validate_cross_field_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Validate cross-field logic rules.
        
        Example rules:
        - age > child_count * 15
        - start_date < end_date
        - salary correlates with experience
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of rule violations with details
        """
        violations = []
        
        # Rule: age >= 18 if has_drivers_license == True
        if "age" in df.columns and "has_drivers_license" in df.columns:
            invalid = df[(df["has_drivers_license"] == True) & (df["age"] < 18)]
            if len(invalid) > 0:
                violations.append({
                    "rule": "age_drivers_license_consistency",
                    "description": "Age must be >= 18 if has driver's license",
                    "count": len(invalid),
                    "severity": "high"
                })
        
        # Rule: start_date < end_date
        if "start_date" in df.columns and "end_date" in df.columns:
            try:
                df_temp = df[["start_date", "end_date"]].copy()
                df_temp["start_date"] = pd.to_datetime(df_temp["start_date"], errors='coerce')
                df_temp["end_date"] = pd.to_datetime(df_temp["end_date"], errors='coerce')
                
                invalid = df_temp[df_temp["start_date"] > df_temp["end_date"]].dropna()
                if len(invalid) > 0:
                    violations.append({
                        "rule": "temporal_consistency",
                        "description": "start_date must be before end_date",
                        "count": len(invalid),
                        "severity": "high"
                    })
            except Exception as e:
                logger.warning(f"Could not validate temporal consistency: {e}")
        
        # Rule: salary should correlate with experience
        if "salary" in df.columns and "years_experience" in df.columns:
            try:
                df_clean = df[["salary", "years_experience"]].dropna()
                if len(df_clean) > 10:
                    correlation = df_clean["salary"].corr(df_clean["years_experience"])
                    
                    if correlation < 0.1:  # Very weak or negative correlation
                        violations.append({
                            "rule": "salary_experience_correlation",
                            "description": f"Salary-experience correlation too weak: {correlation:.2f}",
                            "count": 0,  # Not a per-record violation
                            "severity": "medium",
                            "correlation": float(correlation)
                        })
            except Exception as e:
                logger.warning(f"Could not validate salary-experience correlation: {e}")
        
        return violations
    
    def validate_domain_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate domain-specific rules.
        
        These are application-specific business logic rules.
        Can be extended by loading from policy configuration.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with domain rule violations
        """
        violations = {}
        total_violations = 0
        
        # Example domain rule: amount must be positive for purchase transactions
        if "amount" in df.columns and "transaction_type" in df.columns:
            invalid = df[(df["transaction_type"] == "purchase") & (df["amount"] <= 0)]
            if len(invalid) > 0:
                violations["positive_purchase_amount"] = len(invalid)
                total_violations += len(invalid)
        
        # Add more domain-specific rules here or load from config
        
        return {
            "by_rule": violations,
            "total": total_violations
        }
    
    def compute_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all semantic invariant metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Comprehensive dictionary of semantic violations
        """
        logger.info("Computing semantic invariant metrics...")
        
        # Field-level constraints
        field_violations = self.validate_field_constraints(df)
        
        # Cross-field rules
        cross_field_violations = self.validate_cross_field_rules(df)
        
        # Domain rules
        domain_violations = self.validate_domain_rules(df)
        
        total_semantic_violations = (
            sum(field_violations.values()) +
            sum(v["count"] for v in cross_field_violations) +
            domain_violations["total"]
        )
        
        result = {
            "field_level_violations": field_violations,
            "cross_field_violations": cross_field_violations,
            "domain_rule_violations": domain_violations,
            "total_semantic_violations": total_semantic_violations
        }
        
        logger.info(f"Semantic validation completed: {total_semantic_violations} total violations")
        return result
