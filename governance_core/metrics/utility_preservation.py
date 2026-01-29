"""
Utility Preservation Metrics

Measures whether synthetic data preserves utility for ML tasks.

Metrics:
- Task-based utility (train on synthetic, test on real)
- Predictive similarity
- Feature importance consistency
- Overall utility score (0.0 - 1.0)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class UtilityPreservationMetrics:
    """
    Evaluate utility preservation in synthetic data.
    
    Compares ML model performance when trained on synthetic vs real data.
    
    Example:
        >>> metrics = UtilityPreservationMetrics(target_column="label")
        >>> result = metrics.compute_all(synthetic_df, original_df)
        >>> print(result['utility_score'])  # 0.0 (no utility) to 1.0 (perfect)
    """
    
    def __init__(
        self,
        target_column: Optional[str] = None,
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize utility metrics calculator.
        
        Args:
            target_column: Target variable for supervised learning
            test_size: Fraction of data to use for testing
            random_state: Random seed
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
    
    def compute_task_utility(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute task-based utility metrics.
        
        Trains models on synthetic and real data, compares performance.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            target_column: Target variable (overrides init)
            
        Returns:
            Dictionary with utility metrics
        """
        target_col = target_column or self.target_column
        
        if not target_col:
            raise ValueError("target_column must be specified")
        
        if target_col not in synthetic_df.columns or target_col not in original_df.columns:
            raise ValueError(f"Target column {target_col} not found in data")
        
        # Get numeric feature columns
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c in original_df.columns]
        
        if len(feature_cols) == 0:
            logger.warning("No common numeric features for utility calculation")
            return {
                "synthetic_train_real_test_accuracy": None,
                "real_train_real_test_accuracy": None,
                "accuracy_gap": None,
                "utility_score": 0.5
            }
        
        # Prepare real data
        real_clean = original_df[feature_cols + [target_col]].dropna()
        X_real = real_clean[feature_cols].values
        y_real = real_clean[target_col].values
        
        # Prepare synthetic data
        syn_clean = synthetic_df[feature_cols + [target_col]].dropna()
        X_syn = syn_clean[feature_cols].values
        y_syn = syn_clean[target_col].values
        
        if len(X_real) < 50 or len(X_syn) < 50:
            logger.warning("Insufficient data for utility calculation")
            return {
                "synthetic_train_real_test_accuracy": None,
                "real_train_real_test_accuracy": None,
                "accuracy_gap": None,
                "utility_score": 0.5
            }
        
        # Split real data for testing
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Model 1: Train on synthetic, test on real
        clf_syn = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.random_state)
        clf_syn.fit(X_syn, y_syn)
        y_pred_syn_trained = clf_syn.predict(X_real_test)
        acc_syn_trained = accuracy_score(y_real_test, y_pred_syn_trained)
        
        # Model 2: Train on real, test on real (baseline)
        clf_real = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.random_state)
        clf_real.fit(X_real_train, y_real_train)
        y_pred_real_trained = clf_real.predict(X_real_test)
        acc_real_trained = accuracy_score(y_real_test, y_pred_real_trained)
        
        # Accuracy gap (lower is better for utility)
        accuracy_gap = abs(acc_real_trained - acc_syn_trained)
        
        return {
            "synthetic_train_real_test_accuracy": float(acc_syn_trained),
            "real_train_real_test_accuracy": float(acc_real_trained),
            "accuracy_gap": float(accuracy_gap)
        }
    
    def compute_feature_importance_consistency(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compare feature importance between models trained on synthetic vs real.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            target_column: Target variable
            
        Returns:
            Dictionary with feature importance correlation
        """
        target_col = target_column or self.target_column
        
        if not target_col:
            return {"feature_importance_correlation": None}
        
        if target_col not in synthetic_df.columns or target_col not in original_df.columns:
            return {"feature_importance_correlation": None}
        
        # Get features
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c in original_df.columns]
        
        if len(feature_cols) < 2:
            return {"feature_importance_correlation": None}
        
        # Prepare data
        real_clean = original_df[feature_cols + [target_col]].dropna()
        syn_clean = synthetic_df[feature_cols + [target_col]].dropna()
        
        if len(real_clean) < 50 or len(syn_clean) < 50:
            return {"feature_importance_correlation": None}
        
        X_real = real_clean[feature_cols].values
        y_real = real_clean[target_col].values
        X_syn = syn_clean[feature_cols].values
        y_syn = syn_clean[target_col].values
        
        # Train models
        clf_real = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.random_state)
        clf_real.fit(X_real, y_real)
        
        clf_syn = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.random_state)
        clf_syn.fit(X_syn, y_syn)
        
        # Get feature importances
        importance_real = clf_real.feature_importances_
        importance_syn = clf_syn.feature_importances_
        
        # Compute correlation
        correlation = np.corrcoef(importance_real, importance_syn)[0, 1]
        
        return {
            "feature_importance_correlation": float(correlation),
            "feature_names": feature_cols
        }
    
    def compute_utility_score(
        self,
        accuracy_gap: Optional[float],
        feature_importance_corr: Optional[float]
    ) -> float:
        """
        Compute overall utility score (0.0 - 1.0).
        
        Args:
            accuracy_gap: Gap between synthetic-trained and real-trained models
            feature_importance_corr: Correlation of feature importances
            
        Returns:
            Utility score (1.0 = perfect utility preservation)
        """
        scores = []
        
        # Accuracy component (lower gap = higher utility)
        if accuracy_gap is not None:
            # Assume gap > 0.2 is very bad
            acc_score = max(1.0 - accuracy_gap / 0.2, 0.0)
            scores.append(acc_score)
        
        # Feature importance component
        if feature_importance_corr is not None:
            # Correlation should be high
            fi_score = max(feature_importance_corr, 0.0)
            scores.append(fi_score)
        
        if not scores:
            return 0.5  # Neutral if no metrics available
        
        return float(np.mean(scores))
    
    def classify_utility(self, utility_score: float) -> str:
        """
        Classify utility preservation level.
        
        Args:
            utility_score: Utility score (0.0 - 1.0)
            
        Returns:
            "high", "medium", or "low"
        """
        if utility_score >= 0.7:
            return "high"
        elif utility_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def compute_all(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute all utility preservation metrics.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            target_column: Target variable
            
        Returns:
            Comprehensive dictionary of utility metrics
        """
        logger.info("Computing utility preservation metrics...")
        
        target_col = target_column or self.target_column
        
        result = {}
        
        # Task-based utility
        if target_col:
            try:
                task_metrics = self.compute_task_utility(synthetic_df, original_df, target_col)
                result.update(task_metrics)
            except Exception as e:
                logger.warning(f"Could not compute task utility: {e}")
                result["synthetic_train_real_test_accuracy"] = None
                result["real_train_real_test_accuracy"] = None
                result["accuracy_gap"] = None
            
            # Feature importance
            try:
                fi_metrics = self.compute_feature_importance_consistency(
                    synthetic_df, original_df, target_col
                )
                result.update(fi_metrics)
            except Exception as e:
                logger.warning(f"Could not compute feature importance: {e}")
                result["feature_importance_correlation"] = None
        else:
            logger.warning("No target column specified, skipping utility metrics")
            result["synthetic_train_real_test_accuracy"] = None
            result["real_train_real_test_accuracy"] = None
            result["accuracy_gap"] = None
            result["feature_importance_correlation"] = None
        
        # Overall utility score
        utility_score = self.compute_utility_score(
            result.get("accuracy_gap"),
            result.get("feature_importance_correlation")
        )
        
        result["utility_score"] = utility_score
        result["utility_assessment"] = self.classify_utility(utility_score)
        
        logger.info(f"Utility preservation computed: score={utility_score:.2f}, assessment={result['utility_assessment']}")
        return result
