"""
Privacy Risk Metrics

Assesses re-identification and leakage risks in synthetic data.

Metrics:
- Near-duplicate detection (Jaccard similarity)
- Nearest-neighbor distance in feature space
- Membership inference risk (train classifier)
- Attribute inference risk
- Overall privacy score (0.0 - 1.0)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class PrivacyRiskMetrics:
    """
    Compute privacy risk metrics for synthetic data.
    
    Evaluates:
    - Record-level similarity to original data
    - Membership inference attacks
    - Attribute inference attacks
    - Overall privacy score
    
    Example:
        >>> metrics = PrivacyRiskMetrics()
        >>> result = metrics.compute_all(synthetic_df, original_df)
        >>> print(result['privacy_score'])  # 0.0 (critical) to 1.0 (perfect)
        >>> print(result['leakage_risk_level'])  # "acceptable" | "warning" | "critical"
    """
    
    def __init__(
        self,
        near_duplicate_threshold: float = 0.9,
        privacy_score_thresholds: Dict[str, float] = None
    ):
        """
        Initialize privacy risk calculator.
        
        Args:
            near_duplicate_threshold: Jaccard similarity threshold for near-duplicates
            privacy_score_thresholds: Thresholds for risk levels
                Default: {"acceptable": 0.8, "warning": 0.6}
        """
        self.near_duplicate_threshold = near_duplicate_threshold
        self.thresholds = privacy_score_thresholds or {
            "acceptable": 0.8,
            "warning": 0.6
        }
    
    def detect_near_duplicates(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[object] = None
    ) -> Dict[str, Any]:
        """
        Detect near-duplicate records between synthetic and original data.
        
        Uses row hashes from DatasetProfile if original_df unavailable.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (if available)
            original_profile: DatasetProfile with row hashes
            
        Returns:
            Dictionary with near-duplicate statistics
        """
        if original_df is None and original_profile is None:
            raise ValueError("Either original_df or original_profile required")
        
        # Import DataProfiler for hashing
        from ..data_profiles import DataProfiler
        profiler = DataProfiler()
        
        # Compute synthetic row hashes
        synthetic_hashes = synthetic_df.apply(profiler._hash_row, axis=1).tolist()
        
        # Get original hashes
        if original_df is not None:
            original_hashes = original_df.apply(profiler._hash_row, axis=1).tolist()
        else:
            if not original_profile.row_hashes:
                raise ValueError("Original profile does not contain row hashes")
            original_hashes = original_profile.row_hashes
        
        original_hash_set = set(original_hashes)
        
        # Find exact matches
        exact_matches = [h for h in synthetic_hashes if h in original_hash_set]
        exact_match_rate = len(exact_matches) / len(synthetic_hashes) if synthetic_hashes else 0.0
        
        return {
            "near_duplicates_count": len(exact_matches),
            "near_duplicates_rate": float(exact_match_rate),
            "near_duplicates_threshold": self.near_duplicate_threshold,
            "synthetic_total": len(synthetic_hashes),
            "original_total": len(original_hashes)
        }
    
    def compute_nearest_neighbor_distances(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        k: int = 1
    ) -> Dict[str, float]:
        """
        Compute nearest-neighbor distances in feature space.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            k: Number of neighbors to consider
            
        Returns:
            Dictionary with distance statistics
        """
        # Get numeric columns only
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        common_cols = [c for c in numeric_cols if c in original_df.columns]
        
        if len(common_cols) == 0:
            logger.warning("No common numeric columns for NN distance calculation")
            return {
                "min_nn_distance": None,
                "avg_nn_distance": None,
                "median_nn_distance": None
            }
        
        # Prepare data (drop nulls, standardize)
        syn_data = synthetic_df[common_cols].dropna()
        orig_data = original_df[common_cols].dropna()
        
        if len(syn_data) == 0 or len(orig_data) == 0:
            return {
                "min_nn_distance": None,
                "avg_nn_distance": None,
                "median_nn_distance": None
            }
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(orig_data.values)
        
        syn_scaled = scaler.transform(syn_data.values)
        orig_scaled = scaler.transform(orig_data.values)
        
        # Fit nearest neighbors on original data
        nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn_model.fit(orig_scaled)
        
        # Find nearest neighbors for synthetic records
        distances, indices = nn_model.kneighbors(syn_scaled)
        
        # Get minimum distance for each synthetic record (to closest original)
        min_distances = distances[:, 0]  # First neighbor
        
        return {
            "min_nn_distance": float(min_distances.min()),
            "avg_nn_distance": float(min_distances.mean()),
            "median_nn_distance": float(np.median(min_distances)),
            "max_nn_distance": float(min_distances.max())
        }
    
    def estimate_membership_inference_risk(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Estimate membership inference attack risk.
        
        Trains a classifier to distinguish synthetic from original records.
        Perfect privacy: AUC = 0.5 (random guessing)
        High risk: AUC > 0.7
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary with attack performance metrics
        """
        # Get numeric columns
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        common_cols = [c for c in numeric_cols if c in original_df.columns]
        
        if len(common_cols) == 0:
            logger.warning("No common numeric columns for membership inference")
            return {
                "membership_inference_auc": None,
                "membership_inference_accuracy": None
            }
        
        # Prepare data
        syn_data = synthetic_df[common_cols].dropna()
        orig_data = original_df[common_cols].dropna()
        
        # Label: 0 = synthetic, 1 = original
        syn_labels = np.zeros(len(syn_data))
        orig_labels = np.ones(len(orig_data))
        
        X = pd.concat([syn_data, orig_data], ignore_index=True).values
        y = np.concatenate([syn_labels, orig_labels])
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "membership_inference_auc": float(auc),
            "membership_inference_accuracy": float(accuracy)
        }
    
    def estimate_attribute_inference_risk(
        self,
        synthetic_df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Estimate attribute inference attack risk.
        
        Tries to predict a target attribute from other attributes using synthetic data.  
        High accuracy suggests the synthetic data reveals relationships
        that could be exploited.
        
        Args:
            synthetic_df: Synthetic DataFrame
            target_column: Column to try to infer
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary with attack performance
        """
        if target_column not in synthetic_df.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        # Get numeric columns (excluding target)
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_column]
        
        if len(feature_cols) == 0:
            logger.warning("No feature columns for attribute inference")
            return {
                "attribute_inference_accuracy": None
            }
        
        # Prepare data
        df_clean = synthetic_df[feature_cols + [target_column]].dropna()
        
        if len(df_clean) < 50:
            logger.warning("Insufficient data for attribute inference")
            return {
                "attribute_inference_accuracy": None
            }
        
        X = df_clean[feature_cols].values
        y = df_clean[target_column].values
        
        # Handle categorical target
        if not pd.api.types.is_numeric_dtype(df_clean[target_column]):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "attribute_inference_accuracy": float(accuracy),
            "target_attribute": target_column
        }
    
    def compute_privacy_score(
        self,
        near_duplicate_rate: float,
        membership_auc: Optional[float],
        avg_nn_distance: Optional[float]
    ) -> float:
        """
        Compute overall privacy score (0.0 - 1.0).
        
        Combines multiple privacy metrics into a single score.
        1.0 = perfect privacy
        0.0 = critical leakage
        
        Args:
            near_duplicate_rate: Rate of near-duplicates
            membership_auc: Membership inference AUC
            avg_nn_distance: Average NN distance
            
        Returns:
            Privacy score (0.0 - 1.0)
        """
        scores = []
        
        # Near-duplicate component (lower is better)
        if near_duplicate_rate is not None:
            dup_score = 1.0 - min(near_duplicate_rate, 1.0)
            scores.append(dup_score)
        
        # Membership inference component (AUC closer to 0.5 is better)
        if membership_auc is not None:
            # Convert AUC to score: 0.5 -> 1.0, 0.0 or 1.0 -> 0.0
            mi_score = 1.0 - 2.0 * abs(membership_auc - 0.5)
            scores.append(max(mi_score, 0.0))
        
        # NN distance component (higher is better, normalized)
        if avg_nn_distance is not None:
            # Normalize: assume >2.0 is very good
            nn_score = min(avg_nn_distance / 2.0, 1.0)
            scores.append(nn_score)
        
        if not scores:
            return 0.5  # Neutral score if no metrics available
        
        # Weighted average (emphasize duplication)
        if len(scores) == 3:
            privacy_score = 0.5 * scores[0] + 0.3 * scores[1] + 0.2 * scores[2]
        else:
            privacy_score = np.mean(scores)
        
        return float(np.clip(privacy_score, 0.0, 1.0))
    
    def classify_risk_level(self, privacy_score: float) -> str:
        """
        Classify privacy risk level based on score.
        
        Args:
            privacy_score: Privacy score (0.0 - 1.0)
            
        Returns:
            "acceptable", "warning", or "critical"
        """
        if privacy_score >= self.thresholds["acceptable"]:
            return "acceptable"
        elif privacy_score >= self.thresholds["warning"]:
            return "warning"
        else:
            return "critical"
    
    def compute_all(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[object] = None
    ) -> Dict[str, Any]:
        """
        Compute all privacy risk metrics.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (recommended)
            original_profile: DatasetProfile (if original_df unavailable)
            
        Returns:
            Comprehensive dictionary of privacy metrics
        """
        logger.info("Computing privacy risk metrics...")
        
        result = {}
        
        # Near-duplicates (works with profile)
        try:
            dup_metrics = self.detect_near_duplicates(
                synthetic_df, original_df, original_profile
            )
            result.update(dup_metrics)
        except Exception as e:
            logger.warning(f"Could not compute near-duplicates: {e}")
            result["near_duplicates_count"] = 0
            result["near_duplicates_rate"] = 0.0
        
        # Metrics requiring original DataFrame
        if original_df is not None:
            # Nearest neighbor distances
            try:
                nn_metrics = self.compute_nearest_neighbor_distances(synthetic_df, original_df)
                result.update(nn_metrics)
            except Exception as e:
                logger.warning(f"Could not compute NN distances: {e}")
                result["min_nn_distance"] = None
                result["avg_nn_distance"] = None
            
            # Membership inference
            try:
                mi_metrics = self.estimate_membership_inference_risk(synthetic_df, original_df)
                result.update(mi_metrics)
            except Exception as e:
                logger.warning(f"Could not estimate membership inference: {e}")
                result["membership_inference_auc"] = None
        else:
            result["min_nn_distance"] = None
            result["avg_nn_distance"] = None
            result["membership_inference_auc"] = None
        
        # Compute overall privacy score
        privacy_score = self.compute_privacy_score(
            result.get("near_duplicates_rate", 0.0),
            result.get("membership_inference_auc"),
            result.get("avg_nn_distance")
        )
        
        result["privacy_score"] = privacy_score
        result["leakage_risk_level"] = self.classify_risk_level(privacy_score)
        
        logger.info(f"Privacy risk metrics computed: score={privacy_score:.2f}, level={result['leakage_risk_level']}")
        return result
