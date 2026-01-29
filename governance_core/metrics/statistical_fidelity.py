"""
Statistical Fidelity Metrics

Measures how well synthetic data preserves the statistical properties
of the original dataset.

Metrics:
- Distribution comparison (mean, variance, histogram overlap)
- KL Divergence
- Wasserstein Distance
- Correlation preservation
- Drift detection (PSI, KS test)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial import distance
from scipy.stats import wasserstein_distance, ks_2samp
import logging

logger = logging.getLogger(__name__)


class StatisticalFidelityMetrics:
    """
    Compute statistical fidelity metrics comparing synthetic to original data.
    
    Works with:
    - Raw DataFrames (if original data available)
    - DatasetProfile objects (preferred for privacy)
    
    Example:
        >>> metrics = StatisticalFidelityMetrics()
        >>> result = metrics.compute_all(synthetic_df, original_profile)
        >>> print(result['kl_divergence'])  # Per-field KL divergence
        >>> print(result['drift_classification'])  # "low", "medium", or "high"
    """
    
    def __init__(
        self,
        histogram_bins: int = 30,
        drift_threshold_low: float = 0.1,
        drift_threshold_high: float = 0.25
    ):
        """
        Initialize metric calculator.
        
        Args:
            histogram_bins: Number of bins for histogram comparison
            drift_threshold_low: PSI threshold for "low" drift
            drift_threshold_high: PSI threshold for "high" drift
        """
        self.histogram_bins = histogram_bins
        self.drift_threshold_low = drift_threshold_low
        self.drift_threshold_high = drift_threshold_high
    
    def _compute_histogram_overlap(
        self,
        synthetic_values: np.ndarray,
        original_values: np.ndarray,
        bins: int
    ) -> float:
        """
        Compute histogram overlap (Bhattacharyya coefficient).
        
        Args:
            synthetic_values: Synthetic data values
            original_values: Original data values
            bins: Number of histogram bins
            
        Returns:
            Overlap coefficient (0.0 = no overlap, 1.0 = perfect overlap)
        """
        # Determine common range
        min_val = min(synthetic_values.min(), original_values.min())
        max_val = max(synthetic_values.max(), original_values.max())
        
        # Compute histograms with same bins
        syn_hist, _ = np.histogram(synthetic_values, bins=bins, range=(min_val, max_val), density=True)
        orig_hist, _ = np.histogram(original_values, bins=bins, range=(min_val, max_val), density=True)
        
        # Normalize
        syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
        orig_hist = orig_hist / (orig_hist.sum() + 1e-10)
        
        # Bhattacharyya coefficient
        overlap = np.sum(np.sqrt(syn_hist * orig_hist))
        
        return float(overlap)
    
    def _compute_kl_divergence(
        self,
        synthetic_values: np.ndarray,
        original_values: np.ndarray,
        bins: int
    ) -> float:
        """
        Compute KL divergence between distributions.
        
        Args:
            synthetic_values: Synthetic data values
            original_values: Original data values
            bins: Number of histogram bins
            
        Returns:
            KL divergence (0.0 = identical, higher = more different)
        """
        # Determine common range
        min_val = min(synthetic_values.min(), original_values.min())
        max_val = max(synthetic_values.max(), original_values.max())
        
        # Compute histograms
        syn_hist, _ = np.histogram(synthetic_values, bins=bins, range=(min_val, max_val), density=True)
        orig_hist, _ = np.histogram(original_values, bins=bins, range=(min_val, max_val), density=True)
        
        # Normalize and add small constant to avoid log(0)
        epsilon = 1e-10
        syn_hist = (syn_hist + epsilon) / (syn_hist.sum() + bins * epsilon)
        orig_hist = (orig_hist + epsilon) / (orig_hist.sum() + bins * epsilon)
        
        # KL divergence
        kl_div = np.sum(orig_hist * np.log(orig_hist / syn_hist))
        
        return float(kl_div)
    
    def _compute_psi(
        self,
        synthetic_values: np.ndarray,
        original_values: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI interpretation:
        - < 0.1: No significant drift
        - 0.1 - 0.25: Moderate drift
        - > 0.25: Significant drift
        
        Args:
            synthetic_values: Synthetic data values
            original_values: Original data values
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Determine common range
        min_val = min(synthetic_values.min(), original_values.min())
        max_val = max(synthetic_values.max(), original_values.max())
        
        # Compute histograms
        syn_hist, _ = np.histogram(synthetic_values, bins=bins, range=(min_val, max_val))
        orig_hist, _ = np.histogram(original_values, bins=bins, range=(min_val, max_val))
        
        # Normalize
        epsilon = 1e-10
        syn_pct = (syn_hist + epsilon) / (syn_hist.sum() + bins * epsilon)
        orig_pct = (orig_hist + epsilon) / (orig_hist.sum() + bins * epsilon)
        
        # PSI formula
        psi = np.sum((syn_pct - orig_pct) * np.log(syn_pct / orig_pct))
        
        return float(psi)
    
    def compute_distribution_metrics(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[object] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute distribution comparison metrics for all numeric fields.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (if available)
            original_profile: DatasetProfile (if original_df not available)
            
        Returns:
            Dictionary with per-field metrics:
                - mean_difference
                - variance_ratio
                - histogram_overlap
        """
        if original_df is None and original_profile is None:
            raise ValueError("Either original_df or original_profile required")
        
        metrics = {
            "mean_difference": {},
            "variance_ratio": {},
            "histogram_overlap": {}
        }
        
        # Get numeric columns
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            syn_values = synthetic_df[col].dropna().values
            
            if len(syn_values) == 0:
                continue
            
            # Get original values or profile stats
            if original_df is not None:
                if col not in original_df.columns:
                    continue
                orig_values = original_df[col].dropna().values
                
                if len(orig_values) == 0:
                    continue
                
                orig_mean = orig_values.mean()
                orig_std = orig_values.std()
                
                # Histogram overlap
                overlap = self._compute_histogram_overlap(
                    syn_values, orig_values, self.histogram_bins
                )
                metrics["histogram_overlap"][col] = overlap
                
            else:  # Use profile
                field_profile = original_profile.field_profiles.get(col)
                if not field_profile or field_profile.mean is None:
                    continue
                
                orig_mean = field_profile.mean
                orig_std = field_profile.std
                orig_values = None  # Not available from profile
            
            # Mean difference (normalized)
            syn_mean = syn_values.mean()
            mean_diff = abs(syn_mean - orig_mean) / (abs(orig_mean) + 1e-10)
            metrics["mean_difference"][col] = float(mean_diff)
            
            # Variance ratio
            syn_std = syn_values.std()
            var_ratio = (syn_std ** 2) / ((orig_std ** 2) + 1e-10)
            metrics["variance_ratio"][col] = float(var_ratio)
        
        return metrics
    
    def compute_kl_divergence(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute KL divergence for all numeric fields.
        
        Note: Requires original DataFrame (cannot be computed from profile alone).
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            
        Returns:
            Dictionary mapping field names to KL divergence values
        """
        kl_divs = {}
        
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in original_df.columns:
                continue
            
            syn_values = synthetic_df[col].dropna().values
            orig_values = original_df[col].dropna().values
            
            if len(syn_values) == 0 or len(orig_values) == 0:
                continue
            
            kl_div = self._compute_kl_divergence(syn_values, orig_values, self.histogram_bins)
            kl_divs[col] = kl_div
        
        return kl_divs
    
    def compute_wasserstein_distance(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute Wasserstein distance (Earth Mover's Distance) for numeric fields.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            
        Returns:
            Dictionary mapping field names to Wasserstein distances
        """
        distances = {}
        
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in original_df.columns:
                continue
            
            syn_values = synthetic_df[col].dropna().values
            orig_values = original_df[col].dropna().values
            
            if len(syn_values) == 0 or len(orig_values) == 0:
                continue
            
            w_dist = wasserstein_distance(orig_values, syn_values)
            distances[col] = float(w_dist)
        
        return distances
    
    def compute_correlation_difference(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[object] = None
    ) -> float:
        """
        Compute Frobenius norm of correlation matrix difference.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (if available)
            original_profile: DatasetProfile (if original_df not available)
            
        Returns:
            Frobenius norm of difference (0.0 = perfect match)
        """
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return 0.0
        
        # Compute synthetic correlation matrix
        syn_corr = synthetic_df[numeric_cols].corr()
        
        # Get original correlation matrix
        if original_df is not None:
            orig_corr = original_df[numeric_cols].corr()
        elif original_profile and original_profile.correlation_matrix:
            # Reconstruct from profile
            orig_corr_dict = original_profile.correlation_matrix
            orig_corr = pd.DataFrame(orig_corr_dict)
        else:
            raise ValueError("Original correlation matrix not available")
        
        # Align columns
        common_cols = [c for c in numeric_cols if c in orig_corr.columns]
        syn_corr = syn_corr.loc[common_cols, common_cols]
        orig_corr = orig_corr.loc[common_cols, common_cols]
        
        # Frobenius norm of difference
        diff_matrix = syn_corr.values - orig_corr.values
        frobenius_norm = float(np.linalg.norm(diff_matrix, 'fro'))
        
        return frobenius_norm
    
    def compute_drift_metrics(
        self,
        synthetic_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute drift metrics (PSI, KS test).
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame
            
        Returns:
            Dictionary with drift metrics and classification
        """
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        psi_values = {}
        ks_statistics = {}
        
        for col in numeric_cols:
            if col not in original_df.columns:
                continue
            
            syn_values = synthetic_df[col].dropna().values
            orig_values = original_df[col].dropna().values
            
            if len(syn_values) == 0 or len(orig_values) == 0:
                continue
            
            # PSI
            psi = self._compute_psi(syn_values, orig_values)
            psi_values[col] = psi
            
            # KS test
            ks_stat, ks_pval = ks_2samp(orig_values, syn_values)
            ks_statistics[col] = {"statistic": float(ks_stat), "p_value": float(ks_pval)}
        
        # Overall drift classification
        avg_psi = np.mean(list(psi_values.values())) if psi_values else 0.0
        
        if avg_psi < self.drift_threshold_low:
            drift_class = "low"
        elif avg_psi < self.drift_threshold_high:
            drift_class = "medium"
        else:
            drift_class = "high"
        
        return {
            "psi_by_field": psi_values,
            "ks_statistics": ks_statistics,
            "average_psi": float(avg_psi),
            "drift_classification": drift_class
        }
    
    def compute_all(
        self,
        synthetic_df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        original_profile: Optional[object] = None
    ) -> Dict[str, Any]:
        """
        Compute all statistical fidelity metrics.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_df: Original DataFrame (recommended)
            original_profile: DatasetProfile (if original_df unavailable)
            
        Returns:
            Comprehensive dictionary of all metrics
        """
        logger.info("Computing statistical fidelity metrics...")
        
        # Distribution metrics (works with profile)
        dist_metrics = self.compute_distribution_metrics(
            synthetic_df, original_df, original_profile
        )
        
        result = {
            "mean_difference": dist_metrics["mean_difference"],
            "variance_ratio": dist_metrics["variance_ratio"],
            "histogram_overlap": dist_metrics.get("histogram_overlap", {})
        }
        
        # Metrics requiring original DataFrame
        if original_df is not None:
            result["kl_divergence"] = self.compute_kl_divergence(synthetic_df, original_df)
            result["wasserstein_distance"] = self.compute_wasserstein_distance(synthetic_df, original_df)
            drift_metrics = self.compute_drift_metrics(synthetic_df, original_df)
            result["drift_metrics"] = drift_metrics
            result["drift_classification"] = drift_metrics["drift_classification"]
        else:
            result["drift_classification"] = "unknown"
        
        # Correlation (works with profile)
        try:
            result["correlation_frobenius_norm"] = self.compute_correlation_difference(
                synthetic_df, original_df, original_profile
            )
        except Exception as e:
            logger.warning(f"Could not compute correlation difference: {e}")
            result["correlation_frobenius_norm"] = None
        
        logger.info("Statistical fidelity metrics computed successfully")
        return result
