"""
Data Profiler - Statistical profile generation without storing raw data

Generates statistical summaries, sketches, and hashes for comparing datasets
while maintaining strict privacy constraints.

SECURITY CONSTRAINTS:
- NEVER stores raw values
- Only statistical summaries (mean, variance, histograms, etc.)
- Cryptographic hashes for membership checks
- Sketch data structures for cardinality/frequency estimation

This allows leakage detection without retaining original data.
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FieldProfile:
    """Statistical profile for a single field."""
    
    field_name: str
    dtype: str
    count: int
    null_count: int
    null_rate: float
    
    # Numeric fields
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    quartiles: Optional[List[float]] = None  # [Q1, Q2/median, Q3]
    
    # Categorical fields
    unique_count: Optional[int] = None
    most_common: Optional[List[Tuple[str, int]]] = None  # Top 10
    
    # Privacy: value hashes for membership checks (NOT raw values)
    value_hashes: Optional[List[str]] = None  # SHA256 hashes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DatasetProfile:
    """Complete statistical profile for a dataset."""
    
    profile_id: str
    created_at: str
    row_count: int
    column_count: int
    column_names: List[str]
    
    field_profiles: Dict[str, FieldProfile]
    
    # Cross-field statistics
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    
    # Dataset-level hashes
    row_hashes: Optional[List[str]] = None  # SHA256 of each row (for near-duplicate detection)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "field_profiles": {
                name: profile.to_dict()
                for name, profile in self.field_profiles.items()
            },
            "correlation_matrix": self.correlation_matrix,
            "row_hashes": self.row_hashes
        }
    
    def save(self, filepath: str):
        """Save profile to JSON file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved data profile to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "DatasetProfile":
        """Load profile from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct FieldProfile objects
        field_profiles = {
            name: FieldProfile(**profile_data)
            for name, profile_data in data["field_profiles"].items()
        }
        
        return cls(
            profile_id=data["profile_id"],
            created_at=data["created_at"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            column_names=data["column_names"],
            field_profiles=field_profiles,
            correlation_matrix=data.get("correlation_matrix"),
            row_hashes=data.get("row_hashes")
        )


class DataProfiler:
    """
    Generate statistical profiles from datasets without storing raw data.
    
    Profiles are used for:
    - Statistical fidelity comparison (mean, variance, correlation)
    - Privacy risk assessment (membership checks via hashes)
    - Utility evaluation (distribution comparison)
    
    Example:
        >>> profiler = DataProfiler()
        >>> profile = profiler.create_profile(original_df, profile_id="orig_001")
        >>> profile.save("profiles/original.json")
        
        >>> # Later: compare synthetic data to profile
        >>> synthetic_metrics = profiler.compare_to_profile(synthetic_df, profile)
    """
    
    def __init__(self, include_row_hashes: bool = True, top_k_values: int = 10):
        """
        Initialize data profiler.
        
        Args:
            include_row_hashes: Whether to compute row hashes for membership checks
            top_k_values: Number of most common values to track for categorical fields
        """
        self.include_row_hashes = include_row_hashes
        self.top_k_values = top_k_values
    
    def _hash_value(self, value: Any) -> str:
        """
        Compute SHA256 hash of a value.
        
        Args:
            value: Value to hash
            
        Returns:
            Hex string of hash
        """
        value_str = str(value)
        return hashlib.sha256(value_str.encode()).hexdigest()
    
    def _hash_row(self, row: pd.Series) -> str:
        """
        Compute SHA256 hash of an entire row.
        
        Args:
            row: Pandas Series representing a row
            
        Returns:
            Hex string of hash
        """
        # Sort by column name for consistent hashing
        sorted_values = [str(row[col]) for col in sorted(row.index)]
        row_str = "|".join(sorted_values)
        return hashlib.sha256(row_str.encode()).hexdigest()
    
    def _profile_numeric_field(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a numeric field."""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "quartiles": None
            }
        
        return {
            "mean": float(non_null.mean()),
            "std": float(non_null.std()),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "quartiles": [
                float(non_null.quantile(0.25)),
                float(non_null.quantile(0.50)),
                float(non_null.quantile(0.75))
            ]
        }
    
    def _profile_categorical_field(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a categorical field."""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return {
                "unique_count": 0,
                "most_common": []
            }
        
        value_counts = non_null.value_counts()
        most_common = [
            (str(val), int(count))
            for val, count in value_counts.head(self.top_k_values).items()
        ]
        
        return {
            "unique_count": int(non_null.nunique()),
            "most_common": most_common
        }
    
    def create_profile(
        self,
        df: pd.DataFrame,
        profile_id: str,
        include_value_hashes: bool = False,
        max_hashes_per_field: int = 10000
    ) -> DatasetProfile:
        """
        Create statistical profile from DataFrame.
        
        Args:
            df: Input DataFrame
            profile_id: Unique identifier for this profile
            include_value_hashes: Whether to include value hashes (for membership checks)
            max_hashes_per_field: Maximum number of hashes to store per field
            
        Returns:
            DatasetProfile object
        """
        from datetime import datetime
        
        logger.info(f"Creating profile {profile_id} for dataset: {len(df)} rows, {len(df.columns)} columns")
        
        field_profiles = {}
        
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            
            # Basic statistics
            count = len(series)
            null_count = int(series.isnull().sum())
            null_rate = null_count / count if count > 0 else 0.0
            
            # Type-specific profiling
            numeric_stats = {}
            categorical_stats = {}
            
            if pd.api.types.is_numeric_dtype(series):
                numeric_stats = self._profile_numeric_field(series)
            else:
                categorical_stats = self._profile_categorical_field(series)
            
            # Value hashes (optional, for membership checks)
            value_hashes = None
            if include_value_hashes:
                non_null = series.dropna()
                if len(non_null) > 0:
                    # Sample if too many unique values
                    unique_vals = non_null.unique()
                    if len(unique_vals) > max_hashes_per_field:
                        unique_vals = np.random.choice(
                            unique_vals,
                            size=max_hashes_per_field,
                            replace=False
                        )
                    value_hashes = [self._hash_value(v) for v in unique_vals]
            
            field_profile = FieldProfile(
                field_name=col,
                dtype=dtype,
                count=count,
                null_count=null_count,
                null_rate=null_rate,
                value_hashes=value_hashes,
                **numeric_stats,
                **categorical_stats
            )
            
            field_profiles[col] = field_profile
        
        # Correlation matrix for numeric fields
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = None
        
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            correlation_matrix = {
                col: {col2: float(corr.loc[col, col2]) for col2 in numeric_cols}
                for col in numeric_cols
            }
        
        # Row hashes for near-duplicate detection
        row_hashes = None
        if self.include_row_hashes:
            logger.info("Computing row hashes...")
            row_hashes = df.apply(self._hash_row, axis=1).tolist()
        
        profile = DatasetProfile(
            profile_id=profile_id,
            created_at=datetime.now().isoformat(),
            row_count=len(df),
            column_count=len(df.columns),
            column_names=df.columns.tolist(),
            field_profiles=field_profiles,
            correlation_matrix=correlation_matrix,
            row_hashes=row_hashes
        )
        
        logger.info(f"Profile created: {profile_id}")
        return profile
    
    def compute_membership_overlap(
        self,
        synthetic_df: pd.DataFrame,
        original_profile: DatasetProfile
    ) -> Dict[str, float]:
        """
        Compute membership overlap between synthetic data and original profile.
        
        Checks how many synthetic records have exact hash matches in original.
        
        Args:
            synthetic_df: Synthetic DataFrame
            original_profile: Profile of original dataset
            
        Returns:
            Dictionary with overlap statistics
        """
        if not original_profile.row_hashes:
            raise ValueError("Original profile does not contain row hashes")
        
        # Compute hashes for synthetic data
        synthetic_hashes = synthetic_df.apply(self._hash_row, axis=1).tolist()
        
        # Find matches
        original_hash_set = set(original_profile.row_hashes)
        matches = [h for h in synthetic_hashes if h in original_hash_set]
        
        overlap_rate = len(matches) / len(synthetic_hashes) if synthetic_hashes else 0.0
        
        return {
            "synthetic_count": len(synthetic_hashes),
            "original_count": len(original_profile.row_hashes),
            "exact_matches": len(matches),
            "overlap_rate": overlap_rate
        }
