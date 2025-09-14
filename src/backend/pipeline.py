# #!/usr/bin/env python3
# """
# Complete Behavioral Risk Pipeline - Production Ready

# Clean, working pipeline that creates balanced targets and avoids data leakage.
# Driver-level aggregation with synthetic behavioral risk targets.
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Any
# from pathlib import Path
# import logging
# from dataclasses import dataclass
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# @dataclass
# class PipelineConfig:
#     """Pipeline configuration"""
#     test_size: float = 0.25
#     random_state: int = 42
#     min_trips_per_driver: int = 5
    
#     # Risk thresholds
#     hard_accel_threshold: float = 2.0
#     hard_brake_threshold: float = -2.5
#     speed_multiplier: float = 1.2
    
#     # Target creation
#     risk_percentile: float = 0.3  # Top 30% are high risk

# class BehavioralRiskPipeline:
#     """
#     Complete behavioral risk pipeline - driver level aggregation
#     """
    
#     def __init__(self, config: PipelineConfig = None):
#         self.config = config or PipelineConfig()
#         self.scalers = {}
#         self.encoders = {}
#         self.feature_columns = []
        
#     def load_data(self, data_path: str = "./data/") -> Dict[str, pd.DataFrame]:
#         """Load all CSV files"""
#         logger.info("Loading data...")
        
#         data_path = Path(data_path)
#         tables = {}
        
#         file_mapping = {
#             'trips.csv': 'trips',
#             'driver_profile.csv': 'driver_profile',
#             'vehicle_assignments.csv': 'vehicle_assignments', 
#             'vehicle_info.csv': 'vehicle_info',
#             'trip_labels.csv': 'trip_labels'
#         }
        
#         for filename, table_name in file_mapping.items():
#             filepath = data_path / filename
#             if filepath.exists():
#                 tables[table_name] = pd.read_csv(filepath)
#                 logger.info(f"Loaded {table_name}: {tables[table_name].shape}")
        
#         return tables
    
#     def clean_data(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
#         """Clean all tables"""
#         logger.info("Cleaning data...")
#         cleaned = {}
        
#         for name, df in tables.items():
#             df_clean = df.drop_duplicates()
            
#             if name == 'trips':
#                 # Remove rows with missing critical data
#                 df_clean = df_clean.dropna(subset=['user_id', 'trip_id'])
                
#                 # Clean speed
#                 if 'speed' in df_clean.columns:
#                     df_clean = df_clean.dropna(subset=['speed'])
#                     df_clean = df_clean[df_clean['speed'] >= 0]
#                     # Cap extreme speeds at 99th percentile
#                     speed_cap = df_clean['speed'].quantile(0.99)
#                     df_clean['speed'] = df_clean['speed'].clip(upper=speed_cap)
                
#                 # Clean acceleration
#                 if 'accel' in df_clean.columns:
#                     accel_lower = df_clean['accel'].quantile(0.01)
#                     accel_upper = df_clean['accel'].quantile(0.99)
#                     df_clean['accel'] = df_clean['accel'].clip(lower=accel_lower, upper=accel_upper)
            
#             elif name == 'driver_profile':
#                 # Clean age
#                 if 'age' in df_clean.columns:
#                     df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
#                     df_clean = df_clean[(df_clean['age'] >= 16) & (df_clean['age'] <= 100)]
                
#                 # Clean mileage
#                 if 'annual_mileage_declared' in df_clean.columns:
#                     median_mileage = df_clean['annual_mileage_declared'].median()
#                     df_clean['annual_mileage_declared'] = df_clean['annual_mileage_declared'].fillna(median_mileage)
#                     df_clean['annual_mileage_declared'] = df_clean['annual_mileage_declared'].clip(upper=100000)
            
#             elif name == 'vehicle_info':
#                 # Clean vehicle data
#                 numeric_cols = ['year', 'airbags', 'sensors', 'safety_features_count']
#                 for col in numeric_cols:
#                     if col in df_clean.columns:
#                         df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
#                 if 'year' in df_clean.columns:
#                     current_year = pd.Timestamp.now().year
#                     df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= current_year + 1)]
            
#             cleaned[name] = df_clean
            
#         return cleaned
    
#     def create_trip_features(self, trips_df: pd.DataFrame) -> pd.DataFrame:
#         """Create behavioral features at trip level"""
#         df = trips_df.copy()
        
#         # Speed features
#         if 'speed' in df.columns and 'limit' in df.columns:
#             df['speed_over_limit'] = (df['speed'] > df['limit']).astype(int)
#             df['excessive_speed'] = (df['speed'] > df['limit'] * self.config.speed_multiplier).astype(int)
        
#         # Acceleration features
#         if 'accel' in df.columns:
#             df['hard_accel'] = (df['accel'] > self.config.hard_accel_threshold).astype(int)
#             df['hard_brake'] = (df['accel'] < self.config.hard_brake_threshold).astype(int)
        
#         # Temporal features
#         if 'hour' in df.columns:
#             df['night_driving'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
#             df['rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
#                               ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
        
#         # Weather features
#         if 'rain' in df.columns:
#             df['bad_weather'] = (df['rain'] > 0).astype(int)
        
#         return df
    
#     def aggregate_to_driver_level(self, trips_df: pd.DataFrame) -> pd.DataFrame:
#         """Aggregate trip-level data to driver level"""
#         logger.info("Aggregating to driver level...")
        
#         # Create trip features first
#         trips_with_features = self.create_trip_features(trips_df)
        
#         # Aggregate to driver level
#         agg_dict = {
#             'trip_id': 'nunique',
#             'speed': ['mean', 'std', 'max'] if 'speed' in trips_with_features.columns else [],
#             'accel': ['mean', 'std'] if 'accel' in trips_with_features.columns else [],
#             'hard_accel': ['sum', 'mean'] if 'hard_accel' in trips_with_features.columns else [],
#             'hard_brake': ['sum', 'mean'] if 'hard_brake' in trips_with_features.columns else [],
#             'speed_over_limit': ['sum', 'mean'] if 'speed_over_limit' in trips_with_features.columns else [],
#             'excessive_speed': ['sum', 'mean'] if 'excessive_speed' in trips_with_features.columns else [],
#             'night_driving': 'mean' if 'night_driving' in trips_with_features.columns else [],
#             'rush_hour': 'mean' if 'rush_hour' in trips_with_features.columns else [],
#             'bad_weather': 'mean' if 'bad_weather' in trips_with_features.columns else []
#         }
        
#         # Remove empty aggregations
#         agg_dict = {k: v for k, v in agg_dict.items() if v}
        
#         driver_features = trips_with_features.groupby('user_id').agg(agg_dict).round(4)
        
#         # Flatten column names
#         driver_features.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) and col[1] else col[0] 
#                                   for col in driver_features.columns]
        
#         driver_features = driver_features.reset_index()
        
#         # Create rate features
#         if 'trip_id_nunique' in driver_features.columns:
#             total_trips = driver_features['trip_id_nunique']
            
#             for event_type in ['hard_accel', 'hard_brake', 'speed_over_limit', 'excessive_speed']:
#                 sum_col = f'{event_type}_sum'
#                 if sum_col in driver_features.columns:
#                     driver_features[f'{event_type}_rate'] = driver_features[sum_col] / total_trips
        
#         return driver_features
    
#     def merge_all_data(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
#         """Merge all data sources"""
#         logger.info("Merging all data...")
        
#         # Start with driver-level aggregated features
#         driver_features = self.aggregate_to_driver_level(tables['trips'])
        
#         # Merge driver profile
#         if 'driver_profile' in tables:
#             driver_features = driver_features.merge(
#                 tables['driver_profile'], on='user_id', how='left'
#             )
        
#         # Merge vehicle data
#         if 'vehicle_assignments' in tables:
#             driver_features = driver_features.merge(
#                 tables['vehicle_assignments'], on='user_id', how='left'
#             )
        
#         if 'vehicle_info' in tables and 'vehicle_id' in driver_features.columns:
#             driver_features = driver_features.merge(
#                 tables['vehicle_info'], on='vehicle_id', how='left'
#             )
        
#         logger.info(f"Merged dataset shape: {driver_features.shape}")
#         return driver_features
    
#     def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
#         """
#         Create balanced target variable from behavioral patterns
#         Top X% of risky drivers = 1, rest = 0
#         """
#         logger.info("Creating target variable from behavioral risk...")
        
#         # Identify risk indicators
#         risk_features = []
        
#         # Rate-based features (preferred)
#         rate_features = [col for col in df.columns if col.endswith('_rate')]
#         risk_features.extend(rate_features)
        
#         # Mean-based features for temporal patterns
#         pattern_features = [col for col in df.columns if col in [
#             'night_driving_mean', 'rush_hour_mean', 'bad_weather_mean'
#         ]]
#         risk_features.extend(pattern_features)
        
#         # Speed-based features
#         speed_features = [col for col in df.columns if col in [
#             'speed_mean', 'speed_max', 'speed_std'
#         ]]
#         risk_features.extend(speed_features)
        
#         if not risk_features:
#             raise ValueError("No behavioral risk features found for target creation")
        
#         logger.info(f"Using {len(risk_features)} risk features: {risk_features}")
        
#         # Normalize and combine risk features
#         risk_df = df[risk_features].fillna(0)
        
#         # Standardize each feature to 0-1 scale
#         risk_normalized = pd.DataFrame()
#         for col in risk_features:
#             col_min = risk_df[col].min()
#             col_max = risk_df[col].max()
#             if col_max > col_min:
#                 risk_normalized[col] = (risk_df[col] - col_min) / (col_max - col_min)
#             else:
#                 risk_normalized[col] = 0
        
#         # Calculate composite risk score
#         risk_score = risk_normalized.mean(axis=1)
        
#         # Create binary target: top X% are high risk
#         risk_threshold = risk_score.quantile(1 - self.config.risk_percentile)
#         target = (risk_score >= risk_threshold).astype(int)
        
#         logger.info(f"Target distribution: {target.value_counts().to_dict()}")
#         logger.info(f"Risk threshold: {risk_threshold:.4f}")
        
#         return target
    
#     def engineer_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Create additional derived features"""
        
#         # Vehicle age
#         if 'year' in df.columns:
#             current_year = pd.Timestamp.now().year
#             df['vehicle_age'] = current_year - df['year']
        
#         # Safety score
#         safety_cols = ['airbags', 'sensors', 'safety_features_count']
#         available_safety = [col for col in safety_cols if col in df.columns]
#         if available_safety:
#             df['total_safety_features'] = df[available_safety].sum(axis=1)
        
#         # Experience level
#         if 'age' in df.columns:
#             df['young_driver'] = (df['age'] < 25).astype(int)
#             df['senior_driver'] = (df['age'] > 65).astype(int)
        
#         return df
    
#     def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#         """Prepare final features and target"""
#         logger.info("Preparing features and target...")
        
#         # Filter by minimum trips
#         if 'trip_id_nunique' in df.columns:
#             initial_drivers = len(df)
#             df = df[df['trip_id_nunique'] >= self.config.min_trips_per_driver]
#             logger.info(f"Filtered from {initial_drivers} to {len(df)} drivers with >= {self.config.min_trips_per_driver} trips")
        
#         # Create target
#         target = self.create_target_variable(df)
        
#         # Engineer additional features
#         df = self.engineer_additional_features(df)
        
#         # Select features for modeling
#         exclude_cols = [
#             'user_id', 'vehicle_id', 'policy_id', 'trip_id_nunique'
#         ]
        
#         feature_cols = [col for col in df.columns if col not in exclude_cols]
        
#         # Handle categorical variables
#         categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
        
#         for col in categorical_cols:
#             # Skip high cardinality categoricals
#             if df[col].nunique() > 10:
#                 feature_cols.remove(col)
#                 continue
            
#             # Handle missing values and encode
#             df[col] = df[col].fillna('unknown')
            
#             if col not in self.encoders:
#                 self.encoders[col] = LabelEncoder()
#                 df[col] = self.encoders[col].fit_transform(df[col])
        
#         # Final feature matrix
#         features = df[feature_cols].fillna(0)
#         self.feature_columns = feature_cols
        
#         logger.info(f"Final features shape: {features.shape}")
#         logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        
#         return features, target
    
#     def run_pipeline(self, data_path: str = "./data/") -> Dict[str, Any]:
#         """
#         Run the complete pipeline
        
#         Returns:
#             Dictionary with processed data ready for ML
#         """
#         logger.info("Starting behavioral risk pipeline...")
        
#         try:
#             # Load and clean data
#             tables = self.load_data(data_path)
#             cleaned_tables = self.clean_data(tables)
            
#             # Merge all data
#             merged_df = self.merge_all_data(cleaned_tables)
            
#             # Prepare features and target
#             X, y = self.prepare_features_and_target(merged_df)
            
#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y,
#                 test_size=self.config.test_size,
#                 random_state=self.config.random_state,
#                 stratify=y
#             )
            
#             # Scale features
#             scaler = StandardScaler()
#             numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            
#             X_train_scaled = X_train.copy()
#             X_test_scaled = X_test.copy()
            
#             X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#             X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
            
#             self.scalers['features'] = scaler
            
#             pipeline_output = {
#                 'X_train': X_train_scaled,
#                 'X_test': X_test_scaled,
#                 'y_train': y_train,
#                 'y_test': y_test,
#                 'feature_columns': self.feature_columns,
#                 'scalers': self.scalers,
#                 'encoders': self.encoders,
#                 'driver_data': merged_df
#             }
            
#             logger.info("Pipeline completed successfully!")
#             logger.info(f"Training set: {X_train_scaled.shape}")
#             logger.info(f"Test set: {X_test_scaled.shape}")
#             logger.info(f"Features: {len(self.feature_columns)}")
#             logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
            
#             return pipeline_output
            
#         except Exception as e:
#             logger.error(f"Pipeline failed: {str(e)}")
#             raise

# # Example usage
# if __name__ == "__main__":
#     # Configure pipeline
#     config = PipelineConfig(
#         test_size=0.25,
#         min_trips_per_driver=5,
#         risk_percentile=0.3  # Top 30% are high risk
#     )
    
#     # Run pipeline
#     pipeline = BehavioralRiskPipeline(config)
#     results = pipeline.run_pipeline("./data/")
    
#     print("\nPipeline Results:")
#     print(f"Training samples: {len(results['X_train'])}")
#     print(f"Test samples: {len(results['X_test'])}")
#     print(f"Features: {len(results['feature_columns'])}")
#     print(f"Target distribution: {results['y_train'].value_counts().to_dict()}")
# src/pipeline.py
"""
Robust Driver-Level Telematics Pipeline (Leak-Safe, Claims-First)

- Aggregates trip data to driver-level features (no trip-level modeling).
- Primary target: claims-derived (frequency/severity) with train-only thresholding.
- Degeneracy guards: if claims target collapses (all 1s/0s), switch to behavioral synthetic label.
- No leakage: label threshold, encoders, and scaler are all fit on TRAIN ONLY.
- Optional joins: driver_profile, vehicle_assignments + vehicle_info, area_context (safe if absent).

Outputs:
    {
      'X_train','X_test','y_train','y_test',
      'driver_features_raw',        # before encoding/scaling
      'feature_columns',            # post-encoding columns
      'label_source',               # 'claims' or 'behavioral_synthetic'
      'risk_threshold',             # threshold on train for binarization
      'risk_positive_rate_train',   # achieved positive share on train
      'metadata'                    # extra info for dashboards
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------- logging ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("driver_pipeline")

# ---------------------------- config ----------------------------
@dataclass
class PipelineConfig:
    data_dir: str = "./data/"
    test_size: float = 0.25
    random_state: int = 42
    min_trips_per_driver: int = 5

    # thresholds for event detection
    hard_accel_threshold: float = 2.5     # m/s^2
    hard_brake_threshold: float = -3.0    # m/s^2
    excessive_speed_multiplier: float = 1.25  # > 125% of limit

    # target design
    desired_positive_share: float = 0.30          # ~30% positives on TRAIN
    fallback_positive_grid: Tuple[float, ...] = (0.35, 0.25, 0.20, 0.15, 0.10, 0.40, 0.50)
    min_pos: int = 5                                # minimal minority safeguards
    min_neg: int = 5

    # categorical handling
    max_cardinality_for_onehot: int = 16
    round_context_decimals: int = 2               # for optional area_context spatial binning


# ---------------------------- helpers ----------------------------
def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _minmax_scale_train_only(
    df_all: pd.DataFrame, cols: List[str], fit_index: pd.Index
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Return normalized df (0-1 per col) using min/max computed on fit_index only."""
    norm = pd.DataFrame(index=df_all.index)
    stats: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        series = df_all[c].astype(float).fillna(0)
        cmin = float(series.loc[fit_index].min())
        cmax = float(series.loc[fit_index].max())
        stats[c] = (cmin, cmax)
        if cmax > cmin:
            norm[c] = (series - cmin) / (cmax - cmin)
        else:
            norm[c] = 0.0
    return norm, stats


def _choose_threshold_with_guards(
    train_scores: pd.Series,
    desired_share: float,
    grid: Tuple[float, ...],
    min_pos: int,
    min_neg: int,
) -> Tuple[float, float]:
    """Pick train-only threshold to avoid degenerate classes."""
    n = len(train_scores)
    for share in (desired_share,) + tuple(grid):
        thr = float(train_scores.quantile(1.0 - share))
        y = (train_scores >= thr).astype(int)
        pos, neg = int(y.sum()), int(n - y.sum())
        if pos >= min_pos and neg >= min_neg:
            return thr, float(pos / n)
    # last-resort: median split
    thr = float(train_scores.quantile(0.5))
    y = (train_scores >= thr).astype(int)
    return thr, float(y.mean())


# ---------------------------- pipeline ----------------------------
class RobustDriverPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.scaler: Optional[StandardScaler] = None
        self.train_dummy_cols: Optional[pd.Index] = None
        self.feature_columns: List[str] = []
        self.label_source: str = "claims"  # or 'behavioral_synthetic'
        self.risk_threshold_: Optional[float] = None
        self.risk_positive_rate_train_: Optional[float] = None

    # -------- IO + cleaning --------
    def _load_all(self) -> Dict[str, pd.DataFrame]:
        d = Path(self.cfg.data_dir)
        tables: Dict[str, pd.DataFrame] = {}
        # required
        trips = _read_csv_if_exists(d / "trips.csv")
        if trips is None or trips.empty:
            raise FileNotFoundError("data/trips.csv is required and must be non-empty.")
        tables["trips"] = trips
        log.info(f"Loaded trips: {trips.shape}")

        # optional
        for name in ("trip_labels", "driver_profile", "vehicle_assignments", "vehicle_info", "area_context"):
            df = _read_csv_if_exists(d / f"{name}.csv")
            if df is not None and not df.empty:
                tables[name] = df
                log.info(f"Loaded {name}: {df.shape}")

        return tables

    def _clean_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df = df.dropna(subset=["user_id", "trip_id"])
        # speed
        if "speed" in df.columns:
            df = df[df["speed"] >= 0]
            df["speed"] = df["speed"].clip(upper=df["speed"].quantile(0.99))
        # accel
        if "accel" in df.columns:
            lo, hi = df["accel"].quantile([0.01, 0.99])
            df["accel"] = df["accel"].clip(lower=float(lo), upper=float(hi))
        # basic geo sanity
        if "lat" in df.columns and "lon" in df.columns:
            df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
        return df

    def _clean_trip_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        # coerce to numeric
        for c in ("claim", "severity"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        # keep only necessary cols if present
        keep = [c for c in ("user_id", "trip_id", "claim", "severity", "timestamp") if c in df.columns]
        return df[keep] if keep else df

    def _clean_driver_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df["age"] = df["age"].fillna(df["age"].median())
            df = df[df["age"].between(16, 100)]
        if "annual_mileage_declared" in df.columns:
            df["annual_mileage_declared"] = pd.to_numeric(df["annual_mileage_declared"], errors="coerce")
            df["annual_mileage_declared"] = df["annual_mileage_declared"].fillna(df["annual_mileage_declared"].median())
            df["annual_mileage_declared"] = df["annual_mileage_declared"].clip(upper=100_000)
        return df

    def _clean_vehicle_info(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        for c in ("year", "airbags", "sensors", "safety_features_count"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
                df[c] = df[c].fillna(df[c].median())
        if "year" in df.columns:
            yr = pd.Timestamp.now().year
            df = df[df["year"].between(1990, yr + 1)]
        return df

    def _clean_area_context(self, df: pd.DataFrame) -> pd.DataFrame:
        # keep it permissive — just ensure lat/lon present if we want to use it
        needed = {"lat", "lon"}
        if not needed.issubset(set(df.columns)):
            return pd.DataFrame()
        # any provided per-cell risk fields are ok (crime_rate, crash_index, risk_index, etc.)
        return df.copy()

    def _flag_trip_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event flags per row; later aggregated per driver."""
        out = df.copy()
        if {"limit", "speed"}.issubset(out.columns):
            out["speed_over_limit"] = (out["speed"] > out["limit"]).astype(int)
            out["excessive_speed"] = (out["speed"] > out["limit"] * self.cfg.excessive_speed_multiplier).astype(int)
        if "accel" in out.columns:
            out["hard_accel"] = (out["accel"] > self.cfg.hard_accel_threshold).astype(int)
            out["hard_brake"] = (out["accel"] < self.cfg.hard_brake_threshold).astype(int)
        if "hour" in out.columns:
            out["night_driving"] = ((out["hour"] >= 22) | (out["hour"] <= 5)).astype(int)
            out["rush_hour"] = (((out["hour"] >= 7) & (out["hour"] <= 9)) |
                                ((out["hour"] >= 17) & (out["hour"] <= 19))).astype(int)
        if "rain" in out.columns:
            out["bad_weather"] = (out["rain"] > 0).astype(int)
        return out

    # -------- feature engineering (driver-level) --------
    def _aggregate_driver(self, trips: pd.DataFrame) -> pd.DataFrame:
        t = self._flag_trip_events(trips)
        agg: Dict[str, Any] = {
            "trip_id": "nunique",
            "speed": ["mean", "std", "max"] if "speed" in t.columns else [],
            "accel": ["mean", "std"] if "accel" in t.columns else [],
            "speed_over_limit": ["mean", "sum"] if "speed_over_limit" in t.columns else [],
            "excessive_speed": ["mean", "sum"] if "excessive_speed" in t.columns else [],
            "hard_accel": ["mean", "sum"] if "hard_accel" in t.columns else [],
            "hard_brake": ["mean", "sum"] if "hard_brake" in t.columns else [],
            "night_driving": "mean" if "night_driving" in t.columns else [],
            "rush_hour": "mean" if "rush_hour" in t.columns else [],
            "bad_weather": "mean" if "bad_weather" in t.columns else [],
        }
        agg = {k: v for k, v in agg.items() if v}

        drv = t.groupby("user_id").agg(agg)

        # flatten
        cols = []
        for c in drv.columns:
            if isinstance(c, tuple):
                base, op = c
                cols.append(f"{base}_{op}" if op else base)
            else:
                cols.append(c)
        drv.columns = cols
        drv = drv.reset_index().rename(columns={"trip_id_nunique": "n_trips"})

        # event rates per trip
        if "n_trips" in drv.columns:
            ntrips = drv["n_trips"].replace(0, np.nan)
            for ev in ("hard_accel", "hard_brake", "speed_over_limit", "excessive_speed"):
                s = f"{ev}_sum"
                if s in drv.columns:
                    drv[f"{ev}_rate"] = (drv[s] / ntrips).fillna(0)
            drv["events_per_trip"] = (
                drv.filter(regex="_sum$").sum(axis=1) / ntrips
            ).fillna(0)

        return drv

    def _join_dimensions(
        self,
        driver_df: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        df = driver_df.copy()

        # driver_profile
        if "driver_profile" in tables:
            prof = self._clean_driver_profile(tables["driver_profile"])
            df = df.merge(prof, on="user_id", how="left")

        # vehicle joins
        if "vehicle_assignments" in tables:
            va = tables["vehicle_assignments"].drop_duplicates()
            df = df.merge(va, on="user_id", how="left")
        if "vehicle_info" in tables and "vehicle_id" in df.columns:
            vinf = self._clean_vehicle_info(tables["vehicle_info"])
            df = df.merge(vinf, on="vehicle_id", how="left")
            # vehicle age, safety
            if "year" in df.columns:
                df["vehicle_age"] = pd.Timestamp.now().year - df["year"]
            safety_cols = [c for c in ("airbags", "sensors", "safety_features_count") if c in df.columns]
            if safety_cols:
                df["total_safety_features"] = df[safety_cols].sum(axis=1)

        # optional area_context — approximate spatial binning if possible
        if "area_context" in tables and {"lat", "lon"}.issubset(tables["area_context"].columns):
            ac = self._clean_area_context(tables["area_context"])
            if not ac.empty and {"lat", "lon"}.issubset(ac.columns):
                # round lat/lon in trips, map mean context per user
                if {"lat", "lon"}.issubset(tables["trips"].columns):
                    trips = tables["trips"][["user_id", "lat", "lon"]].copy()
                    r = self.cfg.round_context_decimals
                    trips["lat_r"] = trips["lat"].round(r)
                    trips["lon_r"] = trips["lon"].round(r)
                    ac["lat_r"] = ac["lat"].round(r)
                    ac["lon_r"] = ac["lon"].round(r)

                    # pick any numeric context cols besides lat/lon
                    cand = ac.select_dtypes(include=[np.number]).columns.tolist()
                    cand = [c for c in cand if c not in ("lat", "lon")]
                    if cand:
                        grid_mean = ac.groupby(["lat_r", "lon_r"])[cand].mean().reset_index()
                        joined = trips.merge(grid_mean, on=["lat_r", "lon_r"], how="left")
                        ctx = joined.groupby("user_id")[cand].mean().reset_index()
                        df = df.merge(ctx, on="user_id", how="left")

        return df

    # -------- target building (claims-first, leak-safe) --------
    def _build_claim_targets(
        self, driver_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute claims metrics per driver if trip_labels provided, otherwise empty."""
        if "trip_labels" not in tables:
            return pd.DataFrame({"user_id": driver_df["user_id"]})

        tl = self._clean_trip_labels(tables["trip_labels"])
        # We need (user_id, claim, severity); ensure 'claim' is 0/1
        if "claim" in tl.columns:
            tl["claim_bin"] = (tl["claim"].astype(float) > 0).astype(int)
        else:
            tl["claim_bin"] = 0

        claims_per_user = tl.groupby("user_id").agg(
            total_claims=("claim_bin", "sum"),
            total_severity=("severity", "sum") if "severity" in tl.columns else ("claim_bin", "sum"),
        ).reset_index()

        # merge #trips to compute rates
        out = driver_df[["user_id", "n_trips"]].merge(claims_per_user, on="user_id", how="left").fillna(0)
        out["claim_rate"] = np.where(out["n_trips"] > 0, out["total_claims"] / out["n_trips"], 0.0)
        out["severity_per_trip"] = np.where(out["n_trips"] > 0, out["total_severity"] / out["n_trips"], 0.0)
        out["has_any_claim"] = (out["total_claims"] > 0).astype(int)
        return out

    def _make_binary_target_leak_safe(
        self,
        df_all: pd.DataFrame,
        train_idx: pd.Index,
        claims_table: Optional[pd.DataFrame],
    ) -> Tuple[pd.Series, str, float, float]:
        """
        Try claims-derived label first; if degenerate, fall back to behavioral synthetic.
        Returns: (y_all, label_source, threshold, achieved_train_pos_rate)
        """
        # ---- attempt 1: claims-derived ----
        if claims_table is not None and "claim_rate" in claims_table.columns:
            tmp = df_all[["user_id"]].merge(claims_table, on="user_id", how="left").fillna(0)
            # continuous risk = 0.7*claim_rate + 0.3*norm(severity_per_trip)
            cont = tmp[["claim_rate", "severity_per_trip"]].astype(float).fillna(0)
            sev_norm, _ = _minmax_scale_train_only(cont[["severity_per_trip"]], ["severity_per_trip"], fit_index=train_idx)
            risk_score = 0.7 * cont["claim_rate"] + 0.3 * sev_norm["severity_per_trip"]
            train_scores = risk_score.loc[train_idx]
            thr, pos_share = _choose_threshold_with_guards(
                train_scores, self.cfg.desired_positive_share,
                self.cfg.fallback_positive_grid, self.cfg.min_pos, self.cfg.min_neg
            )
            y_all = (risk_score >= thr).astype(int)
            # check degeneracy across full data (should be fine if train was fine)
            if y_all.nunique() == 2:
                return y_all, "claims", float(thr), float(pos_share)

        # ---- attempt 2: behavioral synthetic (rates only) ----
        rate_cols = [c for c in df_all.columns if c.endswith("_rate")]
        rate_cols += [c for c in ("speed_over_limit_mean", "excessive_speed_mean") if c in df_all.columns]
        if not rate_cols:
            # last fallback: any_claim if present, else all zeros (but guard)
            if claims_table is not None and "has_any_claim" in claims_table.columns:
                any_claim = df_all[["user_id"]].merge(claims_table[["user_id", "has_any_claim"]], on="user_id", how="left").fillna(0)["has_any_claim"]
                # If all one/zero, just median-split by events_per_trip if present, else zeros
                if any_claim.nunique() == 2:
                    return any_claim.astype(int), "claims_any", 0.5, float(any_claim.mean())
            # worst case
            y = pd.Series(0, index=df_all.index, dtype=int)
            return y, "none", 0.0, 0.0

        norm_rates, _ = _minmax_scale_train_only(df_all[rate_cols], rate_cols, fit_index=train_idx)
        risk_score = norm_rates.mean(axis=1)
        thr, pos_share = _choose_threshold_with_guards(
            risk_score.loc[train_idx],
            self.cfg.desired_positive_share,
            self.cfg.fallback_positive_grid,
            self.cfg.min_pos,
            self.cfg.min_neg,
        )
        y_all = (risk_score >= thr).astype(int)
        return y_all, "behavioral_synthetic", float(thr), float(pos_share)

    # -------- encoding & scaling (leak-safe) --------
    def _encode_onehot_train_test(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # select low-cardinality categoricals
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if X_train[c].nunique() <= self.cfg.max_cardinality_for_onehot]
        if not cat_cols:
            return X_train, X_test

        train_dum = pd.get_dummies(X_train[cat_cols], prefix=cat_cols, drop_first=False, dtype=np.uint8)
        test_dum = pd.get_dummies(X_test[cat_cols], prefix=cat_cols, drop_first=False, dtype=np.uint8)
        # align to train columns
        test_dum = test_dum.reindex(columns=train_dum.columns, fill_value=0)
        self.train_dummy_cols = train_dum.columns

        Xtr = pd.concat([X_train.drop(columns=cat_cols).reset_index(drop=True), train_dum.reset_index(drop=True)], axis=1)
        Xte = pd.concat([X_test.drop(columns=cat_cols).reset_index(drop=True), test_dum.reset_index(drop=True)], axis=1)
        return Xtr, Xte

    # -------- main entry --------
    def run(self) -> Dict[str, Any]:
        log.info("Running RobustDriverPipeline…")
        tables = self._load_all()

        # clean core tables
        trips = self._clean_trips(tables["trips"])
        tables["trips"] = trips
        if "trip_labels" in tables:
            tables["trip_labels"] = self._clean_trip_labels(tables["trip_labels"])
        if "driver_profile" in tables:
            tables["driver_profile"] = self._clean_driver_profile(tables["driver_profile"])
        if "vehicle_info" in tables:
            tables["vehicle_info"] = self._clean_vehicle_info(tables["vehicle_info"])
        if "area_context" in tables:
            tables["area_context"] = self._clean_area_context(tables["area_context"])

        # aggregate driver features
        drv = self._aggregate_driver(trips)

        # filter by min trips
        before = len(drv)
        drv = drv[drv["n_trips"] >= self.cfg.min_trips_per_driver].copy()
        log.info(f"Drivers kept after min_trips_per_driver({self.cfg.min_trips_per_driver}): {len(drv)} (from {before})")

        # join dims
        drv_all = self._join_dimensions(drv, tables)

        # split drivers FIRST (no label yet)
        idx = drv_all.index
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            shuffle=True,
        )

        # build claims table (if available)
        claims_tbl = self._build_claim_targets(drv_all, tables)

        # make label (leak-safe)
        y_all, label_src, thr, pos_share = self._make_binary_target_leak_safe(drv_all, tr_idx, claims_tbl if not claims_tbl.empty else None)
        self.label_source = label_src
        self.risk_threshold_ = thr
        self.risk_positive_rate_train_ = pos_share

        y_train, y_test = y_all.loc[tr_idx].copy(), y_all.loc[te_idx].copy()

        # build feature matrix (exclude label-driving identifiers and raw counts where appropriate)
        exclude = {
            "user_id", "vehicle_id", "policy_id",
            "n_trips",  # keep out if you prefer not to scale volume directly; remove if you want it
        }
        # NEVER include label artefacts directly
        if not claims_tbl.empty:
            exclude.update({"total_claims", "total_severity", "claim_rate", "severity_per_trip", "has_any_claim"})

        X_all = drv_all.drop(columns=[c for c in exclude if c in drv_all.columns]).copy()

        # split features to align with indices
        X_train_raw, X_test_raw = X_all.loc[tr_idx].copy(), X_all.loc[te_idx].copy()

        # one-hot on train only
        X_train_enc, X_test_enc = self._encode_onehot_train_test(X_train_raw, X_test_raw)

        # scale numeric on train only
        num_cols = X_train_enc.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        X_train_enc[num_cols] = self.scaler.fit_transform(X_train_enc[num_cols])
        X_test_enc[num_cols] = self.scaler.transform(X_test_enc[num_cols])

        self.feature_columns = X_train_enc.columns.tolist()

        # logs
        log.info(f"Train shape: {X_train_enc.shape} | Test shape: {X_test_enc.shape}")
        log.info(f"Label source: {self.label_source} | Train pos share: {self.risk_positive_rate_train_:.3f}")
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            log.warning("Warning: One of the splits ended up single-class. Consider tuning desired_positive_share/grid.")

        # package
        out = {
            "X_train": X_train_enc,
            "X_test": X_test_enc,
            "y_train": y_train,
            "y_test": y_test,
            "driver_features_raw": drv_all,           # for dashboards
            "feature_columns": self.feature_columns,
            "label_source": self.label_source,
            "risk_threshold": self.risk_threshold_,
            "risk_positive_rate_train": self.risk_positive_rate_train_,
            "metadata": {
                "num_drivers": len(drv_all),
                "kept_min_trips": self.cfg.min_trips_per_driver,
                "train_size": float(1.0 - self.cfg.test_size),
                "test_size": float(self.cfg.test_size),
            },
        }
        return out


# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    cfg = PipelineConfig(
        data_dir="./data/",
        test_size=0.25,
        random_state=42,
        min_trips_per_driver=5,
        desired_positive_share=0.30,
        fallback_positive_grid=(0.35, 0.25, 0.20, 0.15, 0.10, 0.40, 0.50))
    pipe = RobustDriverPipeline(cfg)
    res = pipe.run()

    print("\n=== Driver-Level Pipeline Summary ===")
    print(f"Features (train): {res['X_train'].shape} | (test): {res['X_test'].shape}")
    print(f"y_train distribution: {res['y_train'].value_counts().to_dict()}")
    print(f"y_test  distribution: {res['y_test'].value_counts().to_dict()}")
    print(f"Label source: {res['label_source']}")
    print(f"Train pos share: {res['risk_positive_rate_train']:.3f}")
