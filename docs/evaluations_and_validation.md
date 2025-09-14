1) Data quality gates

Null pressure: warn if any required column has >30% null.

Bounds:

speed >= 0, hard cap at 99th pct to kill spikes.

accel winsorize at 1st/99th pct.

lat ∈ [-90, 90], lon ∈ [-180, 180].

Cardinality sanity: user_id unique count > min drivers; trip_id high cardinality.

2) Split hygiene

Verify no user_id overlaps across train/test.

Verify encoders/scaler were fitted only on train.

3) Metric suite

Binary: ROC-AUC (primary), PR-AUC, F1@threshold, calibration curve/Brier.

Severity: R² and MAPE on claimants.

Stability: PSI across top behavioral features train→test to catch drift.

4) Business sanity

Monotonicity checks:

Higher pct_excess_speed → stochastically higher predicted risk (monotone bins).

More events_per_100km → higher risk on average.

Lift chart: top decile captures materially more claims than average.

5) Reproducibility

Persist:

models/*.joblib

preprocessing_*.joblib (scaler + encoders + selected feature list)

model_performance_summary_*.json

Driver-level engineered_features_*.csv for audit.