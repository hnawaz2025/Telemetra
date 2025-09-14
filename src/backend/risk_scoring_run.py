"""
Risk Scoring Runner (resilient)

- Auto-detects pipeline class & config inside pipeline.py
- Calls the correct run method (run_pipeline / run / run_complete_pipeline)
- Normalizes outputs to {X_train, X_test, y_train, y_test, feature_columns, driver_data}
- Trains calibrated expert models and writes behavioral_risk_scores.csv
"""

from __future__ import annotations
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import modeling bits only (no direct pipeline imports here)
from risk_scoring import ModelConfig, RiskScorer, run_behavioral_risk_modeling

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("risk_scoring_run")


# ---------------------------
# Helpers to adapt to pipeline.py
# ---------------------------
def _import_pipeline_module():
    import importlib
    return importlib.import_module("pipeline")


def _pick_class(mod, candidate_names):
    for name in candidate_names:
        if hasattr(mod, name):
            return getattr(mod, name), name
    return None, None


def _filter_kwargs_for_cls(cls, kwargs):
    """Filter kwargs to those accepted by cls.__init__."""
    try:
        sig = inspect.signature(cls.__init__)
        valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return valid
    except (TypeError, ValueError):
        return {}  # fallback to no kwargs


def _call_pipeline_run(pipeline_obj, data_path: str) -> Dict[str, Any]:
    """
    Try the common run entrypoints in a safe order.
    - run_pipeline(data_path)
    - run(data_path)
    - run()
    - run_complete_pipeline(data_path)
    """
    candidates = [
        ("run_pipeline", {"data_path": data_path}),
        ("run_pipeline", {"data_dir": data_path}),
        ("run", {"data_path": data_path}),
        ("run", {"data_dir": data_path}),
        ("run", {}),
        ("run_complete_pipeline", {"data_path": data_path}),
        ("run_complete_pipeline", {"data_dir": data_path}),
    ]
    for meth, kwargs in candidates:
        if hasattr(pipeline_obj, meth):
            fn = getattr(pipeline_obj, meth)
            try:
                sig = inspect.signature(fn)
                call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return fn(**call_kwargs)
            except Exception as e:
                logger.debug(f"Tried {meth} with {kwargs} -> {e}")
                continue
    raise RuntimeError("Could not find a usable run method in pipeline object.")


def _normalize_outputs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure outputs have the keys expected by risk_scoring:
    X_train, X_test, y_train, y_test, feature_columns, driver_data
    """
    required = {"X_train", "X_test", "y_train", "y_test"}
    if required.issubset(set(raw.keys())):
        # keep as-is; add fallbacks for feature_columns/driver_data
        if "feature_columns" not in raw and "X_train" in raw:
            raw["feature_columns"] = list(getattr(raw["X_train"], "columns", []))
        if "driver_data" not in raw:
            raw["driver_data"] = None
        return raw

    # If the pipeline returns something more "full-stack" (features/targets only),
    # we cannot proceed. Tell user what we found.
    keys = list(raw.keys())
    raise ValueError(
        "Pipeline run did not return the expected train/test matrices.\n"
        f"Expected keys: {sorted(required)}\n"
        f"Got keys: {sorted(keys)}"
    )


def _attach_user_ids(risk_df: pd.DataFrame, X_index, driver_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if driver_data is None or "user_id" not in driver_data.columns:
        return risk_df
    id_map = driver_data[["user_id"]].copy()
    id_map.index = driver_data.index
    id_map = id_map.loc[X_index]
    out = risk_df.copy()
    out.insert(0, "user_id", id_map["user_id"].values)
    return out


def _best_feature_importance(risk_models) -> pd.DataFrame:
    best = risk_models.best_
    if not best:
        return pd.DataFrame()
    model = best["model"]
    base = getattr(model, "base_estimator_", None) or model
    names = risk_models.selected_features_

    if hasattr(base, "feature_importances_"):
        imp = np.asarray(base.feature_importances_)
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    if hasattr(base, "coef_"):
        coef = base.coef_[0] if base.coef_.ndim > 1 else base.coef_
        imp = np.abs(coef)
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    return pd.DataFrame()


# ---------------------------
# Main
# ---------------------------
def main():
    print("=" * 86)
    print("BEHAVIORAL RISK SCORING SYSTEM  —  Resilient Runner")
    print("=" * 86)

    DATA_PATH = "./data/"
    OUT_DIR = Path("./data/risk_socres/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Locate pipeline classes
        pipe_mod = _import_pipeline_module()

        pipeline_cls, pipeline_cls_name = _pick_class(
            pipe_mod,
            ["BehavioralRiskPipeline", "RobustDriverPipeline", "InsuranceOutcomePipeline"]
        )
        if pipeline_cls is None:
            raise ImportError(
                "No supported pipeline class found in pipeline.py. "
                "Expected one of: BehavioralRiskPipeline, RobustDriverPipeline, InsuranceOutcomePipeline."
            )

        config_cls, config_cls_name = _pick_class(
            pipe_mod,
            ["PipelineConfig", "OutcomeConfig", "DriverConfig", "RobustConfig"]
        )
        if config_cls is None:
            # If there's no config class, we’ll try to init the pipeline with no args.
            logger.info("No config class found. Will instantiate pipeline without a config object.")
            cfg = None
        else:
            # Build a config with common knobs, filtering to whatever the class supports
            common_cfg = dict(
                test_size=0.25,
                min_trips_per_driver=5,
                risk_positive_share=0.30,  # ignored by claim-based pipelines
                data_dir=DATA_PATH,        # for pipelines that read from cfg.data_dir
            )
            cfg_kwargs = _filter_kwargs_for_cls(config_cls, common_cfg)
            cfg = config_cls(**cfg_kwargs)

        # 2) Instantiate pipeline
        #    try pipeline_cls(cfg), fall back to pipeline_cls() if constructor doesn’t accept it
        try:
            if cfg is not None:
                pipe = pipeline_cls(cfg)
            else:
                pipe = pipeline_cls()
        except TypeError:
            pipe = pipeline_cls()

        # 3) Run it and normalize outputs
        raw = _call_pipeline_run(pipe, DATA_PATH)
        pipe_results = _normalize_outputs(raw)

        X_train = pipe_results["X_train"]
        X_test = pipe_results["X_test"]
        y_train = pipe_results["y_train"]
        y_test = pipe_results["y_test"]

        print(f"✓ Pipeline: {pipeline_cls_name} | Train: {X_train.shape}  Test: {X_test.shape}")
        if hasattr(y_train, "value_counts"):
            print(f"  Train target: {y_train.value_counts().to_dict()}")
        if hasattr(y_test, "value_counts"):
            print(f"  Test  target: {y_test.value_counts().to_dict()}")

        # 4) Train expert models
        print("\nSTEP 2: Training Expert Risk Models…")
        model_cfg = ModelConfig(
            feature_selection=True,
            max_features=30,
            calibrate_probabilities=True,
            calibration_method="isotonic",
            cv_folds=5,
            threshold_objective="f1",
            cost_fp=1.0,
            cost_fn=5.0,
        )
        modeling = run_behavioral_risk_modeling(pipe_results, model_cfg)
        risk_models = modeling["risk_models_instance"]
        comparison_df = modeling["model_comparison"]
        best = modeling["best_model"]

        print("✓ Models trained.\n")
        print("Model Performance (holdout):")
        for mname, metrics in risk_models.model_scores_.items():
            print(f"  {mname:18s} | AUC={metrics.get('roc_auc', float('nan')):.3f}  "
                  f"AP={metrics.get('avg_precision', float('nan')):.3f}  "
                  f"F1={metrics.get('f1', 0.0):.3f}  Acc={metrics.get('accuracy', 0.0):.3f}")
        if best:
            print(f"\nBest Model: {best['name']}  |  Tuned threshold={risk_models.tuned_threshold_:.3f}")

        # 5) Score everyone (train+test)
        print("\nSTEP 3: Generating Risk Scores…")
        X_all = pd.concat([X_train, X_test], axis=0)  # keep indices for user_id mapping
        y_all = pd.concat([y_train, y_test], axis=0)

        scorer = RiskScorer(risk_models)
        risk_scores = scorer.calculate_risk_score(X_all)
        risk_scores["actual_risk"] = y_all.values
        risk_scores = _attach_user_ids(risk_scores, X_all.index, pipe_results.get("driver_data"))

        # Pricing tiers
        risk_scores["pricing_tier"] = pd.cut(
            risk_scores["risk_score"],
            bins=[-0.1, 20, 40, 60, 80, 100],
            labels=[
                "Tier 1 - Preferred Plus", "Tier 2 - Preferred",
                "Tier 3 - Standard", "Tier 4 - Standard Plus",
                "Tier 5 - High Risk"
            ]
        )
        print(f"✓ Scored {len(risk_scores)} drivers.")

        # 6) Snapshot + save
        print("\nSTEP 4: Results Snapshot")
        cat_counts = risk_scores["risk_category"].value_counts()
        for cat, cnt in cat_counts.items():
            print(f"  {cat:10s}: {cnt:4d} ({cnt/len(risk_scores)*100:5.1f}%)")
        acc = (risk_scores["risk_prediction"] == risk_scores["actual_risk"]).mean()
        print(f"  Holdout Accuracy vs tuned threshold: {acc:.2%}")

        out_scores = OUT_DIR / "behavioral_risk_scores.csv"
        out_comp = OUT_DIR / "model_comparison.csv"
        risk_scores.to_csv(out_scores, index=False)
        modeling["model_comparison"].to_csv(out_comp)
        print("\nSaved:")
        print(f"  - Risk scores -> {out_scores}")
        print(f"  - Model comparison -> {out_comp}")

        # 7) Feature importance
        print("\nSTEP 5: Feature Importance (best model)")
        fi = _best_feature_importance(risk_models)
        if not fi.empty:
            for i, r in enumerate(fi.head(10).itertuples(index=False), 1):
                print(f"  {i:2d}. {r.feature}: {r.importance:.4f}")
        else:
            print("  (No importance available for the selected estimator.)")

        print("\n" + "=" * 86)
        print("BEHAVIORAL RISK SCORING COMPLETED ✅")
        print("=" * 86)
        return risk_scores, risk_models, pipe_results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    _ = main()
