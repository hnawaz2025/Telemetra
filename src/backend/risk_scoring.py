# src/backend/risk_scoring.py
"""
Expert Risk Scoring (calibrated, threshold-tuned)

Works with BehavioralRiskPipeline outputs:
- X_train, X_test, y_train, y_test, feature_columns, driver_data

Models: Logistic Regression, Random Forest (+ optional XGBoost)
Features: optional SelectKBest
Probabilities: calibrated (isotonic/sigmoid)
Threshold: tuned on CV folds (F1 / Youden / cost)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import warnings
from pathlib import Path
import joblib, json

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    brier_score_loss, log_loss, precision_recall_curve, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")
logger = logging.getLogger("risk_scoring")
logger.setLevel(logging.INFO)

# Optional XGBoost
try:
    import xgboost as xgb
    _XGB_OK = True
except Exception:
    _XGB_OK = False


# -------------------------
# Config
# -------------------------
@dataclass
class ModelConfig:
    random_state: int = 42
    cv_folds: int = 5

    # feature selection
    feature_selection: bool = True
    max_features: int = 30

    # probability calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid"

    # threshold objective on TRAIN (CV)
    threshold_objective: str = "f1"       # "f1" | "youden" | "cost"
    cost_fp: float = 1.0                  # used if objective="cost"
    cost_fn: float = 5.0

    # model control
    use_xgboost: bool = True              # will be used only if xgboost is available


# ---------------------
# Helpers
# -------------------------
def _choose_threshold(proba: np.ndarray, y_true: np.ndarray,
                      objective: str = "f1",
                      cost_fp: float = 1.0, cost_fn: float = 5.0) -> float:
    """
    Pick threshold given validation proba + labels.
    """
    # guard when tiny datasets
    if len(np.unique(y_true)) < 2:
        return 0.5

    prec, rec, thr_pr = precision_recall_curve(y_true, proba)
    # scikit returns thresholds of size n-1
    if objective == "f1":
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
        best_idx = int(np.nanargmax(f1s))
        return float(thr_pr[best_idx])

    if objective == "youden":
        fpr, tpr, thr_roc = roc_curve(y_true, proba)
        j = tpr - fpr
        best_idx = int(np.nanargmax(j))
        return float(thr_roc[best_idx])

    if objective == "cost":
        # minimize cost = FP*cost_fp + FN*cost_fn
        thresholds = thr_pr
        best_thr = 0.5
        best_cost = float("inf")
        for t in thresholds:
            pred = (proba >= t).astype(int)
            fp = np.sum((pred == 1) & (y_true == 0))
            fn = np.sum((pred == 0) & (y_true == 1))
            cost = fp * cost_fp + fn * cost_fn
            if cost < best_cost:
                best_cost = cost
                best_thr = t
        return float(best_thr)

    return 0.5


def _calc_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    out = {
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "avg_precision": average_precision_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }
    try:
        out["log_loss"] = log_loss(y_true, proba, labels=[0,1])
    except Exception:
        out["log_loss"] = np.nan
    return out


# -------------------------
# Main class
# -------------------------
class BehavioralRiskModels:
    """
    Trains several calibrated classifiers, tunes threshold on CV,
    and exposes a consistent scoring API.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.models_: Dict[str, Any] = {}
        self.calibrated_: Dict[str, Any] = {}
        self.model_scores_: Dict[str, Dict[str, float]] = {}
        self.best_: Optional[Dict[str, Any]] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.selected_features_: List[str] = []
        self.tuned_threshold_: float = 0.5
        self.threshold_source_: str = "default"

    # ---------- init models ----------
    def _init_models(self) -> Dict[str, Any]:
        rng = self.cfg.random_state
        models: Dict[str, Any] = {
            "logreg": LogisticRegression(
                solver="liblinear", random_state=rng, class_weight="balanced", max_iter=1000
            ),
            "rf": RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=10, class_weight="balanced",
                random_state=rng, n_jobs=-1
            ),
        }
        if self.cfg.use_xgboost and _XGB_OK:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, reg_alpha=0.0, random_state=rng, eval_metric="logloss", n_jobs=-1,
                # handle imbalance-ish
                scale_pos_weight=1.0
            )
        return models

    # ---------- feature selection ----------
    def _fit_feature_selector(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        if not self.cfg.feature_selection:
            self.selected_features_ = X_train.columns.tolist()
            return X_train, self.selected_features_

        k = min(self.cfg.max_features, X_train.shape[1])
        sel = SelectKBest(score_func=f_classif, k=k)
        Xs = sel.fit_transform(X_train, y_train)
        mask = sel.get_support(indices=True)
        cols = X_train.columns[mask].tolist()

        self.feature_selector = sel
        self.selected_features_ = cols
        return pd.DataFrame(Xs, index=X_train.index, columns=cols), cols

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_selector is None:
            return X[self.selected_features_] if self.selected_features_ else X
        Xt = self.feature_selector.transform(X)
        return pd.DataFrame(Xt, index=X.index, columns=self.selected_features_)

    # ---------- threshold via CV on train ----------
    def _tune_threshold_cv(self, clf, Xtr: pd.DataFrame, ytr: pd.Series) -> float:
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        all_oof: List[float] = []
        all_y: List[int] = []

        for tr_idx, va_idx in skf.split(Xtr, ytr):
            X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
            y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]

            # calibrate inside CV
            if self.cfg.calibrate_probabilities:
                cal = CalibratedClassifierCV(clf, cv=3, method=self.cfg.calibration_method)
                cal.fit(X_tr, y_tr)
                proba = cal.predict_proba(X_va)[:, 1]
            else:
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_va)[:, 1]

            all_oof.append(proba)
            all_y.append(y_va.values)

        oof_p = np.concatenate(all_oof)
        oof_y = np.concatenate(all_y)
        thr = _choose_threshold(
            oof_p, oof_y, objective=self.cfg.threshold_objective,
            cost_fp=self.cfg.cost_fp, cost_fn=self.cfg.cost_fn
        )
        return float(thr)

    # ---------- train/evaluate ----------
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Trains all base models, calibrates, tunes threshold on TRAIN only,
        evaluates on TEST, and picks the best by AUC.
        """
        models = self._init_models()

        # feature selection on train only
        Xtr_sel, cols = self._fit_feature_selector(X_train, y_train)
        Xte_sel = self._transform_features(X_test)

        best_name = None
        best_auc = -np.inf

        for name, base in models.items():
            logger.info(f"Training {name}…")

            # tune threshold using CV on train
            thr = self._tune_threshold_cv(base, Xtr_sel, y_train)

            # fit calibrated model on full train
            if self.cfg.calibrate_probabilities:
                model = CalibratedClassifierCV(base, cv=5, method=self.cfg.calibration_method)
                model.fit(Xtr_sel, y_train)
            else:
                base.fit(Xtr_sel, y_train)
                model = base

            # score on test (holdout)
            proba_te = model.predict_proba(Xte_sel)[:, 1]
            metrics = _calc_metrics(y_test.values, proba_te, thr)

            self.models_[name] = base
            self.calibrated_[name] = model
            self.model_scores_[name] = {**metrics, "tuned_threshold": thr}

            if np.isfinite(metrics["roc_auc"]) and metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = name
                self.tuned_threshold_ = thr
                self.threshold_source_ = "CV(train)"

        # store best model
        if best_name is None:
            # fallback
            best_name = list(self.calibrated_.keys())[0]
        
        self.best_ = {
            "name": best_name,
            "model": self.calibrated_[best_name],
            "metrics": self.model_scores_[best_name],
        }
        models_dir = Path("models")               # ✅ relative path to your repo root
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_["model"], models_dir / "claim_calibrated.joblib")

        # Save feature selector (if enabled)
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, models_dir / "feature_selector.joblib")

        # Minimal metadata
        with open(models_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "best_model": self.best_["name"],
                    "metrics": self.best_["metrics"],
                    "selected_features": self.selected_features_,
                },
                f,
                indent=2,)
        return {
            "trained_models": self.calibrated_,
            "model_scores": self.model_scores_,
            "best_model": self.best_,
            "selected_features": self.selected_features_,
        }

        

    # ---------- API used by RiskScorer ----------
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        if model_name is None:
            if self.best_ is None:
                raise RuntimeError("No model has been trained.")
            model = self.best_["model"]
        else:
            model = self.calibrated_.get(model_name)
            if model is None:
                raise KeyError(f"Unknown model '{model_name}'")
        Xs = self._transform_features(X)
        return model.predict_proba(Xs)[:, 1]


# -------------------------
# Scorer / runner helpers
# -------------------------
class RiskScorer:
    def __init__(self, trained: BehavioralRiskModels):
        self.models = trained

    def calculate_risk_score(self, X: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
        proba = self.models.predict_proba(X, model_name=model_name)
        thr = getattr(self.models, "tuned_threshold_", 0.5)

        pred = (proba >= thr).astype(int)
        score = (proba * 100.0).clip(0, 100)

        out = pd.DataFrame({
            "risk_prediction": pred,
            "risk_probability": proba,
            "risk_score": score.round(2),
            "risk_category": pd.cut(
                score, bins=[-0.1, 20, 40, 60, 80, 100],
                labels=["Very Low", "Low", "Medium", "High", "Very High"]
            )
        }, index=X.index)
        return out


def run_behavioral_risk_modeling(pipeline_results: Dict[str, Any],
                                 config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    """
    Entry point used by runner scripts (expects pipeline dict).
    """
    cfg = config or ModelConfig()

    X_train = pipeline_results["X_train"]
    X_test = pipeline_results["X_test"]
    y_train = pipeline_results["y_train"]
    y_test = pipeline_results["y_test"]

    modeler = BehavioralRiskModels(cfg)
    results = modeler.train_models(X_train, y_train, X_test, y_test)

    # comparison table
    comp = pd.DataFrame(results["model_scores"]).T.sort_values("roc_auc", ascending=False)

    out = {
        **results,
        "model_comparison": comp,
        "risk_models_instance": modeler,
    }
    return out


if __name__ == "__main__":
    print("risk_scoring.py ready (expert-calibrated).")
