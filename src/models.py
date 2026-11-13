"""Model suite, leakage-free pipelines and source-only hyperparameter selection.

Chapter 3 §3.6.2 and §3.7. Two properties are enforced structurally rather than
by discipline: every transformer is fitted inside the training fold within one
Pipeline object, and for Tiers 2-4 the search sees source data only — the target
is never used for tuning, so the transfer estimate is honest rather than
optimistic (the artefact Boyer & Veeramachaneni, 2015, identify).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

MODEL_LABELS = {
    "majority": "Majority-class baseline",
    "stratified": "Prevalence-stratified baseline",
    "logreg": "L2 logistic regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
LEARNERS = ["logreg", "rf", "xgb"]


def _ohe():
    """OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(num_cols, cat_cols, scale: bool):
    """Median/modal imputation with retained missingness indicators (§3.6.2)."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", _ohe())]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimator(name: str, seed: int, class_weight=None):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logreg":
        # sklearn deprecated penalty= in 1.8; L2 is the default, so omitting it
        # keeps the same model and stays forward-compatible.
        return LogisticRegression(solver="lbfgs", max_iter=2000,
                                  class_weight=class_weight, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed,
                                      class_weight=class_weight)
    if name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost is required for the 'xgb' learner")
        return XGBClassifier(
            n_jobs=-1, random_state=seed, eval_metric="logloss",
            tree_method="hist", enable_categorical=False,
        )
    raise ValueError(f"unknown model: {name}")


def search_space(name: str) -> dict:
    """Randomised search grids — pre-registered in Appendix B, fixed before any
    test-set evaluation."""
    if name == "logreg":
        return {"clf__C": loguniform(1e-3, 1e2)}
    if name == "rf":
        return {
            "clf__n_estimators": randint(200, 801),
            "clf__max_depth": randint(3, 25),
            "clf__min_samples_leaf": randint(1, 50),
            "clf__max_features": uniform(0.1, 0.8),
        }
    if name == "xgb":
        return {
            "clf__n_estimators": randint(100, 601),
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_depth": randint(2, 9),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_weight": randint(1, 20),
            "clf__reg_lambda": loguniform(1e-2, 1e2),
        }
    return {}


def make_pipeline(name: str, num_cols, cat_cols, seed: int, class_weight=None) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols, scale=(name == "logreg"))),
        ("clf", build_estimator(name, seed, class_weight)),
    ])


def _balanced_scale_pos_weight(y) -> float:
    """XGBoost's equivalent of class_weight='balanced'.

    XGBClassifier has no class_weight parameter, so passing one to it is a silent
    no-op - which would make an imbalance-treatment comparison report two
    identical rows while appearing to have tested something.
    """
    y = np.asarray(y)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return (neg / pos) if pos > 0 else 1.0


def fit_with_search(name, X, y, groups, num_cols, cat_cols, cfg, class_weight=None):
    """Fit `name` with inner-loop randomised search on the data given.

    The caller passes source data only for Tiers 2-4, which is what makes the
    source-only tuning guarantee structural rather than procedural.
    """
    seed = int(cfg["models"]["seed"])
    pipe = make_pipeline(name, num_cols, cat_cols, seed, class_weight)
    if name == "xgb" and class_weight == "balanced":
        pipe.set_params(clf__scale_pos_weight=_balanced_scale_pos_weight(y))
    space = search_space(name)
    if not space:
        pipe.fit(X, y)
        return pipe, {}
    inner = StratifiedGroupKFold(
        n_splits=int(cfg["models"]["inner_folds"]), shuffle=True, random_state=seed
    )
    n_pos = int(np.sum(y))
    if n_pos < inner.get_n_splits() or (len(y) - n_pos) < inner.get_n_splits():
        pipe.fit(X, y)
        return pipe, {}
    search = RandomizedSearchCV(
        pipe, space,
        n_iter=int(cfg["models"]["search_draws"]),
        scoring="roc_auc",
        cv=inner.split(X, y, groups),
        random_state=seed, n_jobs=1, refit=True, error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def predict_proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.ravel(p)
