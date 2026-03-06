from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier


def _build_preprocessor(numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_tf, numeric_cols),
        ("cat", cat_tf, cat_cols)
    ])


def _recall_at_top_k(y_true: np.ndarray, y_score: np.ndarray, top_k_pct: float) -> float:
    if not (0 < top_k_pct <= 1):
        raise ValueError("top_k_pct must be in (0, 1].")

    n = len(y_true)
    k = int(np.ceil(n * top_k_pct))
    if k <= 0:
        return 0.0

    idx = np.argsort(-y_score)[:k]
    y_true = np.asarray(y_true)

    positives = y_true.sum()
    if positives == 0:
        return 0.0

    return float(y_true[idx].sum() / positives)


def run_model_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_num: List[str],
    base_cat: List[str],
    community_cols: List[str],
    model,
    leakage: Optional[List[str]] = None,
    threshold: float = 0.5,
    top_k_pct: float = 0.10,
) -> Dict[str, float]:
    """
    Generic experiment runner for binary classification.

    Returned metrics:
      - auc_roc
      - auc_pr
      - precision
      - recall
      - f1
      - recall_at_top_k
    """
    if leakage is None:
        leakage = []

    train = train.dropna(subset=[target]).copy()
    test = test.dropna(subset=[target]).copy()

    numeric_cols = [c for c in (base_num + community_cols) if c not in leakage]
    cat_cols = list(base_cat)

    X_train = train[numeric_cols + cat_cols].copy()
    X_test = test[numeric_cols + cat_cols].copy()

    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    pre = _build_preprocessor(numeric_cols, cat_cols)

    clf = Pipeline([
        ("preprocess", pre),
        ("model", clone(model))
    ])

    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        raw_score = clf.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-raw_score))  # sigmoid transform
    else:
        raise ValueError("Model must support predict_proba or decision_function.")

    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_score)),
        "auc_pr": float(average_precision_score(y_test, y_score)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall_at_top_k": float(_recall_at_top_k(y_test, y_score, top_k_pct=top_k_pct)),
        "threshold": float(threshold),
        "top_k_pct": float(top_k_pct),
    }

    return metrics

from sklearn.linear_model import LogisticRegression


def run_logit_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_num: List[str],
    base_cat: List[str],
    community_cols: List[str],
    leakage: Optional[List[str]] = None,
    threshold: float = 0.5,
    top_k_pct: float = 0.10,
    logit_params: Optional[Dict] = None,
) -> Dict[str, float]:
    default_params = dict(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    if logit_params:
        default_params.update(logit_params)

    model = LogisticRegression(**default_params)

    return run_model_experiment(
        train=train,
        test=test,
        target=target,
        base_num=base_num,
        base_cat=base_cat,
        community_cols=community_cols,
        model=model,
        leakage=leakage,
        threshold=threshold,
        top_k_pct=top_k_pct,
    )

from sklearn.tree import DecisionTreeClassifier


def run_dt_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_num: List[str],
    base_cat: List[str],
    community_cols: List[str],
    leakage: Optional[List[str]] = None,
    threshold: float = 0.5,
    top_k_pct: float = 0.10,
    dt_params: Optional[Dict] = None,
) -> Dict[str, float]:
    default_params = dict(
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5,
        max_depth=8,
    )
    if dt_params:
        default_params.update(dt_params)

    model = DecisionTreeClassifier(**default_params)

    return run_model_experiment(
        train=train,
        test=test,
        target=target,
        base_num=base_num,
        base_cat=base_cat,
        community_cols=community_cols,
        model=model,
        leakage=leakage,
        threshold=threshold,
        top_k_pct=top_k_pct,
    )

from sklearn.ensemble import GradientBoostingClassifier


def run_gb_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_num: List[str],
    base_cat: List[str],
    community_cols: List[str],
    leakage: Optional[List[str]] = None,
    threshold: float = 0.5,
    top_k_pct: float = 0.10,
    gb_params: Optional[Dict] = None,
) -> Dict[str, float]:
    default_params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    if gb_params:
        default_params.update(gb_params)

    model = GradientBoostingClassifier(**default_params)

    return run_model_experiment(
        train=train,
        test=test,
        target=target,
        base_num=base_num,
        base_cat=base_cat,
        community_cols=community_cols,
        model=model,
        leakage=leakage,
        threshold=threshold,
        top_k_pct=top_k_pct,
    )

from sklearn.ensemble import RandomForestClassifier


def run_rf_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_num: List[str],
    base_cat: List[str],
    community_cols: List[str],
    leakage: Optional[List[str]] = None,
    threshold: float = 0.5,
    top_k_pct: float = 0.10,
    rf_params: Optional[Dict] = None,
) -> Dict[str, float]:
    default_params = dict(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
    )
    if rf_params:
        default_params.update(rf_params)

    model = RandomForestClassifier(**default_params)

    return run_model_experiment(
        train=train,
        test=test,
        target=target,
        base_num=base_num,
        base_cat=base_cat,
        community_cols=community_cols,
        model=model,
        leakage=leakage,
        threshold=threshold,
        top_k_pct=top_k_pct,
    )