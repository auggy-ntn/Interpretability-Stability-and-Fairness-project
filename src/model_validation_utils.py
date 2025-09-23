import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
    """Evaluate a machine learning model using cross-validation.

    Args:
        model (Sklearn Model): Model to evaluate, has fit and predict methods.
        X (pd.DataFrame): Features to use for evaluation.
        y (pd.Series): True labels for the data.
        cv_splits (int, optional): Number of cross-validation splits. Defaults to 5.

    Returns:
        dict: Cross-validation scores for the model.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    results = {
        metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
        for metric in scoring.keys()
    }

    return results
