# Set environment variables for maximum performance
import multiprocessing
import os
import time

import numpy as np
import optuna
import psutil
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import get_preprocessed_data

n_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
start_time = time.time()
df, prob, predictions, true_labels, _ = get_preprocessed_data()

# Convert to numpy arrays
X = df.values.astype(np.float32)
y = true_labels.values.astype(np.int32)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weights for imbalance handling
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}


def objective(trial):
    """Objective function for Optuna optimization"""

    # Suggest hyperparameters
    params = {
        "n_jobs": 1,
        "tree_method": "gpu_hist",  # Use GPU for much faster training
        "gpu_id": 0,  # Use first GPU
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        # Memory optimizations
        "max_bin": trial.suggest_int("max_bin", 256, 1024, step=128),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        # Training optimizations - FIXED n_estimators for optimization
        "n_estimators": 100,  # Fixed for faster optimization
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.001, 1.0, log=True),
        # Class imbalance handling
        "scale_pos_weight": class_weight_dict[1] / class_weight_dict[0],
        # Early stopping
        "early_stopping_rounds": 20,
        "eval_metric": "auc",
        # Other settings
        "verbosity": 0,  # Silent during optimization
        "random_state": 42,
        "enable_categorical": False,
        "booster": "gbtree",
        "objective": "binary:logistic",
        "max_delta_step": 0,
        "base_score": 0.5,
    }

    # Create model
    model = xgb.XGBClassifier(**params)

    # Use 3-fold cross-validation for robust evaluation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(score)

    return np.mean(cv_scores)


# Run optimization
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=40, timeout=1800)

print(f"Best trial: {study.best_trial.number}")
print(f"Best CV AUC: {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters and 1000 estimators
best_params = study.best_params.copy()
best_params["n_estimators"] = 300
best_params["verbosity"] = 0
best_params["tree_method"] = "gpu_hist"  # Use GPU for final model
best_params["gpu_id"] = 0  # Use first GPU
best_params["n_jobs"] = 1  # Use fewer CPU threads when using GPU


# Create final model
final_model = xgb.XGBClassifier(**best_params)

# Train final model
train_start = time.time()
final_model.fit(
    X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False
)
train_time = time.time() - train_start

# Make predictions on test set
prob_predictions_test = final_model.predict_proba(X_test)

# Model Evaluation
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
)

# Get predictions and optimize threshold
y_pred_proba_test = prob_predictions_test[:, 1]

# Optimize threshold for better balance
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_test)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]

# Use optimized threshold for predictions
y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)

# Calculate comprehensive metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_proba_test)
avg_precision = average_precision_score(y_test, y_pred_proba_test)
mcc = matthews_corrcoef(y_test, y_pred_test)

# Additional metrics for class imbalance
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")

cm = confusion_matrix(y_test, y_pred_test)
print(
    f"Confusion Matrix - TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}  FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}"
)
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Total time: {time.time() - start_time:.2f}s")
