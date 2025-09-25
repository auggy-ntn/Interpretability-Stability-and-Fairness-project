import sys
import os
import numpy as np
import time
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, average_precision_score, 
    matthews_corrcoef, precision_recall_curve
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import get_preprocessed_data

def train_xgboost_optimal(X_train, X_test, y_train, y_test, verbose=True):
    """
    Train XGBoost model with optimal parameters on given train/test data.
    
    Args:
        X_train: Training features (numpy array or pandas DataFrame)
        X_test: Test features (numpy array or pandas DataFrame)
        y_train: Training labels (numpy array or pandas Series)
        y_test: Test labels (numpy array or pandas Series)
        verbose: Whether to print training progress
    
    Returns:
        model: Trained XGBoost model
        y_pred_proba: Test set probability predictions
        y_pred: Test set binary predictions
        optimal_threshold: Optimized threshold for binary predictions
        metrics: Dictionary of performance metrics
    """
    # Set environment variables for maximum performance
    n_cores = os.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)
    
    # Convert to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values.astype(np.float32)
    if hasattr(X_test, 'values'):
        X_test = X_test.values.astype(np.float32)
    if hasattr(y_train, 'values'):
        y_train = y_train.values.astype(np.int32)
    if hasattr(y_test, 'values'):
        y_test = y_test.values.astype(np.int32)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    if verbose:
        print(f"Training XGBoost with optimal parameters on {X_train.shape[0]:,} samples")
    
    # Optimal parameters from Optuna
    model = xgb.XGBClassifier(
        n_jobs=n_cores,
        tree_method='hist',
        grow_policy='lossguide',
        max_bin=768,
        subsample=0.9477443357082947,
        colsample_bytree=0.7464202834952258,
        colsample_bylevel=0.9251402267772648,
        colsample_bynode=0.7878653321203397,
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.1795936234963652,
        min_child_weight=4.969727693084857,
        reg_alpha=0.016717049859161764,
        reg_lambda=0.20221883106199293,
        gamma=0.010739324463762273,
        scale_pos_weight=class_weight_dict[1]/class_weight_dict[0],
        early_stopping_rounds=50,
        eval_metric='auc',
        verbosity=0,
        random_state=42,
        enable_categorical=False,
        booster='gbtree',
        objective='binary:logistic',
        max_delta_step=0,
        base_score=0.5
    )
    
    # Train model
    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    train_time = time.time() - train_start
    
    if verbose:
        print(f"Model trained in {train_time:.2f} seconds")
        print(f"Best iteration: {model.best_iteration}")
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'mcc': mcc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'train_time': train_time
    }
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return model, y_pred_proba, y_pred, optimal_threshold, metrics

# Set environment variables for maximum performance
n_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)

if __name__ == "__main__":
    print(f"Using {n_cores} CPU cores")
    print("Loading and preprocessing data...")

    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataproject2025.csv")
    df, prob, predictions, true_labels = get_preprocessed_data(path=data_path)

    # Convert to numpy arrays
    X = df.values.astype(np.float32)
    y = true_labels.values.astype(np.int32)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")

    # Train model using the function
    model, y_pred_proba, y_pred, optimal_threshold, metrics = train_xgboost_optimal(
        X_train, X_test, y_train, y_test, verbose=True
    )

    print(f"\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_filename = 'models/black_box_xgboost.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")

    print(f"Training completed in {metrics['train_time']:.2f} seconds")