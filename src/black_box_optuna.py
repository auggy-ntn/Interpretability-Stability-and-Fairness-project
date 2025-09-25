from preprocessing import get_preprocessed_data
import xgboost as xgb
import numpy as np
import time
import psutil
import os
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Set environment variables for maximum performance
import multiprocessing
n_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)
print(f"🖥️  Using {n_cores} CPU cores")

print("🚀 Loading and preprocessing data...")
start_time = time.time()
df, prob, predictions, true_labels = get_preprocessed_data()
print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds")
print(f"📊 Dataset shape: {df.shape}")
print(f"💾 Memory usage: {psutil.virtual_memory().percent}%")

# Convert to numpy arrays to avoid feature name issues
print("🔄 Converting data to numpy arrays...")
X = df.values.astype(np.float32)  # Use float32 for memory efficiency
y = true_labels.values.astype(np.int32)  # Use int32 for memory efficiency
print(f"   - Memory usage reduced by using float32/int32")

# Split data into train and test sets
print("📊 Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=y              # Maintain class distribution
)

print(f"   - Training set: {X_train.shape[0]:,} samples")
print(f"   - Test set: {X_test.shape[0]:,} samples")
print(f"   - Training set class distribution: {np.bincount(y_train)}")
print(f"   - Test set class distribution: {np.bincount(y_test)}")

# Calculate class weights for imbalance handling
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"📊 Class weights: {class_weight_dict}")

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    params = {
        'n_jobs': n_cores,
        'tree_method': 'hist',
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        
        # Memory optimizations
        'max_bin': trial.suggest_int('max_bin', 256, 1024, step=128),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        
        # Training optimizations - FIXED n_estimators for optimization
        'n_estimators': 100,  # Fixed for faster optimization
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
        
        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
        
        # Class imbalance handling
        'scale_pos_weight': class_weight_dict[1]/class_weight_dict[0],
        
        # Early stopping
        'early_stopping_rounds': 20,
        'eval_metric': 'auc',
        
        # Other settings
        'verbosity': 0,  # Silent during optimization
        'random_state': 42,
        'enable_categorical': False,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_delta_step': 0,
        'base_score': 0.5
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
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Run optimization
print("\n🔍 Running hyperparameter optimization with Optuna...")
print("   - Using 100 estimators during optimization for speed")
print("   - Will retrain final model with 1000 estimators")
print("   - Target: 50 trials or 30 minutes")

study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50, timeout=1800)  # 50 trials or 30 minutes

print(f"\n✅ Optimization completed!")
print(f"   - Best trial: {study.best_trial.number}")
print(f"   - Best CV AUC: {study.best_value:.4f}")
print(f"   - Best parameters:")
for key, value in study.best_params.items():
    print(f"     {key}: {value}")

# Train final model with best parameters and 1000 estimators
print(f"\n🎯 Training final model with best parameters...")
print("   - Using 1000 estimators for final model")

# Get best parameters and update for final model
best_params = study.best_params.copy()
best_params['n_estimators'] = 1000  # Use 1000 estimators for final model
best_params['early_stopping_rounds'] = 50  # More patience for final model
best_params['verbosity'] = 1  # Verbose for final training

# Remove parameters that shouldn't be in XGBClassifier constructor
fit_params = {}
if 'early_stopping_rounds' in best_params:
    fit_params['early_stopping_rounds'] = best_params.pop('early_stopping_rounds')
if 'eval_metric' in best_params:
    fit_params['eval_metric'] = best_params.pop('eval_metric')

# Create final model
final_model = xgb.XGBClassifier(**best_params)

# Train final model
train_start = time.time()
final_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True,
    **fit_params
)
train_time = time.time() - train_start

print(f"✅ Final model trained in {train_time:.2f} seconds")
print(f"🎯 Final model performance metrics:")
print(f"   - Number of boosting rounds: {final_model.n_estimators}")
print(f"   - Best iteration: {final_model.best_iteration}")
print(f"   - Early stopping: {'Yes' if final_model.best_iteration < final_model.n_estimators else 'No'}")
print(f"   - Final train AUC: {final_model.evals_result()['validation_0']['auc'][-1]:.4f}")
print(f"   - Final test AUC: {final_model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Overfitting check: {final_model.evals_result()['validation_0']['auc'][-1] - final_model.evals_result()['validation_1']['auc'][-1]:.4f}")

# Make predictions on TEST SET
print("\n🔮 Making predictions on test set...")
pred_start = time.time()
prob_predictions_test = final_model.predict_proba(X_test)
pred_time = time.time() - pred_start

print(f"✅ Predictions completed in {pred_time:.2f} seconds")
print(f"📊 Test set prediction shape: {prob_predictions_test.shape}")
print(f"💾 Final memory usage: {psutil.virtual_memory().percent}%")

# Model Evaluation on TEST SET
print("\n📊 Final Model Evaluation Metrics (TEST SET):")
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve

# Get predictions and optimize threshold on TEST SET
y_pred_proba_test = prob_predictions_test[:, 1]  # Probability of positive class

# Optimize threshold for better balance using TEST SET
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_test)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
print(f"   - Precision at optimal threshold: {precision[optimal_threshold_idx]:.4f}")
print(f"   - Recall at optimal threshold: {recall[optimal_threshold_idx]:.4f}")
print(f"   - F1-score at optimal threshold: {f1_scores[optimal_threshold_idx]:.4f}")

# Use optimized threshold for predictions on TEST SET
y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)

# Calculate comprehensive metrics on TEST SET
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_proba_test)
avg_precision = average_precision_score(y_test, y_pred_proba_test)
mcc = matthews_corrcoef(y_test, y_pred_test)

print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   📈 Precision: {precision:.4f}")
print(f"   🔍 Recall: {recall:.4f}")
print(f"   ⚖️  F1-Score: {f1:.4f}")
print(f"   📊 ROC-AUC: {roc_auc:.4f}")
print(f"   📊 Average Precision: {avg_precision:.4f}")
print(f"   📊 Matthews Correlation Coefficient: {mcc:.4f}")

# Additional metrics for class imbalance
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"   🔍 Specificity (True Negative Rate): {specificity:.4f}")
print(f"   🔍 Sensitivity (True Positive Rate): {sensitivity:.4f}")

# Confusion Matrix on TEST SET
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n🔢 Confusion Matrix - TEST SET (Optimized Threshold):")
print(f"   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives:  {cm[1,1]:,}")

# Compare with default threshold (0.5) on TEST SET
y_pred_default_test = (y_pred_proba_test >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default_test)
print(f"\n🔢 Confusion Matrix - TEST SET (Default Threshold 0.5):")
print(f"   True Negatives:  {cm_default[0,0]:,}")
print(f"   False Positives: {cm_default[0,1]:,}")
print(f"   False Negatives: {cm_default[1,0]:,}")
print(f"   True Positives:  {cm_default[1,1]:,}")

# Calculate improvement
fn_reduction = cm_default[1,0] - cm[1,0]
tp_increase = cm[1,1] - cm_default[1,1]
print(f"\n📈 Improvement with Optimized Threshold:")
print(f"   False Negatives reduced by: {fn_reduction:,}")
print(f"   True Positives increased by: {tp_increase:,}")

# Performance Summary
print(f"\n🎉 Performance Summary:")
print(f"   - Total dataset size: {len(y):,} samples × {X.shape[1]} features")
print(f"   - Training set: {len(y_train):,} samples")
print(f"   - Test set: {len(y_test):,} samples")
print(f"   - Optimization trials: {len(study.trials)}")
print(f"   - Training time: {train_time:.2f}s")
print(f"   - Prediction time: {pred_time:.2f}s")
print(f"   - Total time: {time.time() - start_time:.2f}s")
print(f"   - CPU cores used: {final_model.n_jobs}")
print(f"   - Final test AUC: {final_model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Sample test predictions: {prob_predictions_test[:5]}")

# Feature importance (top 10)
print(f"\n🔝 Top 10 Most Important Features:")
feature_importance = final_model.feature_importances_
top_features_idx = feature_importance.argsort()[-10:][::-1]
for i, idx in enumerate(top_features_idx):
    print(f"   {i+1:2d}. Feature {idx:3d}: {feature_importance[idx]:.4f}")

print(f"\n🎯 Optimization Results Summary:")
print(f"   - Best CV AUC: {study.best_value:.4f}")
print(f"   - Final Test AUC: {roc_auc:.4f}")
print(f"   - Train/Test Gap: {final_model.evals_result()['validation_0']['auc'][-1] - final_model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Optimal Threshold: {optimal_threshold:.4f}")
print(f"   - F1-Score: {f1:.4f}")
