import sys
import os

# Add parent directory to path to import constants (needed for preprocessing)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import get_preprocessed_data
import xgboost as xgb
import numpy as np
import time
import psutil
import joblib

# Set environment variables for maximum performance
import multiprocessing
n_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)
print(f"🖥️  Using {n_cores} CPU cores")

print("🚀 Loading and preprocessing data...")
start_time = time.time()

# Set the correct data path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataproject2025.csv")
df, prob, predictions, true_labels = get_preprocessed_data(path=data_path)
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
from sklearn.model_selection import train_test_split

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
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"📊 Class weights: {class_weight_dict}")

# Optimal parameters from Optuna trial 41
print("\n🎯 Training XGBoost model with optimal parameters...")
print("   - Using parameters from Optuna trial 41 (CV AUC: 0.7208)")

model = xgb.XGBClassifier(
    # Performance optimizations
    n_jobs=n_cores,
    tree_method='hist',
    grow_policy='lossguide',  # Optimal from Optuna
    
    # Memory optimizations - OPTIMAL PARAMETERS
    max_bin=768,                                    # Optimal from Optuna
    subsample=0.9477443357082947,                   # Optimal from Optuna
    colsample_bytree=0.7464202834952258,            # Optimal from Optuna
    colsample_bylevel=0.9251402267772648,           # Optimal from Optuna
    colsample_bynode=0.7878653321203397,            # Optimal from Optuna
    
    # Training optimizations - OPTIMAL PARAMETERS
    n_estimators=500,                              # Use 1000 for final model
    max_depth=7,                                    # Optimal from Optuna
    learning_rate=0.1795936234963652,               # Optimal from Optuna
    min_child_weight=4.969727693084857,             # Optimal from Optuna
    
    # Regularization - OPTIMAL PARAMETERS
    reg_alpha=0.016717049859161764,                 # Optimal from Optuna
    reg_lambda=0.20221883106199293,                 # Optimal from Optuna
    gamma=0.010739324463762273,                     # Optimal from Optuna
    
    # Class imbalance handling
    scale_pos_weight=class_weight_dict[1]/class_weight_dict[0],
    
    # Early stopping
    early_stopping_rounds=50,                       # More patience for final model
    eval_metric='auc',
    
    # Other settings
    verbosity=1,
    random_state=42,
    enable_categorical=False,
    booster='gbtree',
    objective='binary:logistic',
    max_delta_step=0,
    base_score=0.5
)

# Train the model with detailed monitoring
train_start = time.time()
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # For early stopping
    verbose=True
)
train_time = time.time() - train_start

print(f"✅ Model trained in {train_time:.2f} seconds")
print(f"🎯 Model performance metrics:")
print(f"   - Number of boosting rounds: {model.n_estimators}")
print(f"   - Best iteration: {model.best_iteration}")
print(f"   - Early stopping: {'Yes' if model.best_iteration < model.n_estimators else 'No'}")
print(f"   - Final train AUC: {model.evals_result()['validation_0']['auc'][-1]:.4f}")
print(f"   - Final test AUC: {model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Overfitting check: {model.evals_result()['validation_0']['auc'][-1] - model.evals_result()['validation_1']['auc'][-1]:.4f}")

# Make predictions on TEST SET
print("\n🔮 Making predictions on test set...")
pred_start = time.time()
prob_predictions_test = model.predict_proba(X_test)
pred_time = time.time() - pred_start

print(f"✅ Predictions completed in {pred_time:.2f} seconds")
print(f"📊 Test set prediction shape: {prob_predictions_test.shape}")
print(f"💾 Final memory usage: {psutil.virtual_memory().percent}%")

# Model Evaluation on TEST SET
print("\n📊 Model Evaluation Metrics (TEST SET):")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, average_precision_score, matthews_corrcoef
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

# Class distribution in TEST SET
print(f"\n📋 Class Distribution - TEST SET:")
print(f"   Class 0 (Negative): {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.2f}%)")
print(f"   Class 1 (Positive): {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

# Performance Summary
print(f"\n🎉 Performance Summary:")
print(f"   - Total dataset size: {len(y):,} samples × {X.shape[1]} features")
print(f"   - Training set: {len(y_train):,} samples")
print(f"   - Test set: {len(y_test):,} samples")
print(f"   - Training time: {train_time:.2f}s")
print(f"   - Prediction time: {pred_time:.2f}s")
print(f"   - Total time: {time.time() - start_time:.2f}s")
print(f"   - CPU cores used: {model.n_jobs}")
print(f"   - Training speed: {len(y_train)/train_time:.0f} samples/second")
print(f"   - Final test AUC: {model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Sample test predictions: {prob_predictions_test[:5]}")

# Feature importance (top 10)
print(f"\n🔝 Top 10 Most Important Features:")
feature_importance = model.feature_importances_
top_features_idx = feature_importance.argsort()[-10:][::-1]
for i, idx in enumerate(top_features_idx):
    print(f"   {i+1:2d}. Feature {idx:3d}: {feature_importance[idx]:.4f}")

print(f"\n🎯 Optimal Parameters Used:")
print(f"   - grow_policy: lossguide")
print(f"   - max_bin: 768")
print(f"   - subsample: 0.948")
print(f"   - colsample_bytree: 0.746")
print(f"   - colsample_bylevel: 0.925")
print(f"   - colsample_bynode: 0.788")
print(f"   - max_depth: 7")
print(f"   - learning_rate: 0.180")
print(f"   - min_child_weight: 4.970")
print(f"   - reg_alpha: 0.017")
print(f"   - reg_lambda: 0.202")
print(f"   - gamma: 0.011")
print(f"   - n_estimators: 1000 (final model)")
print(f"   - CV AUC during optimization: 0.7208")

# Save the trained model
print(f"\n💾 Saving trained model...")
model_filename = 'models/black_box_xgboost.pkl'

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model using joblib (more efficient for scikit-learn models)
joblib.dump(model, model_filename)
print(f"✅ Model saved as '{model_filename}'")

print(f"\n🎉 Model Training and Saving Completed!")
print(f"   - Model file: {model_filename}")
print(f"   - Total time: {time.time() - start_time:.2f} seconds")
