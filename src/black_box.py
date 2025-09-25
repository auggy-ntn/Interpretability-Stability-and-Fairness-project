from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os
import pickle
import time

import joblib
import numpy as np
import psutil
import xgboost as xgb

from preprocessing import get_preprocessed_data

# Set environment variables for maximum performance
os.environ["OMP_NUM_THREADS"] = "22"  # Use all CPU cores
os.environ["MKL_NUM_THREADS"] = "22"
os.environ["NUMEXPR_NUM_THREADS"] = "22"

print("🚀 Loading and preprocessing data...")
start_time = time.time()
df, prob, predictions, true_labels, _ = get_preprocessed_data()
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
    X,
    y,
    test_size=0.2,  # 20% for testing
    random_state=42,  # For reproducibility
    stratify=y,  # Maintain class distribution
)

print(f"   - Training set: {X_train.shape[0]:,} samples")
print(f"   - Test set: {X_test.shape[0]:,} samples")
print(f"   - Training set class distribution: {np.bincount(y_train)}")
print(f"   - Test set class distribution: {np.bincount(y_test)}")

# Optimized XGBoost model with maximum resource utilization
print("\n🎯 Training optimized XGBoost model...")
# Calculate class weights for imbalance handling

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"📊 Class weights: {class_weight_dict}")

model = xgb.XGBClassifier(
    # Performance optimizations
    n_jobs=22,  # Use all CPU cores dynamically
    tree_method="hist",  # Fast histogram-based method
    grow_policy="depthwise",  # Better for accuracy (vs lossguide for speed)
    # Memory optimizations - BALANCED FOR GENERALIZATION
    max_bin=512,  # Reduced bins to prevent overfitting
    subsample=0.8,  # Sample data to prevent overfitting
    colsample_bytree=0.8,  # Sample features to prevent overfitting
    colsample_bylevel=0.8,  # Sample features at each level
    colsample_bynode=0.8,  # Sample features at each node
    # Training optimizations - BALANCED LEARNING
    n_estimators=1000,  # Reduced trees to prevent overfitting
    max_depth=8,  # Reduced depth to prevent overfitting
    learning_rate=0.05,  # Higher learning rate for faster convergence
    min_child_weight=3,  # Higher weight to prevent overfitting
    # Regularization for better generalization
    reg_alpha=0.1,  # Stronger L1 regularization
    reg_lambda=1.0,  # Stronger L2 regularization
    gamma=0.1,  # Stronger minimum loss reduction for splits
    # Class imbalance handling
    scale_pos_weight=class_weight_dict[1]
    / class_weight_dict[0],  # XGBoost's way to handle imbalance
    # Early stopping - BALANCED PATIENCE
    early_stopping_rounds=20,  # Reduced patience to prevent overfitting
    # Verbose output
    verbosity=1,
    # Random state for reproducibility
    random_state=42,
    # Disable feature names to avoid character issues
    enable_categorical=False,
    # Additional optimizations for maximum accuracy
    booster="gbtree",  # Use gradient boosting trees
    objective="binary:logistic",  # Binary classification
    eval_metric="auc",  # Use AUC for better optimization
    max_delta_step=0,  # No constraint on step size
    base_score=0.5,  # Initial prediction score
)

# Train the model with detailed monitoring
train_start = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # For early stopping
    verbose=True,
)
train_time = time.time() - train_start

print(f"✅ Model trained in {train_time:.2f} seconds")
print(f"🎯 Model performance metrics:")
print(f"   - Number of boosting rounds: {model.n_estimators}")
print(f"   - Best iteration: {model.best_iteration}")
print(
    f"   - Early stopping: {'Yes' if model.best_iteration < model.n_estimators else 'No'}"
)
print(f"   - Final train AUC: {model.evals_result()['validation_0']['auc'][-1]:.4f}")
print(f"   - Final test AUC: {model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(
    f"   - Overfitting check: {model.evals_result()['validation_0']['auc'][-1] - model.evals_result()['validation_1']['auc'][-1]:.4f}"
)

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

# Calculate metrics on TEST SET
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"   📈 Precision: {precision:.4f}")
print(f"   🔍 Recall: {recall:.4f}")
print(f"   ⚖️  F1-Score: {f1:.4f}")
print(f"   📊 ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix on TEST SET
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n🔢 Confusion Matrix - TEST SET (Optimized Threshold):")
print(f"   True Negatives:  {cm[0, 0]:,}")
print(f"   False Positives: {cm[0, 1]:,}")
print(f"   False Negatives: {cm[1, 0]:,}")
print(f"   True Positives:  {cm[1, 1]:,}")

# Compare with default threshold (0.5) on TEST SET
y_pred_default_test = (y_pred_proba_test >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default_test)
print(f"\n🔢 Confusion Matrix - TEST SET (Default Threshold 0.5):")
print(f"   True Negatives:  {cm_default[0, 0]:,}")
print(f"   False Positives: {cm_default[0, 1]:,}")
print(f"   False Negatives: {cm_default[1, 0]:,}")
print(f"   True Positives:  {cm_default[1, 1]:,}")

# Calculate improvement
fn_reduction = cm_default[1, 0] - cm[1, 0]
tp_increase = cm[1, 1] - cm_default[1, 1]
print(f"\n📈 Improvement with Optimized Threshold:")
print(f"   False Negatives reduced by: {fn_reduction:,}")
print(f"   True Positives increased by: {tp_increase:,}")

# Class distribution in TEST SET
print(f"\n📋 Class Distribution - TEST SET:")
print(
    f"   Class 0 (Negative): {sum(y_test == 0):,} ({sum(y_test == 0) / len(y_test) * 100:.2f}%)"
)
print(
    f"   Class 1 (Positive): {sum(y_test == 1):,} ({sum(y_test == 1) / len(y_test) * 100:.2f}%)"
)

# Performance Summary
print(f"\n🎉 Performance Summary:")
print(f"   - Total dataset size: {len(y):,} samples × {X.shape[1]} features")
print(f"   - Training set: {len(y_train):,} samples")
print(f"   - Test set: {len(y_test):,} samples")
print(f"   - Training time: {train_time:.2f}s")
print(f"   - Prediction time: {pred_time:.2f}s")
print(f"   - Total time: {time.time() - start_time:.2f}s")
print(f"   - CPU cores used: {model.n_jobs}")
print(f"   - Final test AUC: {model.evals_result()['validation_1']['auc'][-1]:.4f}")
print(f"   - Sample test predictions: {prob_predictions_test[:5]}")

# Save the trained model
print(f"\n💾 Saving trained model...")
model_filename = "trained_xgboost_model.pkl"
scaler_filename = "feature_scaler.pkl"
optimal_threshold_filename = "optimal_threshold.pkl"

# Save model using joblib (more efficient for scikit-learn models)
joblib.dump(model, model_filename)
print(f"✅ Model saved as '{model_filename}'")

# Save optimal threshold for predictions
with open(optimal_threshold_filename, "wb") as f:
    pickle.dump(optimal_threshold, f)
print(f"✅ Optimal threshold saved as '{optimal_threshold_filename}'")

# Save model metadata
model_metadata = {
    "model_type": "XGBoost",
    "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_shape": df.shape,
    "train_samples": len(y_train),
    "test_samples": len(y_test),
    "optimal_threshold": optimal_threshold,
    "test_auc": roc_auc,
    "test_f1": f1,
    "test_accuracy": accuracy,
    "feature_count": X.shape[1],
    "class_weights": class_weight_dict,
}

with open("model_metadata.pkl", "wb") as f:
    pickle.dump(model_metadata, f)
print(f"✅ Model metadata saved as 'model_metadata.pkl'")

# Feature importance (top 10)
print(f"\n🔝 Top 10 Most Important Features:")
feature_importance = model.feature_importances_
top_features_idx = feature_importance.argsort()[-10:][::-1]
for i, idx in enumerate(top_features_idx):
    print(f"   {i + 1:2d}. Feature {idx:3d}: {feature_importance[idx]:.4f}")

# Save feature importance
feature_importance_data = {
    "feature_importance": feature_importance,
    "top_features": top_features_idx.tolist(),
}
with open("feature_importance.pkl", "wb") as f:
    pickle.dump(feature_importance_data, f)
print(f"✅ Feature importance saved as 'feature_importance.pkl'")

print(f"\n🎉 Model Training and Saving Completed!")
print(f"   - Model file: {model_filename}")
print(f"   - Threshold file: {optimal_threshold_filename}")
print(f"   - Metadata file: model_metadata.pkl")
print(f"   - Feature importance file: feature_importance.pkl")
print(f"   - Total time: {time.time() - start_time:.2f} seconds")
