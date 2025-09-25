from preprocessing import get_preprocessed_data
import numpy as np
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

# Scale features (essential for neural networks)
print("🔄 Scaling features for neural network...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   - Features scaled to mean=0, std=1")

# Define different neural network architectures
print("\n🧠 Creating Neural Network Models...")

# 1. Simple Neural Network (1 hidden layer)
simple_nn = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    class_weight='balanced',
    random_state=42
)

# 2. Deep Neural Network (3 hidden layers)
deep_nn = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,  # Lighter regularization
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    class_weight='balanced',
    random_state=42
)

# 3. Wide Neural Network (1 wide hidden layer)
wide_nn = MLPClassifier(
    hidden_layer_sizes=(500,),
    activation='relu',
    solver='adam',
    alpha=0.01,  # Stronger regularization for wide network
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    class_weight='balanced',
    random_state=42
)

# 4. Complex Neural Network (4 hidden layers)
complex_nn = MLPClassifier(
    hidden_layer_sizes=(300, 200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    max_iter=1500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=40,
    class_weight='balanced',
    random_state=42
)

# 5. Neural Network with Tanh activation
tanh_nn = MLPClassifier(
    hidden_layer_sizes=(200, 100),
    activation='tanh',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    class_weight='balanced',
    random_state=42
)

# Train and evaluate neural networks
models = {
    'Simple NN (100)': simple_nn,
    'Deep NN (200,100,50)': deep_nn,
    'Wide NN (500)': wide_nn,
    'Complex NN (300,200,100,50)': complex_nn,
    'Tanh NN (200,100)': tanh_nn
}

results = {}

print("\n🔍 Training Neural Networks...")
for name, model in models.items():
    print(f"\n📊 Training {name}...")
    train_start = time.time()
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'training_time': train_time,
        'n_iterations': model.n_iter_,
        'loss': model.loss_
    }
    
    print(f"   ✅ {name} Results:")
    print(f"      - Accuracy: {accuracy:.4f}")
    print(f"      - Precision: {precision:.4f}")
    print(f"      - Recall: {recall:.4f}")
    print(f"      - F1-Score: {f1:.4f}")
    print(f"      - ROC-AUC: {roc_auc:.4f}")
    print(f"      - Training time: {train_time:.2f}s")
    print(f"      - Iterations: {model.n_iter_}")
    print(f"      - Final loss: {model.loss_:.4f}")

# Find best neural network
best_nn_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
best_nn = models[best_nn_name]

print(f"\n🏆 Best Neural Network: {best_nn_name}")
print(f"   - ROC-AUC: {results[best_nn_name]['roc_auc']:.4f}")
print(f"   - F1-Score: {results[best_nn_name]['f1']:.4f}")
print(f"   - Accuracy: {results[best_nn_name]['accuracy']:.4f}")

# Detailed analysis of best neural network
print(f"\n📊 Detailed Analysis of Best Neural Network ({best_nn_name}):")

# Get predictions from best model
y_pred_proba_best = best_nn.predict_proba(X_test_scaled)[:, 1]
y_pred_best = best_nn.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print(f"\n🔢 Confusion Matrix:")
print(f"   True Negatives:  {tn:,}")
print(f"   False Positives: {fp:,}")
print(f"   False Negatives: {fn:,}")
print(f"   True Positives:  {tp:,}")

# Additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n📈 Additional Metrics:")
print(f"   - Specificity: {specificity:.4f}")
print(f"   - Sensitivity: {sensitivity:.4f}")

# Threshold optimization
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_best)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"\n🎯 Threshold Optimization:")
print(f"   - Optimal threshold: {optimal_threshold:.4f}")
print(f"   - F1-score at optimal threshold: {f1_scores[optimal_threshold_idx]:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_pred_proba_best >= optimal_threshold).astype(int)
optimal_accuracy = accuracy_score(y_test, y_pred_optimal)
optimal_precision = precision_score(y_test, y_pred_optimal)
optimal_recall = recall_score(y_test, y_pred_optimal)
optimal_f1 = f1_score(y_test, y_pred_optimal)

print(f"\n📊 Performance with Optimal Threshold:")
print(f"   - Accuracy: {optimal_accuracy:.4f}")
print(f"   - Precision: {optimal_precision:.4f}")
print(f"   - Recall: {optimal_recall:.4f}")
print(f"   - F1-Score: {optimal_f1:.4f}")

# Create confusion matrix visualization
print(f"\n🎨 Creating Confusion Matrix Visualization...")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Count'})

plt.title(f'Confusion Matrix: {best_nn_name}', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
plt.tight_layout()

# Save the plot
plt.savefig('neural_network_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved as 'neural_network_confusion_matrix.png'")

# Model comparison table
print(f"\n🏆 Neural Network Comparison:")
print("=" * 80)
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
print("-" * 80)

for name, scores in results.items():
    marker = " 🏆" if name == best_nn_name else ""
    print(f"{name:<25} {scores['accuracy']:<10.4f} {scores['precision']:<10.4f} {scores['recall']:<10.4f} {scores['f1']:<10.4f} {scores['roc_auc']:<10.4f}{marker}")

# Save results
print(f"\n💾 Saving Results...")
import pandas as pd

# Create results DataFrame
results_data = []
for name, scores in results.items():
    results_data.append({
        'Model': name,
        'Accuracy': scores['accuracy'],
        'Precision': scores['precision'],
        'Recall': scores['recall'],
        'F1_Score': scores['f1'],
        'ROC_AUC': scores['roc_auc'],
        'Training_Time': scores['training_time'],
        'Iterations': scores['n_iterations'],
        'Final_Loss': scores['loss']
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('ROC_AUC', ascending=False)
results_df.to_csv('neural_network_results.csv', index=False)
print("✅ Results saved as 'neural_network_results.csv'")

# Neural network architecture details
print(f"\n🧠 Neural Network Architecture Details:")
print("=" * 50)
for name, model in models.items():
    print(f"\n{name}:")
    print(f"   - Hidden layers: {model.hidden_layer_sizes}")
    print(f"   - Activation: {model.activation}")
    print(f"   - Solver: {model.solver}")
    print(f"   - Alpha (L2 reg): {model.alpha}")
    print(f"   - Learning rate: {model.learning_rate_init}")
    print(f"   - Max iterations: {model.max_iter}")
    print(f"   - Early stopping: {model.early_stopping}")
    print(f"   - Class weight: {model.class_weight}")

print(f"\n🎉 Neural Network Analysis Completed!")
print(f"   - Total time: {time.time() - start_time:.2f} seconds")
print(f"   - Best model: {best_nn_name}")
print(f"   - Best ROC-AUC: {results[best_nn_name]['roc_auc']:.4f}")
print(f"   - Files created:")
print(f"     - neural_network_confusion_matrix.png")
print(f"     - neural_network_results.csv")

print(f"\n🎯 Key Insights:")
print("=" * 15)
if results[best_nn_name]['roc_auc'] > 0.8:
    print(f"   - Excellent neural network performance (ROC-AUC: {results[best_nn_name]['roc_auc']:.3f})")
elif results[best_nn_name]['roc_auc'] > 0.7:
    print(f"   - Good neural network performance (ROC-AUC: {results[best_nn_name]['roc_auc']:.3f})")
else:
    print(f"   - Neural network performance could be improved (ROC-AUC: {results[best_nn_name]['roc_auc']:.3f})")

print(f"   - Best architecture: {best_nn_name}")
print(f"   - Training iterations: {results[best_nn_name]['n_iterations']}")
print(f"   - Final loss: {results[best_nn_name]['loss']:.4f}")
