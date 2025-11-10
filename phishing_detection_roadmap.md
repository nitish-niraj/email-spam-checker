# Phishing Email Detection Project Roadmap

## Project Overview

You'll build a machine learning system to classify emails as phishing or legitimate using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms. Your dataset contains 5,172 emails with 3,000 pre-extracted word frequency features.

---

## Phase 1: Environment Setup and Data Understanding

### Step 1.1: Set Up Your Development Environment

Before you begin coding, you need to prepare your Python environment with the necessary libraries. Think of this as gathering all your tools before starting a construction project.

Install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Here's what each library does:
- **pandas**: Handles your dataset like a spreadsheet, making it easy to read and manipulate data
- **numpy**: Performs mathematical operations efficiently on large arrays of numbers
- **scikit-learn**: Contains the machine learning algorithms (KNN, SVM) and evaluation tools
- **matplotlib & seaborn**: Create visualizations to understand your data and results

### Step 1.2: Load and Explore Your Dataset

Create a new Python file called `phishing_detection.py` and start by loading your data:

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('emails.csv')

# Display basic information about your dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for the target column (it might be named 'label', 'spam', or 'phishing')
print("\nColumn names:")
print(df.columns.tolist())
```

Understanding what you see here is crucial. The shape tells you how many emails (rows) and features (columns) you have. The column names will reveal which column contains your labels (phishing vs legitimate).

### Step 1.3: Understand Your Data Structure

Examine the composition of your dataset:

```python
# Check the distribution of phishing vs legitimate emails
print("\nClass Distribution:")
print(df['Email Type'].value_counts())  # Adjust column name as needed

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Get statistical summary of features
print("\nStatistical Summary:")
print(df.describe())
```

This step is like taking inventory. You need to know if your dataset is balanced (equal phishing and legitimate emails) or imbalanced, and whether there are any missing values that need handling.

---

## Phase 2: Data Preprocessing

### Step 2.1: Separate Features and Target Variable

Machine learning models need clear inputs (features) and outputs (labels). Think of features as the characteristics the model uses to make decisions, and the target as what you want to predict.

```python
# Separate features (X) and target variable (y)
# Assuming the target column is named 'Email Type' or similar
X = df.drop(['Email Type'], axis=1)  # All columns except the target
y = df['Email Type']  # The target column

print("Features shape:", X.shape)
print("Target shape:", y.shape)
```

### Step 2.2: Encode Categorical Labels

If your labels are text (like "phishing" and "legitimate"), you need to convert them to numbers because machine learning algorithms work with numerical data.

```python
from sklearn.preprocessing import LabelEncoder

# Convert text labels to numbers (0 and 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nOriginal labels:", y.unique())
print("Encoded labels:", np.unique(y_encoded))
print("Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
```

### Step 2.3: Split Data into Training and Testing Sets

You need to divide your data so you can train your models on one portion and test them on another. This is like practicing for an exam with practice questions, then taking a real exam with different questions to see if you truly learned.

```python
from sklearn.model_selection import train_test_split

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y_encoded  # Maintains class distribution in both sets
)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])
print("Training set class distribution:", np.bincount(y_train))
print("Testing set class distribution:", np.bincount(y_test))
```

### Step 2.4: Feature Scaling

Feature scaling is critical for both KNN and SVM. Imagine comparing distances where one feature is in kilometers and another in millimeters – the algorithm would be biased toward the feature with larger numbers. Scaling puts all features on the same scale.

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nOriginal feature range (first feature):")
print(f"Min: {X_train.iloc[:, 0].min()}, Max: {X_train.iloc[:, 0].max()}")
print("\nScaled feature range (first feature):")
print(f"Min: {X_train_scaled[:, 0].min():.2f}, Max: {X_train_scaled[:, 0].max():.2f}")
```

---

## Phase 3: Building the KNN Model

### Step 3.1: Understanding K-Nearest Neighbors

KNN is like asking your neighbors for advice. When classifying a new email, KNN looks at the K most similar emails in your training data and predicts the class based on majority vote. If most of your "neighbors" are phishing emails, the new email is likely phishing too.

### Step 3.2: Find the Optimal K Value

The value of K (number of neighbors) significantly affects performance. Too small, and the model is sensitive to noise; too large, and it becomes too general.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Test different K values
k_values = range(1, 31, 2)  # Test odd numbers from 1 to 30
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Training Accuracy', marker='o')
plt.plot(k_values, test_scores, label='Testing Accuracy', marker='s')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs K Value')
plt.legend()
plt.grid(True)
plt.savefig('knn_k_optimization.png')
plt.show()

# Find the best K
best_k = k_values[np.argmax(test_scores)]
print(f"\nOptimal K value: {best_k}")
print(f"Best testing accuracy: {max(test_scores):.4f}")
```

### Step 3.3: Train Final KNN Model

Now train your final KNN model with the optimal K value:

```python
# Train the final KNN model
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn_final.predict(X_test_scaled)

print("\nKNN Model trained successfully!")
```

---

## Phase 4: Building the SVM Model

### Step 4.1: Understanding Support Vector Machines

SVM works differently from KNN. Instead of looking at neighbors, SVM finds the best boundary (called a hyperplane) that separates phishing from legitimate emails. Think of it as drawing the clearest line possible between two groups of points, maximizing the distance from the line to the nearest points on either side.

### Step 4.2: Choose and Test SVM Kernels

SVMs can use different "kernels" to handle complex patterns. The kernel transforms your data to make it easier to separate:

- **Linear kernel**: Works well when classes are already separable by a straight line
- **RBF (Radial Basis Function)**: Can handle more complex, non-linear patterns
- **Polynomial**: Good for curved decision boundaries

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Test different kernels
kernels = ['linear', 'rbf', 'poly']
svm_results = {}

for kernel in kernels:
    print(f"\nTesting SVM with {kernel} kernel...")
    
    # Create and train the SVM
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    svm_results[kernel] = accuracy
    
    print(f"{kernel.capitalize()} kernel accuracy: {accuracy:.4f}")

# Find the best kernel
best_kernel = max(svm_results, key=svm_results.get)
print(f"\nBest performing kernel: {best_kernel}")
print(f"Best accuracy: {svm_results[best_kernel]:.4f}")
```

### Step 4.3: Fine-tune the Best SVM Model

For the RBF kernel (which is often best), you can optimize two important parameters:
- **C**: Controls the trade-off between a smooth decision boundary and classifying training points correctly
- **gamma**: Defines how far the influence of a single training example reaches

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Create SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    svm_rbf, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available processors
    verbose=1
)

print("\nPerforming grid search for optimal parameters...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model
svm_final = grid_search.best_estimator_
y_pred_svm = svm_final.predict(X_test_scaled)
```

---

## Phase 5: Model Evaluation and Comparison

### Step 5.1: Calculate Comprehensive Metrics

Accuracy alone doesn't tell the full story. In phishing detection, you want to know:
- **Precision**: When the model says "phishing," how often is it correct?
- **Recall**: Of all actual phishing emails, how many did the model catch?
- **F1-Score**: The harmonic mean of precision and recall

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_true, y_pred, model_name):
    """
    Comprehensive evaluation of a classification model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Legitimate', 'Phishing']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives (Correct Legitimate): {cm[0][0]}")
    print(f"False Positives (Legitimate marked as Phishing): {cm[0][1]}")
    print(f"False Negatives (Phishing marked as Legitimate): {cm[1][0]}")
    print(f"True Positives (Correct Phishing): {cm[1][1]}")
    
    return accuracy, cm

# Evaluate both models
knn_accuracy, knn_cm = evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")
svm_accuracy, svm_cm = evaluate_model(y_test, y_pred_svm, "Support Vector Machine")
```

### Step 5.2: Visualize Confusion Matrices

Confusion matrices are easier to understand visually:

```python
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KNN Confusion Matrix
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            ax=axes[0])
axes[0].set_title(f'KNN Confusion Matrix\nAccuracy: {knn_accuracy:.4f}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            ax=axes[1])
axes[1].set_title(f'SVM Confusion Matrix\nAccuracy: {svm_accuracy:.4f}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
```

### Step 5.3: Compare Model Performance

Create a comprehensive comparison:

```python
# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['KNN', 'SVM'],
    'Accuracy': [knn_accuracy, svm_accuracy],
    'Training Time': ['Fast', 'Moderate'],
    'Prediction Speed': ['Slow for large datasets', 'Fast'],
    'Interpretability': ['High', 'Moderate']
})

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(comparison.to_string(index=False))

# Visualize accuracy comparison
plt.figure(figsize=(8, 6))
models = ['KNN', 'SVM']
accuracies = [knn_accuracy, svm_accuracy]
colors = ['skyblue', 'lightgreen']

bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim([0.8, 1.0])  # Adjust based on your results

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.savefig('model_comparison.png')
plt.show()
```

---

## Phase 6: Model Interpretation and Feature Analysis

### Step 6.1: Analyze Feature Importance (for interpretation)

While KNN and SVM don't directly provide feature importance, you can analyze which features vary most between classes:

```python
# Calculate mean feature values for each class
legitimate_emails = X_train[y_train == 0]
phishing_emails = X_train[y_train == 1]

feature_diff = abs(legitimate_emails.mean() - phishing_emails.mean())
top_features = feature_diff.nlargest(20)

plt.figure(figsize=(10, 8))
top_features.plot(kind='barh')
plt.xlabel('Absolute Difference in Mean Values')
plt.title('Top 20 Discriminative Features')
plt.tight_layout()
plt.savefig('top_features.png')
plt.show()

print("\nTop 10 most discriminative features:")
print(top_features.head(10))
```

### Step 6.2: Test with New Email Examples

Create a function to classify new emails:

```python
def predict_email(email_features, model, model_name):
    """
    Predict if an email is phishing or legitimate
    
    Parameters:
    email_features: array-like, the word frequency features of the email
    model: trained classifier
    model_name: string, name of the model for display
    """
    # Scale the features
    email_scaled = scaler.transform([email_features])
    
    # Make prediction
    prediction = model.predict(email_scaled)[0]
    probability = model.predict_proba(email_scaled)[0] if hasattr(model, 'predict_proba') else None
    
    # Display result
    result = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
    print(f"\n{model_name} Classification: {result}")
    
    if probability is not None:
        print(f"Confidence: Legitimate: {probability[0]:.2%}, Phishing: {probability[1]:.2%}")
    
    return prediction

# Example: Test with a sample from your test set
sample_email = X_test.iloc[0].values
print("Testing sample email...")
predict_email(sample_email, knn_final, "KNN")
predict_email(sample_email, svm_final, "SVM")
```

---

## Phase 7: Save Your Models

### Step 7.1: Save Trained Models for Future Use

Once you've trained your models, you'll want to save them so you don't have to retrain every time:

```python
import joblib

# Save models
joblib.dump(knn_final, 'knn_phishing_detector.pkl')
joblib.dump(svm_final, 'svm_phishing_detector.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print("\nModels saved successfully!")

# Later, you can load them like this:
# knn_loaded = joblib.load('knn_phishing_detector.pkl')
# svm_loaded = joblib.load('svm_phishing_detector.pkl')
# scaler_loaded = joblib.load('feature_scaler.pkl')
```

---

## Phase 8: Documentation and Reporting

### Step 8.1: Create a Final Report

Document your findings in a structured format:

```python
# Generate final report
report = f"""
PHISHING EMAIL DETECTION PROJECT - FINAL REPORT
{'='*60}

DATASET INFORMATION:
- Total Emails: {len(df)}
- Features: {X.shape[1]}
- Training Samples: {len(X_train)}
- Testing Samples: {len(X_test)}

MODEL PERFORMANCE:

1. K-Nearest Neighbors (K={best_k})
   - Accuracy: {knn_accuracy:.4f}
   - Strengths: Simple, no training time, interpretable
   - Weaknesses: Slow predictions, sensitive to irrelevant features

2. Support Vector Machine ({best_kernel} kernel)
   - Accuracy: {svm_accuracy:.4f}
   - Strengths: Effective in high dimensions, fast predictions
   - Weaknesses: Longer training time, requires parameter tuning

RECOMMENDATION:
{'SVM performs better' if svm_accuracy > knn_accuracy else 'KNN performs better'} 
for this phishing detection task with a 
{abs(svm_accuracy - knn_accuracy)*100:.2f}% accuracy difference.

KEY INSIGHTS:
- Both models show {('strong' if min(knn_accuracy, svm_accuracy) > 0.9 else 'moderate')} 
  performance in detecting phishing emails
- Feature scaling was crucial for model performance
- The dataset appears to be {'balanced' if abs(np.bincount(y_encoded)[0] - np.bincount(y_encoded)[1]) < len(y_encoded)*0.1 else 'imbalanced'}
"""

print(report)

# Save report to file
with open('project_report.txt', 'w') as f:
    f.write(report)
```

---

## Common Issues and Troubleshooting

### Issue 1: Low Accuracy
**Possible causes:**
- Features not scaled properly
- Inappropriate K value or SVM parameters
- Imbalanced dataset

**Solutions:**
- Verify StandardScaler was applied to both train and test sets
- Try different K values or SVM kernels
- Consider using class_weight='balanced' in SVM

### Issue 2: Overfitting
**Symptoms:** High training accuracy but low testing accuracy

**Solutions:**
- Increase K in KNN
- Reduce C parameter in SVM
- Use cross-validation to validate performance

### Issue 3: Slow Training
**Solutions:**
- Reduce dataset size for initial testing
- Use fewer features (feature selection)
- For SVM, use a smaller parameter grid

---

## Next Steps and Extensions

Once you've completed the basic project, consider these enhancements:

1. **Feature Selection**: Use techniques like Recursive Feature Elimination to identify the most important features and reduce dimensionality

2. **Ensemble Methods**: Combine KNN and SVM predictions using voting or stacking

3. **Cross-Validation**: Implement k-fold cross-validation for more robust performance estimates

4. **Real-time Detection**: Build a simple web interface where users can input email text and get instant classification

5. **Deep Learning**: Compare your results with a neural network approach

6. **Error Analysis**: Examine misclassified emails to understand model limitations

---

## Conclusion

By following this roadmap, you've built a complete phishing email detection system using two different machine learning approaches. You've learned how KNN makes predictions based on similarity to neighboring examples, while SVM finds optimal decision boundaries. Both approaches have their strengths, and comparing them gives you valuable insight into how different algorithms handle the same problem.

Remember that machine learning is iterative – don't be afraid to experiment with different parameters, feature selections, and preprocessing techniques to improve your results!
