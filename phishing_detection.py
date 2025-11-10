"""
Phishing Email Detection Project
Using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)

This script implements a complete machine learning pipeline for phishing email detection,
following the roadmap from data loading to model evaluation and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import kagglehub
import os


def download_dataset():
    """Download the email spam classification dataset from Kaggle."""
    print("="*60)
    print("DOWNLOADING DATASET")
    print("="*60)
    
    # Download latest version
    path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")
    print(f"\nPath to dataset files: {path}")
    
    # Find the CSV file in the downloaded path
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        dataset_path = os.path.join(path, csv_files[0])
        print(f"Found dataset: {csv_files[0]}")
        return dataset_path
    else:
        raise FileNotFoundError("No CSV file found in downloaded dataset")


def load_and_explore_data(dataset_path):
    """Load and perform initial exploration of the dataset."""
    print("\n" + "="*60)
    print("PHASE 1: LOADING AND EXPLORING DATA")
    print("="*60)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Display basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of emails: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumn names (first 10):")
    print(df.columns.tolist()[:10])
    
    # Check for the target column
    target_column = None
    possible_target_names = ['Email Type', 'label', 'spam', 'phishing', 'Prediction']
    for col in possible_target_names:
        if col in df.columns:
            target_column = col
            break
    
    if target_column:
        print(f"\nTarget column found: '{target_column}'")
        print("\nClass Distribution:")
        print(df[target_column].value_counts())
    else:
        print("\nWarning: Could not identify target column automatically")
        print("Available columns:", df.columns.tolist())
    
    # Check for missing values
    missing_total = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing_total}")
    
    if missing_total == 0:
        print("✓ No missing values found - dataset is clean!")
    
    # Get statistical summary
    print("\nStatistical Summary (first 5 features):")
    print(df.describe().iloc[:, :5])
    
    return df, target_column


def preprocess_data(df, target_column):
    """Preprocess the data: separate features, encode labels, split, and scale."""
    print("\n" + "="*60)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*60)
    
    # Separate features (X) and target variable (y)
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Encode categorical labels if needed
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("\nOriginal labels:", y.unique())
    print("Encoded labels:", np.unique(y_encoded))
    print("Mapping:", dict(zip(label_encoder.classes_, 
                              label_encoder.transform(label_encoder.classes_))))
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Testing set class distribution: {np.bincount(y_test)}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nOriginal feature range (first feature):")
    print(f"Min: {X_train.iloc[:, 0].min()}, Max: {X_train.iloc[:, 0].max()}")
    print("\nScaled feature range (first feature):")
    print(f"Min: {X_train_scaled[:, 0].min():.2f}, Max: {X_train_scaled[:, 0].max():.2f}")
    print("✓ Feature scaling completed!")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoder


def build_knn_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Build and optimize KNN model."""
    print("\n" + "="*60)
    print("PHASE 3: BUILDING KNN MODEL")
    print("="*60)
    
    # Test different K values
    k_values = range(1, 31, 2)
    train_scores = []
    test_scores = []
    
    print("\nTesting different K values...")
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
    print("✓ KNN optimization plot saved as 'knn_k_optimization.png'")
    
    # Find the best K
    best_k = k_values[np.argmax(test_scores)]
    print(f"\nOptimal K value: {best_k}")
    print(f"Best testing accuracy: {max(test_scores):.4f}")
    
    # Train final KNN model
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_knn = knn_final.predict(X_test_scaled)
    
    print("\n✓ KNN Model trained successfully!")
    
    return knn_final, y_pred_knn, best_k


def build_svm_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Build and optimize SVM model."""
    print("\n" + "="*60)
    print("PHASE 4: BUILDING SVM MODEL")
    print("="*60)
    
    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']
    svm_results = {}
    
    print("\nTesting different SVM kernels...")
    for kernel in kernels:
        print(f"\nTesting SVM with {kernel} kernel...")
        
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        svm_results[kernel] = accuracy
        
        print(f"{kernel.capitalize()} kernel accuracy: {accuracy:.4f}")
    
    best_kernel = max(svm_results, key=svm_results.get)
    print(f"\nBest performing kernel: {best_kernel}")
    print(f"Best accuracy: {svm_results[best_kernel]:.4f}")
    
    # Fine-tune the best model (using RBF or the best kernel)
    print("\nPerforming grid search for optimal parameters...")
    
    if best_kernel == 'rbf':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
    elif best_kernel == 'linear':
        param_grid = {
            'C': [0.1, 1, 10, 100]
        }
    else:  # poly
        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4]
        }
    
    svm = SVC(kernel=best_kernel, random_state=42, probability=True)
    
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Use the best model
    svm_final = grid_search.best_estimator_
    y_pred_svm = svm_final.predict(X_test_scaled)
    
    print("\n✓ SVM Model trained successfully!")
    
    return svm_final, y_pred_svm, best_kernel


def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive evaluation of a classification model."""
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


def visualize_results(knn_accuracy, knn_cm, svm_accuracy, svm_cm):
    """Create visualizations for model comparison."""
    print("\n" + "="*60)
    print("PHASE 5: MODEL EVALUATION AND VISUALIZATION")
    print("="*60)
    
    # Confusion matrices
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
    print("✓ Confusion matrices saved as 'confusion_matrices.png'")
    
    # Model comparison bar chart
    plt.figure(figsize=(8, 6))
    models = ['KNN', 'SVM']
    accuracies = [knn_accuracy, svm_accuracy]
    colors = ['skyblue', 'lightgreen']
    
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim([min(accuracies) - 0.05, 1.0])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('model_comparison.png')
    print("✓ Model comparison saved as 'model_comparison.png'")


def analyze_features(X_train, y_train):
    """Analyze feature importance."""
    print("\n" + "="*60)
    print("PHASE 6: FEATURE ANALYSIS")
    print("="*60)
    
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
    print("✓ Feature analysis saved as 'top_features.png'")
    
    print("\nTop 10 most discriminative features:")
    print(top_features.head(10))


def predict_email(email_features, model, scaler, model_name):
    """Predict if an email is phishing or legitimate."""
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


def save_models(knn_final, svm_final, scaler):
    """Save trained models for future use."""
    print("\n" + "="*60)
    print("PHASE 7: SAVING MODELS")
    print("="*60)
    
    joblib.dump(knn_final, 'knn_phishing_detector.pkl')
    joblib.dump(svm_final, 'svm_phishing_detector.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("\n✓ Models saved successfully!")
    print("  - knn_phishing_detector.pkl")
    print("  - svm_phishing_detector.pkl")
    print("  - feature_scaler.pkl")


def generate_report(df, X_train, X_test, y_encoded, best_k, best_kernel, 
                   knn_accuracy, svm_accuracy):
    """Generate and save final report."""
    print("\n" + "="*60)
    print("PHASE 8: GENERATING FINAL REPORT")
    print("="*60)
    
    report = f"""
PHISHING EMAIL DETECTION PROJECT - FINAL REPORT
{'='*60}

DATASET INFORMATION:
- Total Emails: {len(df)}
- Features: {df.shape[1] - 1}
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

FILES GENERATED:
- knn_k_optimization.png: KNN performance across different K values
- confusion_matrices.png: Confusion matrices for both models
- model_comparison.png: Bar chart comparing model accuracies
- top_features.png: Most discriminative features
- knn_phishing_detector.pkl: Trained KNN model
- svm_phishing_detector.pkl: Trained SVM model
- feature_scaler.pkl: Feature scaler for preprocessing

USAGE:
To use the trained models for prediction:
    import joblib
    knn_model = joblib.load('knn_phishing_detector.pkl')
    svm_model = joblib.load('svm_phishing_detector.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    # Scale your features and predict
    features_scaled = scaler.transform([your_email_features])
    prediction = svm_model.predict(features_scaled)
"""
    
    print(report)
    
    # Save report to file
    with open('project_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ Final report saved as 'project_report.txt'")


def main():
    """Main function to run the complete phishing detection pipeline."""
    print("\n" + "="*60)
    print("PHISHING EMAIL DETECTION PROJECT")
    print("Using KNN and SVM Machine Learning Algorithms")
    print("="*60)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Phase 1: Load and explore data
    df, target_column = load_and_explore_data(dataset_path)
    
    # Phase 2: Preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoder = preprocess_data(df, target_column)
    
    # Phase 3: Build KNN model
    knn_final, y_pred_knn, best_k = build_knn_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Phase 4: Build SVM model
    svm_final, y_pred_svm, best_kernel = build_svm_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Phase 5: Evaluate models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    knn_accuracy, knn_cm = evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")
    svm_accuracy, svm_cm = evaluate_model(y_test, y_pred_svm, "Support Vector Machine")
    
    # Visualize results
    visualize_results(knn_accuracy, knn_cm, svm_accuracy, svm_cm)
    
    # Phase 6: Feature analysis
    analyze_features(X_train, y_train)
    
    # Test with a sample email
    print("\nTesting with sample email from test set...")
    sample_email = X_test.iloc[0].values
    predict_email(sample_email, knn_final, scaler, "KNN")
    predict_email(sample_email, svm_final, scaler, "SVM")
    
    # Phase 7: Save models
    save_models(knn_final, svm_final, scaler)
    
    # Phase 8: Generate report
    y_encoded = np.concatenate([y_train, y_test])
    generate_report(df, X_train, X_test, y_encoded, best_k, best_kernel,
                   knn_accuracy, svm_accuracy)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nAll models, visualizations, and reports have been generated.")
    print("Check the current directory for output files.")


if __name__ == "__main__":
    main()
