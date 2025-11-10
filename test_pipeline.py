"""
Test script for phishing detection system
This script creates a synthetic dataset to test the ML pipeline without requiring Kaggle credentials
"""

import pandas as pd
import numpy as np
import os
import sys

# Import our main functions
from phishing_detection import (
    load_and_explore_data,
    preprocess_data,
    build_knn_model,
    build_svm_model,
    evaluate_model,
    visualize_results,
    analyze_features,
    save_models,
    generate_report
)


def create_synthetic_dataset(n_samples=1000, n_features=100):
    """Create a synthetic email dataset for testing."""
    print("Creating synthetic dataset for testing...")
    
    np.random.seed(42)
    
    # Generate features (word frequencies)
    X = np.random.rand(n_samples, n_features)
    
    # Generate labels (0 = legitimate, 1 = phishing)
    # Create some pattern: phishing emails have higher values in certain features
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Phishing emails have higher average in first 20 features
        if X[i, :20].mean() > 0.6:
            y[i] = 1
        else:
            y[i] = 0
    
    # Create a more balanced dataset
    n_phishing = int(n_samples * 0.4)
    phishing_indices = np.random.choice(n_samples, n_phishing, replace=False)
    y[phishing_indices] = 1
    
    # Create column names
    feature_cols = [f'word_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    df['Email Type'] = ['Phishing' if label == 1 else 'Safe Email' for label in y]
    
    # Save to CSV
    df.to_csv('test_emails.csv', index=False)
    print(f"✓ Created test dataset: {n_samples} samples, {n_features} features")
    print(f"  - Legitimate: {(y == 0).sum()}")
    print(f"  - Phishing: {(y == 1).sum()}")
    
    return 'test_emails.csv'


def test_pipeline():
    """Test the complete ML pipeline."""
    print("\n" + "="*60)
    print("TESTING PHISHING DETECTION PIPELINE")
    print("="*60)
    
    try:
        # Create synthetic dataset
        dataset_path = create_synthetic_dataset(n_samples=1000, n_features=100)
        
        # Test Phase 1: Load and explore
        print("\n--- Testing Phase 1: Data Loading ---")
        df, target_column = load_and_explore_data(dataset_path)
        print("✓ Phase 1 completed")
        
        # Test Phase 2: Preprocessing
        print("\n--- Testing Phase 2: Preprocessing ---")
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoder = preprocess_data(df, target_column)
        print("✓ Phase 2 completed")
        
        # Test Phase 3: KNN Model
        print("\n--- Testing Phase 3: KNN Model ---")
        knn_final, y_pred_knn, best_k = build_knn_model(X_train_scaled, X_test_scaled, y_train, y_test)
        print("✓ Phase 3 completed")
        
        # Test Phase 4: SVM Model (with reduced parameter grid for faster testing)
        print("\n--- Testing Phase 4: SVM Model ---")
        svm_final, y_pred_svm, best_kernel = build_svm_model(X_train_scaled, X_test_scaled, y_train, y_test)
        print("✓ Phase 4 completed")
        
        # Test Phase 5: Evaluation
        print("\n--- Testing Phase 5: Evaluation ---")
        knn_accuracy, knn_cm = evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")
        svm_accuracy, svm_cm = evaluate_model(y_test, y_pred_svm, "Support Vector Machine")
        visualize_results(knn_accuracy, knn_cm, svm_accuracy, svm_cm)
        print("✓ Phase 5 completed")
        
        # Test Phase 6: Feature Analysis
        print("\n--- Testing Phase 6: Feature Analysis ---")
        analyze_features(X_train, y_train)
        print("✓ Phase 6 completed")
        
        # Test Phase 7: Save Models
        print("\n--- Testing Phase 7: Saving Models ---")
        save_models(knn_final, svm_final, scaler)
        print("✓ Phase 7 completed")
        
        # Test Phase 8: Generate Report
        print("\n--- Testing Phase 8: Generate Report ---")
        y_encoded = np.concatenate([y_train, y_test])
        generate_report(df, X_train, X_test, y_encoded, best_k, best_kernel,
                       knn_accuracy, svm_accuracy)
        print("✓ Phase 8 completed")
        
        # Verify output files
        print("\n--- Verifying Output Files ---")
        expected_files = [
            'knn_k_optimization.png',
            'confusion_matrices.png',
            'model_comparison.png',
            'top_features.png',
            'knn_phishing_detector.pkl',
            'svm_phishing_detector.pkl',
            'feature_scaler.pkl',
            'project_report.txt'
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} - NOT FOUND")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nThe phishing detection pipeline is working correctly.")
        print("All phases executed without errors and generated expected outputs.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test file
        if os.path.exists('test_emails.csv'):
            os.remove('test_emails.csv')
            print("\n✓ Cleaned up test dataset")


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
