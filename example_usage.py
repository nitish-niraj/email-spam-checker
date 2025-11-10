"""
Example: Using Pre-trained Models for Email Classification

This script demonstrates how to load and use the trained phishing detection models
to classify new emails.
"""

import joblib
import numpy as np
import os


def load_models():
    """Load the pre-trained models and scaler."""
    print("Loading pre-trained models...")
    
    if not os.path.exists('knn_phishing_detector.pkl'):
        print("Error: Models not found. Please run phishing_detection.py first to train the models.")
        return None, None, None
    
    knn_model = joblib.load('knn_phishing_detector.pkl')
    svm_model = joblib.load('svm_phishing_detector.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    print("✓ Models loaded successfully!")
    return knn_model, svm_model, scaler


def predict_single_email(email_features, knn_model, svm_model, scaler):
    """
    Predict if a single email is phishing or legitimate.
    
    Parameters:
    -----------
    email_features : array-like
        Word frequency features of the email (should match training feature count)
    knn_model : sklearn model
        Trained KNN model
    svm_model : sklearn model
        Trained SVM model
    scaler : sklearn StandardScaler
        Fitted scaler for feature normalization
    
    Returns:
    --------
    dict : Dictionary containing predictions and confidence scores
    """
    # Validate input
    email_features = np.array(email_features).reshape(1, -1)
    
    # Scale the features
    email_scaled = scaler.transform(email_features)
    
    # Get predictions from both models
    knn_prediction = knn_model.predict(email_scaled)[0]
    svm_prediction = svm_model.predict(email_scaled)[0]
    
    # Get probability estimates
    knn_proba = knn_model.predict_proba(email_scaled)[0] if hasattr(knn_model, 'predict_proba') else None
    svm_proba = svm_model.predict_proba(email_scaled)[0] if hasattr(svm_model, 'predict_proba') else None
    
    # Prepare results
    results = {
        'knn_prediction': 'Phishing' if knn_prediction == 1 else 'Legitimate',
        'svm_prediction': 'Phishing' if svm_prediction == 1 else 'Legitimate',
        'knn_confidence': knn_proba,
        'svm_confidence': svm_proba,
        'consensus': 'Phishing' if (knn_prediction + svm_prediction) >= 1 else 'Legitimate'
    }
    
    return results


def display_prediction_results(results):
    """Display prediction results in a formatted way."""
    print("\n" + "="*60)
    print("EMAIL CLASSIFICATION RESULTS")
    print("="*60)
    
    print(f"\nKNN Prediction: {results['knn_prediction']}")
    if results['knn_confidence'] is not None:
        print(f"  Confidence: Legitimate: {results['knn_confidence'][0]:.2%}, "
              f"Phishing: {results['knn_confidence'][1]:.2%}")
    
    print(f"\nSVM Prediction: {results['svm_prediction']}")
    if results['svm_confidence'] is not None:
        print(f"  Confidence: Legitimate: {results['svm_confidence'][0]:.2%}, "
              f"Phishing: {results['svm_confidence'][1]:.2%}")
    
    print(f"\nConsensus Prediction: {results['consensus']}")
    print("="*60)


def example_usage():
    """Example usage of the prediction system."""
    print("\n" + "="*60)
    print("PHISHING EMAIL DETECTION - EXAMPLE USAGE")
    print("="*60)
    
    # Load models
    knn_model, svm_model, scaler = load_models()
    
    if knn_model is None:
        return
    
    # Example 1: Create a random email feature vector
    # In a real scenario, these would be extracted word frequencies from an email
    print("\n\nExample 1: Classifying a random email...")
    print("-" * 60)
    
    # Generate random features (replace with actual email features in production)
    np.random.seed(42)
    n_features = scaler.n_features_in_
    random_email = np.random.rand(n_features)
    
    print(f"Email features: {n_features} word frequency values")
    print(f"First 10 features: {random_email[:10].round(3)}")
    
    results = predict_single_email(random_email, knn_model, svm_model, scaler)
    display_prediction_results(results)
    
    # Example 2: Batch prediction
    print("\n\nExample 2: Batch classification of multiple emails...")
    print("-" * 60)
    
    n_emails = 5
    batch_emails = np.random.rand(n_emails, n_features)
    
    for i, email_features in enumerate(batch_emails, 1):
        results = predict_single_email(email_features, knn_model, svm_model, scaler)
        print(f"\nEmail {i}: KNN={results['knn_prediction']}, "
              f"SVM={results['svm_prediction']}, "
              f"Consensus={results['consensus']}")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("""
To use this with real email data:

1. Extract word frequencies from your email
   (this depends on your feature extraction pipeline)

2. Create a feature vector with the same number of features
   as the training data

3. Call predict_single_email() with your features

Example:
    # Your email features (e.g., from TF-IDF, word counts, etc.)
    my_email_features = extract_features(email_text)
    
    # Load models
    knn, svm, scaler = load_models()
    
    # Predict
    results = predict_single_email(my_email_features, knn, svm, scaler)
    
    # Display
    display_prediction_results(results)
""")


if __name__ == "__main__":
    example_usage()
