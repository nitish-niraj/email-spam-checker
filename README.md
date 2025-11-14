# Phishing Email Detection Project

A complete machine learning system to classify emails as phishing or legitimate using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms.

## 🚀 Run on Google Colab

**You can now run this project directly in your browser without any installation!**

### Complete All-in-One Notebook (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitish-niraj/email-spam-checker/blob/main/Complete_Email_Spam_Detection.ipynb)

**New!** This comprehensive notebook includes:
- ✅ All dependencies pre-configured
- ✅ Complete ML pipeline from start to finish
- ✅ Works with uploaded emails.csv file (no Kaggle API needed!)
- ✅ Interactive visualizations and detailed analysis
- ✅ **Automatically saves all trained models at the end**
- ✅ Includes usage examples for predictions

### Alternative: Dataset Auto-Download Version
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitish-niraj/email-spam-checker/blob/main/phishing_detection_colab.ipynb)

This version automatically downloads the dataset from Kaggle (requires Kaggle API credentials).

**Perfect for:** Quick testing, learning, demonstrations, or if you don't want to set up a local environment.

## 📋 Project Overview

This project implements a comprehensive phishing email detection system that:
- Processes a dataset of 5,172 emails with 3,000 pre-extracted word frequency features
- Trains and compares two machine learning models (KNN and SVM)
- Provides detailed evaluation metrics and visualizations
- Saves trained models for future use

## 🚀 Features

- **Automated Dataset Download**: Uses Kaggle API to download the email spam classification dataset
- **Complete ML Pipeline**: From data loading to model deployment
- **Model Comparison**: Side-by-side comparison of KNN and SVM performance
- **Hyperparameter Optimization**: Grid search for optimal model parameters
- **Comprehensive Visualizations**: 
  - KNN K-value optimization plots
  - Confusion matrices for both models
  - Model performance comparison charts
  - Feature importance analysis
- **Model Persistence**: Saves trained models for future predictions

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nitish-niraj/email-spam-checker.git
cd email-spam-checker
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Kaggle API (required for dataset download):
   - Go to your Kaggle account settings
   - Create a new API token (downloads `kaggle.json`)
   - Place the file in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

## 🎯 Usage

### Option 1: Google Colab (Recommended for Quick Start)

#### Using the Complete Notebook (Easiest)
1. Click the "Complete All-in-One Notebook" badge above
2. Upload the `emails.csv` file to your Colab session (or use the file upload widget in the notebook)
3. Run all cells to train and evaluate the models
4. The notebook will automatically save all trained models at the end
5. Download the trained models and visualizations

#### Using the Auto-Download Version
1. Click the "Alternative: Dataset Auto-Download Version" badge
2. Follow the notebook instructions to upload your Kaggle API credentials
3. Run all cells - the dataset will be downloaded automatically
4. Download the trained models and visualizations

### Option 2: Running Locally

Simply run the main script:

```bash
python phishing_detection.py
```

This will:
1. Download the dataset from Kaggle
2. Load and explore the data
3. Preprocess and scale features
4. Train KNN model with optimal K value
5. Train SVM model with best kernel and parameters
6. Evaluate both models
7. Generate visualizations
8. Save trained models
9. Create a comprehensive report

### Using Trained Models

After training, you can load and use the models for predictions:

```python
import joblib
import numpy as np

# Load the trained models
knn_model = joblib.load('knn_phishing_detector.pkl')
svm_model = joblib.load('svm_phishing_detector.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Prepare your email features (3000-dimensional vector)
email_features = np.array([...])  # Your email word frequencies

# Scale the features
email_scaled = scaler.transform([email_features])

# Make predictions
knn_prediction = knn_model.predict(email_scaled)
svm_prediction = svm_model.predict(email_scaled)

# Get prediction probabilities (SVM)
svm_probabilities = svm_model.predict_proba(email_scaled)

print("KNN Prediction:", "Phishing" if knn_prediction[0] == 1 else "Legitimate")
print("SVM Prediction:", "Phishing" if svm_prediction[0] == 1 else "Legitimate")
print(f"SVM Confidence: {svm_probabilities[0][1]:.2%}")
```

## 📊 Output Files

After running the script, the following files will be generated:

### Models
- `knn_phishing_detector.pkl` - Trained KNN model
- `svm_phishing_detector.pkl` - Trained SVM model
- `feature_scaler.pkl` - StandardScaler for feature preprocessing
- `label_encoder.pkl` - LabelEncoder for target variable encoding (saved by the Complete notebook)

### Visualizations
- `knn_k_optimization.png` - KNN performance across different K values
- `confusion_matrices.png` - Confusion matrices for both models
- `model_comparison.png` - Bar chart comparing model accuracies
- `top_features.png` - Top 20 most discriminative features

### Reports
- `project_report.txt` - Comprehensive project report with all metrics and insights

## 🔬 Project Phases

### Phase 1: Environment Setup and Data Understanding
- Dataset download and loading
- Exploratory data analysis
- Data structure examination

### Phase 2: Data Preprocessing
- Feature and target separation
- Label encoding
- Train-test split (80-20)
- Feature scaling with StandardScaler

### Phase 3: KNN Model Building
- K-value optimization (testing K from 1 to 30)
- Training final KNN model with optimal K
- Performance evaluation

### Phase 4: SVM Model Building
- Kernel comparison (linear, RBF, polynomial)
- Grid search for hyperparameter tuning
- Training final SVM model

### Phase 5: Model Evaluation
- Accuracy, precision, recall, F1-score calculation
- Confusion matrix generation
- Model comparison

### Phase 6: Feature Analysis
- Discriminative feature identification
- Feature importance visualization

### Phase 7: Model Persistence
- Saving trained models and scaler

### Phase 8: Documentation
- Final report generation

## 📈 Expected Performance

The models typically achieve:
- **KNN**: ~92-96% accuracy (depends on optimal K value)
- **SVM**: ~93-97% accuracy (with optimized hyperparameters)

Performance may vary based on the specific dataset characteristics.

## 🛠️ Troubleshooting

### Issue: Kaggle API Authentication Error
**Solution**: Ensure your `kaggle.json` file is properly placed and has correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json  # On Linux/Mac
```

### Issue: Memory Error During Training
**Solution**: If you encounter memory issues with large datasets:
- Reduce the parameter grid size for SVM grid search
- Use a smaller subset of the data for initial testing
- Increase system swap space

### Issue: Slow Training
**Solution**: 
- Ensure `n_jobs=-1` is set in GridSearchCV to use all CPU cores
- Consider using a smaller K range for KNN optimization
- For SVM, use fewer C and gamma values in the grid

## 📚 Dataset

This project uses the [Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv) from Kaggle, which contains:
- 5,172 emails
- 3,000 word frequency features
- Binary classification (spam/legitimate)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Nitish Niraj

## 🙏 Acknowledgments

- Dataset provided by Balaka Biswas on Kaggle
- scikit-learn library for machine learning algorithms
- The open-source community for excellent documentation and examples

## 📮 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational purposes and demonstrates machine learning classification techniques. For production use, additional validation, testing, and security considerations would be necessary.
