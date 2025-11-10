# Quick Start Guide

Get up and running with the Phishing Email Detection system in just a few minutes!

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### Step 2: Configure Kaggle API (2 minutes)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token" (downloads `kaggle.json`)
4. Move the file to the correct location:

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### Step 3: Run the Pipeline (2 minutes)

```bash
python phishing_detection.py
```

That's it! The script will automatically:
- ✓ Download the dataset from Kaggle
- ✓ Train both KNN and SVM models
- ✓ Generate visualizations
- ✓ Save trained models
- ✓ Create a comprehensive report

## 📁 What Gets Generated

After running, you'll find these files:

### Models (for predictions)
- `knn_phishing_detector.pkl` - KNN classifier
- `svm_phishing_detector.pkl` - SVM classifier  
- `feature_scaler.pkl` - Feature preprocessor

### Visualizations (to understand performance)
- `knn_k_optimization.png` - Shows optimal K value
- `confusion_matrices.png` - Model accuracy breakdown
- `model_comparison.png` - KNN vs SVM performance
- `top_features.png` - Most important features

### Report
- `project_report.txt` - Complete analysis and results

## 🔮 Making Predictions

Once trained, use the models to classify new emails:

```python
import joblib
import numpy as np

# Load models (one time)
svm_model = joblib.load('svm_phishing_detector.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Classify an email
email_features = np.array([...])  # Your email's word frequencies
email_scaled = scaler.transform([email_features])
prediction = svm_model.predict(email_scaled)[0]

print("Phishing!" if prediction == 1 else "Safe")
```

Or use the example script:

```bash
python example_usage.py
```

## 🧪 Testing Without Kaggle

Want to test the system without Kaggle setup?

```bash
python test_pipeline.py
```

This creates synthetic data and verifies all components work correctly.

## ⚡ Common Issues

**Issue**: `OSError: Could not find kaggle.json`
- **Fix**: Complete Step 2 above to set up Kaggle API credentials

**Issue**: `ImportError: No module named 'sklearn'`
- **Fix**: Run `pip install -r requirements.txt`

**Issue**: Training is slow
- **Fix**: This is normal for the first run. SVM grid search with cross-validation takes 2-5 minutes depending on your CPU.

## 📊 Expected Results

Typical performance on the real dataset:
- **Training time**: 3-7 minutes (depending on hardware)
- **KNN accuracy**: 92-96%
- **SVM accuracy**: 93-97%
- **Dataset size**: 5,172 emails with 3,000 features each

## 🎯 Next Steps

1. ✅ **Run the pipeline** - See it in action with real data
2. 📊 **Review visualizations** - Understand model performance
3. 🔮 **Make predictions** - Use trained models on new emails
4. 🔬 **Experiment** - Try different parameters, features, or algorithms

## 📚 Full Documentation

For detailed information, see [README.md](README.md)

For the complete project roadmap, see [phishing_detection_roadmap.md](phishing_detection_roadmap.md)

---

**Need help?** Open an issue on GitHub!
