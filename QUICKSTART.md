# Quick Start Guide

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ML-Classification-Assignment.git
cd Assignment_2_Final
```

### 2. Create Virtual Environment (if needed)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (if starting fresh)
```bash
python3 train_models.py
```
This will:
- Load the bakery sales dataset
- Preprocess and feature engineer
- Train all 5 models
- Save models to `models/` directory
- Generate performance metrics

### 5. Run Streamlit Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## Project Files Overview

| File | Purpose |
|------|---------|
| `train_models.py` | Model training and evaluation script |
| `streamlit_app.py` | Web application for predictions |
| `requirements.txt` | Python package dependencies |
| `README.md` | Comprehensive documentation |
| `test_data.csv` | Sample data for testing |
| `models/` | Trained model files (.joblib) |

---

## Using the Streamlit App

### Pages Available:

1. **📈 Model Performance**
   - View all model metrics
   - Compare accuracy, precision, recall, F1, AUC, MCC
   - Visual comparisons with charts

2. **🔮 Make Predictions**
   - Upload CSV data
   - Select model for predictions
   - Get confidence scores

3. **📋 Dataset Info**
   - Dataset statistics
   - Feature descriptions
   - Data samples

4. **📊 Comparison Analysis**
   - Detailed metric analysis
   - Model recommendations
   - Use-case suggestions

---

## Performance Results Summary

### Best Models by Metric:
- **Accuracy**: Logistic Regression & SVM (90.22%)
- **Precision**: Logistic Regression & SVM (90.22%)
- **Recall**: Logistic Regression & SVM (100%)
- **F1 Score**: Logistic Regression & SVM (0.9486)
- **AUC**: Decision Tree (0.6637)
- **MCC**: Decision Tree (0.1464)

---

## Making Predictions with Python

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/support_vector_machine.joblib')
scaler = joblib.load('models/scaler.joblib')
feature_cols = joblib.load('models/feature_columns.joblib')

# Load your data
data = pd.read_csv('your_data.csv')

# Preprocess
X = data[feature_cols].fillna(data[feature_cols].mean())
X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]
```

---

## Deployment to Streamlit Cloud

1. Create GitHub repository with these files
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click

**Repository Structure for Streamlit Cloud:**
```
repo/
├── streamlit_app.py (or app.py)
├── requirements.txt
├── models/
│   ├── *.joblib files
├── test_data.csv
└── README.md
```

---

## Troubleshooting

### Models Not Loading?
- Check that `models/` directory exists with all .joblib files
- Verify file permissions: `chmod 644 models/*.joblib`

### Data Upload Errors?
- Ensure CSV has required columns matching training data
- Check for missing values and NaN entries

### Slow Predictions?
- SVM can be slow on large datasets
- Use Logistic Regression for real-time applications

### Streamlit Port Already in Use?
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## Dataset Notes

- **Dataset**: Bakery Sales (19,321 records, 40 features)
- **Target**: Customer_Rating (converted to binary: High/Low)
- **Features Used**: 28 (19 numerical + 9 categorical encoded)
- **Train/Test Split**: 80/20 with stratification

---

## Performance Summary

All models achieve >88% accuracy on the test set:

| Model | Accuracy | Best At |
|-------|----------|---------|
| Logistic Regression | 90.22% | Accuracy, Recall |
| Decision Tree | 89.94% | AUC, Interpretability |
| K-Nearest Neighbors | 89.39% | Non-parametric learning |
| Naive Bayes | 88.80% | Fast inference |
| SVM | 90.22% | Balanced performance |

---

## Next Steps

1. ✅ Train models: `python3 train_models.py`
2. ✅ Run app: `streamlit run streamlit_app.py`
3. ✅ Upload to GitHub
4. ✅ Deploy to Streamlit Cloud
5. ✅ Share links for submission

---

For questions or issues, refer to README.md for detailed documentation.
