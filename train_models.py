"""
ML Classification Models Training Script - Assignment 2
For Bakery Sales Dataset Classification
Student ID: 2026AK05229
BITS WILP M.Tech (AIML)

This script trains 5 classification models and evaluates them using
multiple metrics: Accuracy, Precision, Recall, F1, AUC, and MCC.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DATASET_PATH = '../Assignment_2/bakery_sales_dataset.csv'
MODEL_DIR = './models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== Data Loading & Preprocessing ====================
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Target Variable: Customer_Rating (will convert to binary classification)
# 1-2: Low Rating (0), 3-5: High Rating (1)
print("\nTarget Variable: Customer_Rating")
print(df['Customer_Rating'].value_counts().sort_index())

df['Rating_Category'] = (df['Customer_Rating'] >= 3).astype(int)
print("\nBinary Classification Target: Rating_Category")
print(df['Rating_Category'].value_counts())

# Feature Selection: Use numerical features + encoded categorical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('Customer_Rating')  # Remove old target
if 'Rating_Category' in numerical_cols:
    numerical_cols.remove('Rating_Category')  # Remove new target

print(f"\nNumerical Features ({len(numerical_cols)}): {numerical_cols[:5]}...")

# Select categorical features to encode
categorical_cols = ['Customer_Gender', 'Customer_Segment', 'Loyalty_Member', 
                   'Weekend', 'Weather', 'Payment_Method', 'Promotion_Applied',
                   'Storage_Type', 'Perishability']

# Encode categorical variables
df_processed = df.copy()
le_dict = {}

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le

feature_cols = numerical_cols + [col + '_encoded' for col in categorical_cols if col in df.columns]

X = df_processed[feature_cols]
y = df_processed['Rating_Category']

print(f"\nTotal Features Used: {len(feature_cols)}")
print(f"Feature set shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class Distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, f'{MODEL_DIR}/scaler.joblib')
print("\n✓ Scaler saved")

# ==================== Model Training & Evaluation ====================
print("\n" + "=" * 60)
print("TRAINING & EVALUATING MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

results = {}

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 40)
    
    # For KNN and Naive Bayes, use non-scaled data
    if model_name in ['K-Nearest Neighbors', 'Naive Bayes']:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'MCC': mcc,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    
    # Save model
    model_path = f'{MODEL_DIR}/{model_name.lower().replace(" ", "_")}.joblib'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")

# ==================== Results Summary ====================
print("\n" + "=" * 60)
print("RESULTS SUMMARY - ALL MODELS")
print("=" * 60)

results_df = pd.DataFrame({
    model_name: results[model_name] 
    for model_name in results.keys()
    if model_name not in ['Confusion Matrix', 'Classification Report']
})
results_df = results_df.T
results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'MCC']]

print("\n" + results_df.to_string())

# Save results to CSV
results_df.to_csv(f'{MODEL_DIR}/model_results.csv')
print(f"\n✓ Results saved to {MODEL_DIR}/model_results.csv")

# Save feature column names for later use
joblib.dump(feature_cols, f'{MODEL_DIR}/feature_columns.joblib')
print(f"✓ Feature columns saved")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
