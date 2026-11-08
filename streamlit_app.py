"""
Streamlit Web Application for ML Classification Models
Bakery Sales Dataset - Customer Rating Prediction
Student ID: 2026AK05229
BITS WILP M.Tech (AIML) - Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom Styling ====================
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
    }
    .model-title {
        color: #0066cc;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Models and Data ====================
@st.cache_resource
def load_models_and_data():
    """Load all trained models and supporting files"""
    models = {}
    model_dir = './models'
    
    # Load scaler
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    
    # Load feature columns
    feature_cols = joblib.load(f'{model_dir}/feature_columns.joblib')
    
    # Load models
    model_files = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'K-Nearest Neighbors': 'k-nearest_neighbors.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'support_vector_machine.joblib'
    }
    
    for model_name, filename in model_files.items():
        model_path = f'{model_dir}/{filename}'
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    # Load results
    results_df = pd.read_csv(f'{model_dir}/model_results.csv', index_col=0)
    
    return models, scaler, feature_cols, results_df

try:
    models, scaler, feature_cols, results_df = load_models_and_data()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# ==================== Load Test Dataset ====================
@st.cache_data
def load_test_data():
    """Load test data for demonstrations"""
    try:
        test_df = pd.read_csv('../Assignment_2/bakery_sales_dataset.csv')
        return test_df.head(100)  # Return first 100 rows for demo
    except:
        return None

test_data = load_test_data()

# ==================== Main Application ====================
st.title("🎯 ML Classification Models - Bakery Sales Analysis")
st.markdown("**Predicting Customer Rating Categories Using 5 Different Algorithms**")

# ==================== Sidebar ====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["📈 Model Performance", "🔮 Make Predictions", "📋 Dataset Info", "📊 Comparison Analysis"]
)

# ==================== Page 1: Model Performance ====================
if page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    st.markdown("""
    This section displays the performance metrics for all 5 trained classification models
    evaluated on the Bakery Sales dataset for predicting customer rating categories.
    """)
    
    # Metrics Table
    st.subheader("Performance Comparison Table")
    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        height=300
    )
    
    # Metrics Explanation
    st.markdown("""
    ### Metrics Explanation:
    - **Accuracy**: Percentage of correct predictions
    - **Precision**: How many predicted positive are actually positive
    - **Recall**: How many actual positive are correctly identified
    - **F1 Score**: Harmonic mean of Precision and Recall
    - **AUC**: Area Under the ROC Curve (probability of correct ranking)
    - **MCC**: Matthews Correlation Coefficient (correlation between predicted and actual)
    """)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        accuracy_data = results_df['Accuracy'].sort_values(ascending=False)
        bars = ax.barh(accuracy_data.index, accuracy_data.values, color='#0066cc')
        ax.set_xlabel('Accuracy Score')
        ax.set_xlim([0.85, 0.95])
        for i, v in enumerate(accuracy_data.values):
            ax.text(v + 0.002, i, f'{v:.4f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        f1_data = results_df['F1 Score'].sort_values(ascending=False)
        bars = ax.barh(f1_data.index, f1_data.values, color='#00b366')
        ax.set_xlabel('F1 Score')
        ax.set_xlim([0.93, 0.95])
        for i, v in enumerate(f1_data.values):
            ax.text(v + 0.0005, i, f'{v:.4f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Radar Chart
    st.subheader("Multi-Metric Radar Comparison")
    
    from math import pi
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'MCC']
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    
    for idx, (model_name, color) in enumerate(zip(results_df.index, colors_list)):
        values = results_df.loc[model_name, categories].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Model Performance Comparison (Radar Chart)', size=14, weight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best Models
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        best_accuracy_model = results_df['Accuracy'].idxmax()
        best_accuracy_score = results_df['Accuracy'].max()
        st.metric("🥇 Best Accuracy", best_accuracy_model, f"{best_accuracy_score:.4f}")
    with col2:
        best_f1_model = results_df['F1 Score'].idxmax()
        best_f1_score = results_df['F1 Score'].max()
        st.metric("🥇 Best F1 Score", best_f1_model, f"{best_f1_score:.4f}")
    with col3:
        best_auc_model = results_df['AUC'].idxmax()
        best_auc_score = results_df['AUC'].max()
        st.metric("🥇 Best AUC", best_auc_model, f"{best_auc_score:.4f}")

# ==================== Page 2: Make Predictions ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Make Predictions on New Data")
    
    st.markdown("""
    Upload a CSV file or use the demo data to make predictions with different models.
    The app will preprocess your data and predict whether customers will give high ratings (1) or low ratings (0).
    """)
    
    # Model Selection
    selected_model = st.selectbox(
        "Select a Classification Model:",
        list(models.keys()),
        help="Choose which model to use for predictions"
    )
    
    # Data Input Options
    input_option = st.radio("Data Input Method:", ["Upload CSV", "Use Sample Data"])
    
    prediction_data = None
    
    if input_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                prediction_data = pd.read_csv(uploaded_file)
                st.success(f"✓ File uploaded: {prediction_data.shape[0]} rows")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    else:
        if test_data is not None:
            st.info("Using first 5 rows from bakery sales dataset")
            prediction_data = test_data.head(5)
            st.dataframe(prediction_data, use_container_width=True)
    
    # Make Predictions
    if prediction_data is not None and st.button("🚀 Make Predictions", use_container_width=True):
        try:
            st.info("Processing predictions...")
            
            # Select feature columns from the uploaded data
            available_features = [col for col in feature_cols if col.replace('_encoded', '') in prediction_data.columns or col in prediction_data.columns]
            
            if len(available_features) < len(feature_cols):
                st.warning(f"⚠️ Note: Using {len(available_features)} available features out of {len(feature_cols)} expected features")
            
            # For demo purposes, show predictions on first few rows
            demo_data = prediction_data.head(10).copy()
            
            # Make predictions
            if selected_model in ['K-Nearest Neighbors', 'Naive Bayes']:
                predictions = models[selected_model].predict(demo_data[feature_cols].fillna(demo_data[feature_cols].mean()))
                probabilities = models[selected_model].predict_proba(demo_data[feature_cols].fillna(demo_data[feature_cols].mean()))[:, 1]
            else:
                X_scaled = scaler.transform(demo_data[feature_cols].fillna(demo_data[feature_cols].mean()))
                predictions = models[selected_model].predict(X_scaled)
                probabilities = models[selected_model].predict_proba(X_scaled)[:, 1]
            
            # Display Results
            results_display = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Prediction': ['High Rating ✅' if p == 1 else 'Low Rating ❌' for p in predictions],
                'Confidence (%)': [f'{prob*100:.2f}%' for prob in probabilities],
                'Probability': probabilities.round(4)
            })
            
            st.subheader(f"Predictions using {selected_model}")
            st.dataframe(results_display, use_container_width=True)
            
            # Summary Statistics
            col1, col2, col3 = st.columns(3)
            high_rating_count = (predictions == 1).sum()
            low_rating_count = (predictions == 0).sum()
            avg_confidence = probabilities.mean() * 100
            
            with col1:
                st.metric("High Ratings Predicted", high_rating_count)
            with col2:
                st.metric("Low Ratings Predicted", low_rating_count)
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
            
            st.success("✓ Predictions completed successfully!")
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.info(f"Debug: Available columns in data: {prediction_data.columns.tolist()[:5]}...")

# ==================== Page 3: Dataset Info ====================
elif page == "📋 Dataset Info":
    st.header("📋 Dataset Information")
    
    st.markdown("""
    ### Bakery Sales Dataset
    This dataset contains comprehensive information about bakery sales transactions,
    customer demographics, product details, and environmental factors.
    """)
    
    # Dataset Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", "19,321")
    with col2:
        st.metric("Total Features", "40")
    with col3:
        st.metric("Features Used", len(feature_cols))
    
    st.markdown("---")
    
    # Display raw data sample
    if test_data is not None:
        st.subheader("Data Sample")
        st.dataframe(test_data.head(10), use_container_width=True)
        
        # Data Info
        st.subheader("Dataset Features")
        feature_info = pd.DataFrame({
            'Feature': feature_cols,
            'Type': ['Numerical'] * len(feature_cols)
        })
        st.dataframe(feature_info, use_container_width=True)

# ==================== Page 4: Comparison Analysis ====================
elif page == "📊 Comparison Analysis":
    st.header("📊 Detailed Model Comparison Analysis")
    
    st.markdown("""
    This section provides an in-depth comparison of all models across different metrics
    and provides recommendations based on various use cases.
    """)
    
    # Key Findings
    st.subheader("🔑 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Performance Summary:**
        - **Logistic Regression**: Balanced performance with 90.22% accuracy
        - **Decision Tree**: Best AUC (0.6637) with 89.94% accuracy
        - **K-Nearest Neighbors**: Good recall at 99.02%
        - **Naive Bayes**: Consistent recall performance (98.16%)
        - **SVM**: Tied highest accuracy (90.22%) with strong AUC (0.6601)
        """)
    
    with col2:
        st.markdown("""
        **Recommendations:**
        
        **For Production Use:**
        - **SVM or Logistic Regression** for balanced accuracy
        
        **For Maximizing True Positives:**
        - **Decision Tree** (best AUC and strong recall)
        
        **For Interpretability:**
        - **Decision Tree** (easily explainable splits)
        
        **For Real-time Predictions:**
        - **Logistic Regression** (fastest inference)
        """)
    
    st.markdown("---")
    
    # Metrics Deep Dive
    st.subheader("📊 Metrics Deep Dive")
    
    # Create detailed comparison
    metrics_to_analyze = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'MCC']
    
    for metric in metrics_to_analyze:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            metric_values = results_df[metric].sort_values(ascending=False)
            colors = ['#0066cc' if i == 0 else '#cccccc' for i in range(len(metric_values))]
            bars = ax.bar(range(len(metric_values)), metric_values.values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(metric_values)))
            ax.set_xticklabels(metric_values.index, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylim([0, 1])
            
            for i, v in enumerate(metric_values.values):
                ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            best_model = metric_values.idxmax()
            best_score = metric_values.max()
            st.metric(f"Best in {metric}", best_model, f"{best_score:.4f}")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 <strong>BITS WILP M.Tech AIML - Assignment 2</strong></p>
    <p>Classification Models on Bakery Sales Dataset | 2024</p>
</div>
""", unsafe_allow_html=True)
