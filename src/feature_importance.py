import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib
import os

from classification_experiment_settings import numeric_features, categorical_features
from util import fill_nan_values
from dataloader import load_arff_to_dataframe

def load_models_and_data(model_dir='saved_models', data_path='speeddating.arff'):
    """Load trained models and prepare data"""
    # Load models
    models = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.joblib') and not model_file.startswith(('learning_curves', 'validation_curves')):
            model_name = model_file.replace('.joblib', '')
            models[model_name] = joblib.load(os.path.join(model_dir, model_file))
    
    # Load and preprocess data
    df = load_arff_to_dataframe(data_path)
    df = df[numeric_features + categorical_features]  # Only keep features used in training
    df = fill_nan_values(df)
    
    return models, df

def get_feature_importance(model, model_name, X, y):
    """Extract feature importance based on model type and normalize them"""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        # For tree-based models (Random Forest, Decision Tree)
        importance = model.named_steps['model'].feature_importances_
        importance_type = 'native'
    elif hasattr(model.named_steps['model'], 'coef_'):
        # For linear models (Logistic Regression)
        importance = np.abs(model.named_steps['model'].coef_[0])
        importance_type = 'native'
    else:
        # For models without built-in feature importance (SVM)
        # Use permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance = perm_importance.importances_mean
        importance_type = 'permutation'

    # Normalize feature importance values
    if importance.sum() > 0:
        importance = importance / importance.sum()  # Ensures all values sum to 1
    
    return importance, importance_type

def get_feature_names(model):
    """Get feature names after preprocessing"""
    numeric_names = numeric_features
    
    # Get categorical feature names after one-hot encoding
    cat_names = []
    if hasattr(model.named_steps['preprocessor'].named_transformers_['cat'], 'get_feature_names_out'):
        cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    return np.concatenate([numeric_names, cat_names])

def plot_feature_importance(importance_dict, output_dir='feature_importance_plots'):
    """Plot feature importance for all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot individual model importances
    for model_name, data in importance_dict.items():
        plt.figure(figsize=(12, 6))
        
        # Sort features by importance and get top 10
        sorted_idx = np.argsort(data['importance'])[::-1][:10]  # Changed to get top 10
        pos = np.arange(len(sorted_idx)) + .5
        
        plt.barh(pos, data['importance'][sorted_idx])
        plt.yticks(pos, data['feature_names'][sorted_idx])
        plt.xlabel(f"Feature Importance ({data['type']})")
        plt.title(f"Feature Importance - Top 10 Features - {model_name}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_importance.png"))
        plt.close()
    
    # Create comparison plot for top 10 features
    plt.figure(figsize=(15, 8))
    
    # Prepare data for comparison
    model_names = list(importance_dict.keys())
    feature_importance_df = pd.DataFrame()
    
    for model_name in model_names:
        data = importance_dict[model_name]
        # Get top 10 features for this model
        sorted_idx = np.argsort(data['importance'])[::-1][:10]
        df = pd.DataFrame({
            'feature': data['feature_names'][sorted_idx],
            'importance': data['importance'][sorted_idx],
            'model': model_name
        })
        feature_importance_df = pd.concat([feature_importance_df, df])
    
    # Plot comparison
    sns.barplot(data=feature_importance_df, x='importance', y='feature', hue='model')
    plt.title("Top 10 Feature Importance Comparison Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_comparison.png"))
    plt.close()

def analyze_feature_importance():
    # Load models and data
    models, df = load_models_and_data()
    
    # Prepare data
    X = df
    y = pd.read_pickle('preprocessed.pkl')['match']  # Get target variable from preprocessed data
    
    # Calculate feature importance for each model
    importance_dict = {}
    
    for model_name, model in models.items():
        # Get feature importance
        if model_name == 'Support Vector Machine':
              continue
            
        importance, importance_type = get_feature_importance(model, model_name, X, y)
        
        # Get feature names
        feature_names = get_feature_names(model)
        
        # Store results
        importance_dict[model_name] = {
            'importance': importance,
            'feature_names': feature_names,
            'type': importance_type
        }
    
    # Plot results
    plot_feature_importance(importance_dict)
    
    return importance_dict

if __name__ == "__main__":
    importance_results = analyze_feature_importance()
    
    # Print top 10 most important features for each model
    for model_name, data in importance_results.items():
        print(f"\nTop 10 most important features for {model_name}:")
        
        # Sort features by importance
        sorted_idx = np.argsort(data['importance'])[::-1]
        for idx in sorted_idx[:10]:
            print(f"{data['feature_names'][idx]}: {data['importance'][idx]:.4f}")
