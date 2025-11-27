"""
Comprehensive evaluation and comparison of all models.
"""
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import time

from train_mlp import MLPRegressor, prepare_data_for_pytorch
from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_feature_columns
from train_lightgbm import train_test_split_data
from quantization import get_model_size, measure_inference_time


def load_all_models(feature_cols):
    """
    Load all trained models.
    
    Args:
        feature_cols: List of feature column names
        
    Returns:
        Dictionary of loaded models
    """
    print("=== Loading All Models ===")
    models = {}
    
    # Load LightGBM
    try:
        models['LightGBM'] = lgb.Booster(model_file='../models/lightgbm_model.txt')
        print("✓ LightGBM loaded")
    except:
        print("✗ LightGBM model not found")
    
    # Load MLP FP32
    try:
        mlp_fp32 = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
        mlp_fp32.load_state_dict(torch.load('../models/mlp_fp32.pt', map_location='cpu'))
        mlp_fp32.eval()
        models['MLP FP32'] = mlp_fp32
        print("✓ MLP FP32 loaded")
    except:
        print("✗ MLP FP32 model not found")
    
    # Load MLP Pruned
    try:
        mlp_pruned = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
        mlp_pruned.load_state_dict(torch.load('../models/mlp_pruned.pt', map_location='cpu'))
        mlp_pruned.eval()
        models['MLP Pruned FP32'] = mlp_pruned
        print("✓ MLP Pruned loaded")
    except:
        print("✗ MLP Pruned model not found")
    
    # Load MLP Quantized
    try:
        mlp_quantized = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
        mlp_quantized.load_state_dict(torch.load('../models/mlp_int8.pt', map_location='cpu'))
        mlp_quantized.eval()
        models['MLP Quantized INT8'] = mlp_quantized
        print("✓ MLP Quantized loaded")
    except:
        print("✗ MLP Quantized model not found")
    
    # Load scaler for neural network models
    try:
        with open('../models/mlp_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")
    except:
        print("✗ Scaler not found")
        scaler = None
    
    return models, scaler


def evaluate_all_models(models, scaler, X_test, y_test):
    """
    Evaluate all models on test set.
    
    Args:
        models: Dictionary of models
        scaler: Scaler for neural network inputs
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison metrics
    """
    print("\n=== Evaluating All Models ===")
    
    results = []
    predictions_dict = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        if 'LightGBM' in model_name:
            y_pred = model.predict(X_test)
        else:  # Neural network models
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
            else:
                X_test_tensor = torch.FloatTensor(X_test.values)
            
            with torch.no_grad():
                y_pred = model(X_test_tensor).numpy().flatten()
        
        predictions_dict[model_name] = y_pred
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Measure inference time
        if 'LightGBM' in model_name:
            start = time.time()
            for _ in range(100):
                _ = model.predict(X_test)
            inference_time = (time.time() - start) * 1000 / 100 / len(X_test)
        else:
            inference_time = measure_inference_time(model, X_test_tensor, device='cpu')
        
        # Get model size
        if 'LightGBM' in model_name:
            size = os.path.getsize('../models/lightgbm_model.txt') / (1024 * 1024)
        elif 'Pruned' in model_name:
            size = get_model_size(model, '../models/mlp_pruned.pt')
        elif 'Quantized' in model_name:
            size = get_model_size(model, '../models/mlp_int8.pt')
        else:  # FP32
            size = get_model_size(model, '../models/mlp_fp32.pt')
        
        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Size (MB)': size,
            'Inference Time (ms/sample)': inference_time
        })
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        print(f"  Size: {size:.2f} MB, Inference: {inference_time:.6f} ms/sample")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions_dict


def plot_comparison_table(results_df):
    """
    Create and save comparison table visualization.
    
    Args:
        results_df: DataFrame with model comparison metrics
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the table
    table_data = results_df.copy()
    table_data['RMSE'] = table_data['RMSE'].apply(lambda x: f"{x:.2f}")
    table_data['MAE'] = table_data['MAE'].apply(lambda x: f"{x:.2f}")
    table_data['R²'] = table_data['R²'].apply(lambda x: f"{x:.4f}")
    table_data['Size (MB)'] = table_data['Size (MB)'].apply(lambda x: f"{x:.2f}")
    table_data['Inference Time (ms/sample)'] = table_data['Inference Time (ms/sample)'].apply(lambda x: f"{x:.6f}")
    
    table = ax.table(cellText=table_data.values, 
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('../models/comparison_table.png', dpi=150, bbox_inches='tight')
    print("\nComparison table saved to models/comparison_table.png")
    plt.close()


def plot_predictions_vs_actual(y_test, predictions_dict):
    """
    Plot predictions vs actual values for all models.
    
    Args:
        y_test: True labels
        predictions_dict: Dictionary of predictions per model
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual FLT_TOT_1', fontsize=10)
        ax.set_ylabel('Predicted FLT_TOT_1', fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../models/predictions_vs_actual.png', dpi=150)
    print("Predictions plot saved to models/predictions_vs_actual.png")
    plt.close()


def plot_residuals(y_test, predictions_dict):
    """
    Plot residual distributions for all models.
    
    Args:
        y_test: True labels
        predictions_dict: Dictionary of predictions per model
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        residuals = y_test.values - y_pred
        
        # Histogram
        ax.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{model_name} - Residuals Distribution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../models/residuals_distribution.png', dpi=150)
    print("Residuals plot saved to models/residuals_distribution.png")
    plt.close()


def plot_metrics_comparison(results_df):
    """
    Plot bar charts comparing metrics across models.
    
    Args:
        results_df: DataFrame with model comparison metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE
    axes[0, 0].bar(results_df['Model'], results_df['RMSE'], color='steelblue')
    axes[0, 0].set_title('RMSE Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R²
    axes[0, 1].bar(results_df['Model'], results_df['R²'], color='green')
    axes[0, 1].set_title('R² Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model Size
    axes[1, 0].bar(results_df['Model'], results_df['Size (MB)'], color='orange')
    axes[1, 0].set_title('Model Size Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Size (MB)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Inference Time
    axes[1, 1].bar(results_df['Model'], results_df['Inference Time (ms/sample)'], color='red')
    axes[1, 1].set_title('Inference Time Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Time (ms/sample)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../models/metrics_comparison.png', dpi=150)
    print("Metrics comparison plot saved to models/metrics_comparison.png")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Load and preprocess data
    df = preprocess_data("../data/european_flights.csv")
    df_feat, encoders = engineer_features(df, create_lags=True)
    
    # Get features
    feature_cols = get_feature_columns(df_feat)
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split_data(
        df_feat, feature_cols, test_size=0.2, random_state=42
    )
    
    # Load all models
    models, scaler = load_all_models(feature_cols)
    
    if not models:
        print("\nNo models found! Please train models first.")
        return
    
    # Evaluate all models
    results_df, predictions_dict = evaluate_all_models(models, scaler, X_test, y_test)
    
    # Save results
    results_df.to_csv('../models/evaluation_results.csv', index=False)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nResults saved to models/evaluation_results.csv")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_comparison_table(results_df)
    plot_predictions_vs_actual(y_test, predictions_dict)
    plot_residuals(y_test, predictions_dict)
    plot_metrics_comparison(results_df)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    results = main()
