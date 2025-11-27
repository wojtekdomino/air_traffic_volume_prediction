"""
Train and evaluate LightGBM regressor for air traffic prediction.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_feature_columns


def train_test_split_data(df: pd.DataFrame, 
                          feature_cols: list, 
                          target_col: str = 'FLT_TOT_1',
                          test_size: float = 0.2,
                          random_state: int = 42):
    """
    Split data into train and test sets.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_lightgbm(X_train, y_train, X_test, y_test, params=None):
    """
    Train LightGBM regressor.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        params: LightGBM parameters (optional)
        
    Returns:
        Trained model
    """
    print("\n=== Training LightGBM ===")
    
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="LightGBM"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for display
        
    Returns:
        Dictionary with metrics
    """
    print(f"\n=== Evaluating {model_name} ===")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    # Print metrics
    print(f"\nTrain RMSE: {metrics['train_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Train MAE: {metrics['train_mae']:.2f}")
    print(f"Test MAE: {metrics['test_mae']:.2f}")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    
    return metrics, y_test_pred


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance (LightGBM)')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.savefig('../models/lightgbm_feature_importance.png', dpi=150)
    print("\nFeature importance plot saved to models/lightgbm_feature_importance.png")
    plt.close()


def save_model(model, filepath='../models/lightgbm_model.txt'):
    """
    Save LightGBM model.
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    print(f"\nModel saved to {filepath}")


def main():
    """Main training pipeline."""
    # Load and preprocess data
    df = preprocess_data("../data/european_flights.csv")
    
    # Engineer features
    df_feat, encoders = engineer_features(df, create_lags=True)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_feat)
    print(f"\nUsing {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        df_feat, feature_cols, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics, predictions = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(model, feature_cols, top_n=20)
    
    # Save model
    save_model(model)
    
    # Save test predictions for later comparison
    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred_lightgbm': predictions
    })
    test_results.to_csv('../models/lightgbm_predictions.csv', index=False)
    print("Test predictions saved to models/lightgbm_predictions.csv")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
