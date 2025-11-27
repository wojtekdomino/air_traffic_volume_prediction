"""
Quantization for MLP model to reduce size and improve inference speed.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

from train_mlp import MLPRegressor, prepare_data_for_pytorch
from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_feature_columns
from train_lightgbm import train_test_split_data


def quantize_model(model, backend='fbgemm'):
    """
    Apply dynamic quantization to model.
    
    Args:
        model: FP32 model to quantize
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    print(f"\n=== Applying Dynamic Quantization (backend={backend}) ===")
    
    # Set quantization backend
    if backend == 'fbgemm':
        torch.backends.quantized.engine = 'fbgemm'
    elif backend == 'qnnpack':
        torch.backends.quantized.engine = 'qnnpack'
    
    # Model must be in eval mode for quantization
    model.eval()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8  # Use int8 quantization
    )
    
    print("Quantization completed!")
    
    return quantized_model


def get_model_size(model, filepath=None):
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        filepath: Optional path to saved model file
        
    Returns:
        Size in MB
    """
    if filepath and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
    else:
        # Save to temporary file to get size
        temp_path = '../models/temp_model.pt'
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    
    return size_mb


def measure_inference_time(model, X_test_tensor, device='cpu', num_runs=100):
    """
    Measure average inference time.
    
    Args:
        model: Model to evaluate
        X_test_tensor: Test data tensor
        device: Device to run on
        num_runs: Number of inference runs for averaging
        
    Returns:
        Average time per sample in milliseconds
    """
    model.eval()
    X_test_tensor = X_test_tensor.to(device)
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_test_tensor)
    
    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(X_test_tensor)
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time_per_batch = total_time / num_runs
    avg_time_per_sample = avg_time_per_batch / len(X_test_tensor)
    
    return avg_time_per_sample


def evaluate_quantized_model(model, X_test_tensor, y_test, device='cpu'):
    """
    Evaluate quantized model.
    
    Args:
        model: Quantized model
        X_test_tensor: Test features
        y_test: True labels
        device: Device
        
    Returns:
        Metrics and predictions
    """
    model.eval()
    
    with torch.no_grad():
        # Note: quantized models typically run on CPU
        predictions = model(X_test_tensor.cpu()).numpy().flatten()
    
    y_true = y_test.values if hasattr(y_test, 'values') else y_test.flatten()
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
        'mae': mean_absolute_error(y_true, predictions),
        'r2': r2_score(y_true, predictions)
    }
    
    print(f"\n=== Quantized MLP Evaluation ===")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return metrics, predictions


def save_quantized_model(model, filepath='../models/mlp_int8.pt'):
    """Save quantized model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"\nQuantized model saved to {filepath}")


def main():
    """Main quantization pipeline."""
    # Quantized models typically run on CPU
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df = preprocess_data("../data/european_flights.csv")
    df_feat, encoders = engineer_features(df, create_lags=True)
    
    # Get features
    feature_cols = get_feature_columns(df_feat)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        df_feat, feature_cols, test_size=0.2, random_state=42
    )
    
    # Prepare for PyTorch
    train_loader, test_loader, scaler, X_test_tensor, y_test_tensor = prepare_data_for_pytorch(
        X_train, X_test, y_train, y_test, batch_size=256
    )
    
    # Load pruned model (or original if pruned not available)
    print("\nLoading pruned MLP model...")
    model = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
    
    try:
        model.load_state_dict(torch.load('../models/mlp_pruned.pt', map_location=device))
        print("Loaded pruned model")
        model_type = "pruned"
    except:
        print("Pruned model not found, loading original FP32 model")
        model.load_state_dict(torch.load('../models/mlp_fp32.pt', map_location=device))
        model_type = "fp32"
    
    # Evaluate original model
    print(f"\n=== Original {model_type.upper()} Model Performance ===")
    metrics_original, _ = evaluate_quantized_model(model, X_test_tensor, y_test, device=device)
    
    # Measure original model size and speed
    size_original = get_model_size(model, f'../models/mlp_{model_type}.pt')
    time_original = measure_inference_time(model, X_test_tensor, device=device)
    
    print(f"\nOriginal model size: {size_original:.2f} MB")
    print(f"Original inference time: {time_original:.4f} ms/sample")
    
    # Apply quantization
    quantized_model = quantize_model(model, backend='fbgemm')
    
    # Evaluate quantized model
    metrics_quantized, predictions = evaluate_quantized_model(
        quantized_model, X_test_tensor, y_test, device=device
    )
    
    # Save quantized model
    save_quantized_model(quantized_model)
    
    # Measure quantized model size and speed
    size_quantized = get_model_size(quantized_model, '../models/mlp_int8.pt')
    time_quantized = measure_inference_time(quantized_model, X_test_tensor, device=device)
    
    print(f"\nQuantized model size: {size_quantized:.2f} MB")
    print(f"Quantized inference time: {time_quantized:.4f} ms/sample")
    
    # Compare performance
    print("\n=== Compression Results ===")
    print(f"Size reduction: {size_original:.2f} MB -> {size_quantized:.2f} MB ({100 * (1 - size_quantized/size_original):.1f}% smaller)")
    print(f"Speed improvement: {time_original:.4f} ms -> {time_quantized:.4f} ms ({time_original/time_quantized:.2f}x faster)")
    print(f"\nAccuracy comparison:")
    print(f"Original - RMSE: {metrics_original['rmse']:.2f}, R²: {metrics_original['r2']:.4f}")
    print(f"Quantized - RMSE: {metrics_quantized['rmse']:.2f}, R²: {metrics_quantized['r2']:.4f}")
    
    return quantized_model, metrics_quantized


if __name__ == "__main__":
    model, metrics = main()
