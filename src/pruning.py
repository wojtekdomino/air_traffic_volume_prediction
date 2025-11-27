"""
Structured pruning for MLP model.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from train_mlp import MLPRegressor, prepare_data_for_pytorch, train_epoch, validate
from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_feature_columns
from train_lightgbm import train_test_split_data


def count_parameters(model):
    """Count total and non-zero parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params


def structured_prune_model(model, amount=0.3):
    """
    Apply structured pruning to MLP model.
    
    Args:
        model: MLP model to prune
        amount: Fraction of neurons to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    print(f"\n=== Applying Structured Pruning (amount={amount}) ===")
    
    # Get original parameter count
    total_before, nonzero_before = count_parameters(model)
    print(f"Parameters before pruning: {total_before:,} (non-zero: {nonzero_before:,})")
    
    # Apply structured pruning to linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Prune neurons (structured pruning along dimension 0)
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    # Get parameter count after pruning (with mask)
    total_after, nonzero_after = count_parameters(model)
    print(f"Parameters after pruning: {total_after:,} (non-zero: {nonzero_after:,})")
    print(f"Sparsity: {100 * (1 - nonzero_after / total_after):.2f}%")
    
    return model


def remove_pruning_reparameterization(model):
    """
    Remove pruning reparameterization to make pruning permanent.
    
    Args:
        model: Pruned model
        
    Returns:
        Model with pruning made permanent
    """
    print("\nRemoving pruning reparameterization...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass  # No pruning on this layer
    
    return model


def fine_tune_pruned_model(model, train_loader, test_loader, 
                          epochs=10, learning_rate=0.0001, device='cpu'):
    """
    Fine-tune pruned model.
    
    Args:
        model: Pruned model
        train_loader, test_loader: Data loaders
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        device: Device
        
    Returns:
        Fine-tuned model, loss history
    """
    print(f"\n=== Fine-tuning Pruned Model ({epochs} epochs) ===")
    
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return model, {'train': train_losses, 'test': test_losses}


def evaluate_pruned_model(model, X_test_tensor, y_test, device='cpu'):
    """
    Evaluate pruned model.
    
    Args:
        model: Pruned model
        X_test_tensor: Test features
        y_test: True labels
        device: Device
        
    Returns:
        Metrics and predictions
    """
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    y_true = y_test.values if hasattr(y_test, 'values') else y_test.flatten()
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
        'mae': mean_absolute_error(y_true, predictions),
        'r2': r2_score(y_true, predictions)
    }
    
    print(f"\n=== Pruned MLP Evaluation ===")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return metrics, predictions


def save_pruned_model(model, filepath='../models/mlp_pruned.pt'):
    """Save pruned model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"\nPruned model saved to {filepath}")


def main():
    """Main pruning pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Load original MLP model
    print("\nLoading original MLP model...")
    model = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32]).to(device)
    model.load_state_dict(torch.load('../models/mlp_fp32.pt', map_location=device))
    
    # Evaluate original model
    print("\n=== Original Model Performance ===")
    metrics_original, _ = evaluate_pruned_model(model, X_test_tensor, y_test, device=device)
    
    # Apply structured pruning
    model = structured_prune_model(model, amount=0.3)
    
    # Evaluate immediately after pruning (before fine-tuning)
    print("\n=== Performance After Pruning (before fine-tuning) ===")
    metrics_pruned, _ = evaluate_pruned_model(model, X_test_tensor, y_test, device=device)
    
    # Fine-tune pruned model
    model, history = fine_tune_pruned_model(
        model, train_loader, test_loader,
        epochs=10, learning_rate=0.0001, device=device
    )
    
    # Remove pruning reparameterization
    model = remove_pruning_reparameterization(model)
    
    # Final evaluation
    metrics_final, predictions = evaluate_pruned_model(model, X_test_tensor, y_test, device=device)
    
    # Save pruned model
    save_pruned_model(model)
    
    # Compare performance
    print("\n=== Performance Comparison ===")
    print(f"Original - RMSE: {metrics_original['rmse']:.2f}, R²: {metrics_original['r2']:.4f}")
    print(f"After Pruning - RMSE: {metrics_pruned['rmse']:.2f}, R²: {metrics_pruned['r2']:.4f}")
    print(f"After Fine-tuning - RMSE: {metrics_final['rmse']:.2f}, R²: {metrics_final['r2']:.4f}")
    
    return model, metrics_final


if __name__ == "__main__":
    model, metrics = main()
