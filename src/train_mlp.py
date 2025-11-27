"""
Train MLP (Multi-Layer Perceptron) regressor using PyTorch.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
import time

from preprocessing import preprocess_data
from feature_engineering import engineer_features, get_feature_columns
from train_lightgbm import train_test_split_data


class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for regression."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        """
        Initialize MLP.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
        """
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def prepare_data_for_pytorch(X_train, X_test, y_train, y_test, batch_size=256):
    """
    Prepare data for PyTorch training.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader, test_loader, scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, X_test_tensor, y_test_tensor


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def train_mlp(train_loader, test_loader, input_size, 
              hidden_sizes=[128, 64, 32], 
              epochs=50, 
              learning_rate=0.001,
              device='cpu'):
    """
    Train MLP model.
    
    Args:
        train_loader, test_loader: Data loaders
        input_size: Number of input features
        hidden_sizes: Hidden layer sizes
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained model, loss history
    """
    print(f"\n=== Training MLP on {device} ===")
    print(f"Architecture: Input({input_size}) -> {hidden_sizes} -> Output(1)")
    
    # Initialize model
    model = MLPRegressor(input_size, hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return model, {'train': train_losses, 'test': test_losses}


def evaluate_mlp(model, X_test_tensor, y_test, device='cpu'):
    """
    Evaluate MLP model.
    
    Args:
        model: Trained model
        X_test_tensor: Test features tensor
        y_test: True test labels
        device: Device
        
    Returns:
        Dictionary with metrics and predictions
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
    
    print(f"\n=== MLP Evaluation ===")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return metrics, predictions


def plot_training_curves(history, save_path='../models/mlp_training_curves.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train Loss', linewidth=2)
    plt.plot(history['test'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MLP Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def save_mlp_model(model, scaler, filepath='../models/mlp_fp32.pt', 
                   scaler_path='../models/mlp_scaler.pkl'):
    """Save MLP model and scaler."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {filepath}")
    print(f"Scaler saved to {scaler_path}")


def main():
    """Main training pipeline."""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df = preprocess_data("../data/european_flights.csv")
    df_feat, encoders = engineer_features(df, create_lags=True)
    
    # Get features
    feature_cols = get_feature_columns(df_feat)
    print(f"\nUsing {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        df_feat, feature_cols, test_size=0.2, random_state=42
    )
    
    # Prepare for PyTorch
    train_loader, test_loader, scaler, X_test_tensor, y_test_tensor = prepare_data_for_pytorch(
        X_train, X_test, y_train, y_test, batch_size=256
    )
    
    # Train model
    model, history = train_mlp(
        train_loader, test_loader,
        input_size=len(feature_cols),
        hidden_sizes=[128, 64, 32],
        epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    # Evaluate
    metrics, predictions = evaluate_mlp(model, X_test_tensor, y_test, device=device)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Save model
    save_mlp_model(model, scaler)
    
    # Save predictions
    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred_mlp': predictions
    })
    test_results.to_csv('../models/mlp_predictions.csv', index=False)
    print("Test predictions saved to models/mlp_predictions.csv")
    
    return model, scaler, metrics


if __name__ == "__main__":
    model, scaler, metrics = main()
