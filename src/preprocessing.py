"""
Data loading and preprocessing for European Flights Dataset.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the European flights dataset from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling duplicates and missing values.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n=== Data Cleaning ===")
    initial_rows = len(df)
    
    # Drop duplicates
    df = df.drop_duplicates()
    print(f"Dropped {initial_rows - len(df)} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values per column:\n{missing[missing > 0]}")
    
    # Drop rows with missing target variable
    if 'FLT_TOT_1' in df.columns:
        df = df.dropna(subset=['FLT_TOT_1'])
        print(f"Dropped rows with missing FLT_TOT_1")
    
    # Keep only relevant columns
    relevant_cols = [
        'YEAR', 'MONTH_NUM', 'APT_ICAO', 'APT_NAME', 
        'STATE_NAME', 'FLT_TOT_1', 'FLT_DEP_1', 'FLT_ARR_1'
    ]
    
    # Only keep columns that exist in the dataset
    available_cols = [col for col in relevant_cols if col in df.columns]
    df = df[available_cols]
    
    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline: load and clean data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = load_data(filepath)
    df = clean_data(df)
    
    # Basic statistics
    print("\n=== Target Variable Statistics ===")
    if 'FLT_TOT_1' in df.columns:
        print(df['FLT_TOT_1'].describe())
    
    return df


if __name__ == "__main__":
    # Example usage
    df = preprocess_data("../data/european_flights.csv")
    print(f"\nPreprocessed data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
