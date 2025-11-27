"""
Feature engineering for air traffic prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from YEAR and MONTH_NUM.
    
    Args:
        df: DataFrame with YEAR and MONTH_NUM columns
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Normalize year for trend analysis
    if 'YEAR' in df.columns:
        min_year = df['YEAR'].min()
        df['YEAR_TREND'] = df['YEAR'] - min_year
    
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal features based on month.
    
    Args:
        df: DataFrame with MONTH_NUM column
        
    Returns:
        DataFrame with seasonal features
    """
    df = df.copy()
    
    if 'MONTH_NUM' not in df.columns:
        return df
    
    # Season mapping
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df['SEASON'] = df['MONTH_NUM'].map(season_map)
    
    # Binary seasonal flags
    df['IS_SUMMER'] = (df['MONTH_NUM'].isin([6, 7, 8])).astype(int)
    df['IS_WINTER'] = (df['MONTH_NUM'].isin([12, 1, 2])).astype(int)
    
    # Cyclical encoding for month (preserves circular nature)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       target_col: str = 'FLT_TOT_1',
                       lags: List[int] = [1, 3]) -> pd.DataFrame:
    """
    Create lag features per airport.
    
    Args:
        df: DataFrame with airport and target columns
        target_col: Name of the target column
        lags: List of lag periods to create
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    if 'APT_ICAO' not in df.columns or target_col not in df.columns:
        return df
    
    # Sort by airport and time
    df = df.sort_values(['APT_ICAO', 'YEAR', 'MONTH_NUM'])
    
    # Create lag features grouped by airport
    for lag in lags:
        if lag == 1:
            df[f'lag_1'] = df.groupby('APT_ICAO')[target_col].shift(1)
        elif lag == 3:
            # Average of previous 3 months
            df[f'lag_3'] = df.groupby('APT_ICAO')[target_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
            )
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                cat_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using label encoding.
    
    Args:
        df: DataFrame with categorical columns
        cat_columns: List of categorical column names to encode
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df = df.copy()
    
    if cat_columns is None:
        cat_columns = ['APT_ICAO', 'STATE_NAME', 'SEASON']
    
    # Only encode columns that exist
    cat_columns = [col for col in cat_columns if col in df.columns]
    
    encoders = {}
    
    for col in cat_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df, encoders


def engineer_features(df: pd.DataFrame, 
                     create_lags: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        create_lags: Whether to create lag features
        
    Returns:
        Tuple of (DataFrame with engineered features, encoders dictionary)
    """
    print("\n=== Feature Engineering ===")
    
    # Time features
    df = create_time_features(df)
    print("Created time features")
    
    # Seasonal features
    df = create_seasonal_features(df)
    print("Created seasonal features")
    
    # Lag features
    if create_lags:
        df = create_lag_features(df)
        print("Created lag features")
        # Drop rows with NaN lags
        initial_rows = len(df)
        df = df.dropna(subset=['lag_1'])
        print(f"Dropped {initial_rows - len(df)} rows due to lag features")
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    print("Encoded categorical features")
    
    print(f"\nFinal feature set: {len(df)} rows, {len(df.columns)} columns")
    
    return df, encoders


def get_feature_columns(df: pd.DataFrame, target_col: str = 'FLT_TOT_1') -> List[str]:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column to exclude
        
    Returns:
        List of feature column names
    """
    # Exclude original categorical columns, identifiers, and target
    exclude_cols = [
        target_col, 'FLT_DEP_1', 'FLT_ARR_1',
        'APT_ICAO', 'APT_NAME', 'STATE_NAME', 'SEASON'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure we have numeric features only
    feature_cols = [col for col in feature_cols 
                   if df[col].dtype in ['int64', 'float64']]
    
    return feature_cols


if __name__ == "__main__":
    from preprocessing import preprocess_data
    
    # Example usage
    df = preprocess_data("../data/european_flights.csv")
    df_feat, encoders = engineer_features(df)
    
    feature_cols = get_feature_columns(df_feat)
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols)
    
    print(f"\nSample of engineered features:\n{df_feat[feature_cols].head()}")
