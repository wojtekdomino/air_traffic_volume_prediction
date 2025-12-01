"""
================================================================================
PROJEKT AKADEMICKI: PREDYKCJA NATĘŻENIA RUCHU LOTNICZEGO
================================================================================

Autor: Wojciech Domino & Mateusz Maj
Cel: Przewidywanie liczby operacji lotniczych (IFR movements) na lotniskach 
     europejskich przy użyciu modeli uczenia maszynowego

Dataset: European Flights Dataset - miesięczne dane o operacjach lotniczych

Modele wykorzystane w projekcie:
1. LightGBM - model bazowy (gradient boosting)
2. MLP (Multi-Layer Perceptron) - sieć neuronowa w PyTorch
3. MLP z przycinaniem (pruning) - kompresja sieci
4. MLP z kwantyzacją (quantization) - redukcja rozmiaru modelu

Struktura projektu:
- Sekcja 1: Import bibliotek i konfiguracja
- Sekcja 2: Przygotowanie danych (preprocessing)
- Sekcja 3: Inżynieria cech (feature engineering)
- Sekcja 4: Trenowanie modelu LightGBM
- Sekcja 5: Trenowanie sieci neuronowej MLP
- Sekcja 6: Kompresja modeli (pruning i quantization)
- Sekcja 7: Porównanie wszystkich modeli
- Sekcja 8: Główna funkcja uruchamiająca projekt

================================================================================
"""

# ============================================================================
# SEKCJA 1: IMPORT BIBLIOTEK I KONFIGURACJA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
from typing import Tuple, Dict, List

# Machine Learning - modele klasyczne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils.prune as prune

# Konfiguracja wyświetlania
import warnings
warnings.filterwarnings('ignore')

# Stałe konfiguracyjne
RANDOM_STATE = 42          # Ziarno losowości dla reprodukowalności wyników
TEST_SIZE = 0.2           # Proporcja zbioru testowego (20%)
BATCH_SIZE = 512          # Rozmiar paczki dla treningu sieci neuronowej
LEARNING_RATE = 0.001     # Współczynnik uczenia dla sieci neuronowej
EPOCHS = 50               # Liczba epok treningu sieci neuronowej (zmniejszone dla szybszego treningu)
PRUNING_AMOUNT = 0.3      # Procent neuronów do przycięcia (30%)

# Ścieżki do plików
DATA_PATH = 'data/european_flights.csv'
MODELS_DIR = 'models'


# ============================================================================
# SEKCJA 2: PRZYGOTOWANIE DANYCH (PREPROCESSING)
# ============================================================================

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Funkcja do wczytywania i czyszczenia danych.
    
    Operacje wykonywane:
    1. Wczytanie danych z pliku CSV
    2. Usunięcie duplikatów
    3. Usunięcie wierszy z brakującymi wartościami w zmiennej docelowej
    4. Wybór istotnych kolumn
    
    Parametry:
        filepath (str): Ścieżka do pliku CSV z danymi
        
    Zwraca:
        pd.DataFrame: Oczyszczone dane
    """
    print("="*80)
    print("ETAP 1: WCZYTYWANIE I CZYSZCZENIE DANYCH")
    print("="*80)
    
    # Wczytanie danych
    print(f"\n[1.1] Wczytywanie danych z pliku: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Wczytano {len(df):,} wierszy i {len(df.columns)} kolumn")
    
    # Usunięcie duplikatów
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"\n[1.2] Usunięto {duplicates_removed:,} duplikatów")
    
    # Analiza braków danych
    print(f"\n[1.3] Analiza braków danych:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("      Brak brakujących wartości")
    
    # Usunięcie wierszy z brakującą zmienną docelową
    if 'FLT_TOT_1' in df.columns:
        rows_before = len(df)
        df = df.dropna(subset=['FLT_TOT_1'])
        rows_removed = rows_before - len(df)
        print(f"\n[1.4] Usunięto {rows_removed:,} wierszy z brakującą zmienną docelową")
    
    # Wybór istotnych kolumn
    # FLT_TOT_1 - zmienna docelowa (total IFR movements)
    # FLT_DEP_1 - liczba odlotów
    # FLT_ARR_1 - liczba przylotów
    relevant_cols = [
        'YEAR', 'MONTH_NUM', 'APT_ICAO', 'APT_NAME', 
        'STATE_NAME', 'FLT_TOT_1', 'FLT_DEP_1', 'FLT_ARR_1'
    ]
    
    # Zachowanie tylko kolumn, które istnieją w danych
    available_cols = [col for col in relevant_cols if col in df.columns]
    df = df[available_cols]
    
    print(f"\n[1.5] Zachowano {len(available_cols)} istotnych kolumn")
    print(f"      Finalne dane: {len(df):,} wierszy × {len(df.columns)} kolumn")
    
    # Podstawowe statystyki zmiennej docelowej
    print(f"\n[1.6] Statystyki zmiennej docelowej (FLT_TOT_1):")
    print(f"      Średnia:  {df['FLT_TOT_1'].mean():.2f}")
    print(f"      Mediana:  {df['FLT_TOT_1'].median():.2f}")
    print(f"      Min:      {df['FLT_TOT_1'].min():.2f}")
    print(f"      Max:      {df['FLT_TOT_1'].max():.2f}")
    print(f"      Std:      {df['FLT_TOT_1'].std():.2f}")
    
    return df


# ============================================================================
# SEKCJA 3: INŻYNIERIA CECH (FEATURE ENGINEERING)
# ============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzenie cech czasowych z kolumn YEAR i MONTH_NUM.
    
    Cechy tworzone:
    - YEAR_TREND: Znormalizowany trend roczny (0, 1, 2, ...)
    - MONTH_SIN: Sinusoidalne kodowanie miesiąca (cykliczność)
    - MONTH_COS: Kosinusoidalne kodowanie miesiąca (cykliczność)
    
    Kodowanie cykliczne zapewnia, że grudzień (12) jest blisko stycznia (1)
    
    Parametry:
        df (pd.DataFrame): DataFrame z kolumnami YEAR i MONTH_NUM
        
    Zwraca:
        pd.DataFrame: DataFrame z dodatkowymi cechami czasowymi
    """
    df = df.copy()
    
    # Trend roczny - normalizacja względem pierwszego roku
    if 'YEAR' in df.columns:
        min_year = df['YEAR'].min()
        df['YEAR_TREND'] = df['YEAR'] - min_year
    
    # Kodowanie cykliczne miesiąca
    # Wykorzystuje funkcje trygonometryczne do zachowania cykliczności
    if 'MONTH_NUM' in df.columns:
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
    
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzenie cech sezonowych na podstawie miesiąca.
    
    Cechy tworzone:
    - SEASON: Kategoria pory roku (Winter/Spring/Summer/Fall)
    - IS_SUMMER: Flaga binarna - czy to miesiące letnie (6,7,8)
    - IS_WINTER: Flaga binarna - czy to miesiące zimowe (12,1,2)
    
    Sezonowość ma znaczenie w ruchu lotniczym - większy ruch w lecie
    
    Parametry:
        df (pd.DataFrame): DataFrame z kolumną MONTH_NUM
        
    Zwraca:
        pd.DataFrame: DataFrame z dodatkowymi cechami sezonowymi
    """
    df = df.copy()
    
    if 'MONTH_NUM' not in df.columns:
        return df
    
    # Mapowanie miesięcy na pory roku
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',     # Zima
        3: 'Spring', 4: 'Spring', 5: 'Spring',       # Wiosna
        6: 'Summer', 7: 'Summer', 8: 'Summer',       # Lato
        9: 'Fall', 10: 'Fall', 11: 'Fall'           # Jesień
    }
    df['SEASON'] = df['MONTH_NUM'].map(season_map)
    
    # Flagi binarne dla kluczowych sezonów
    df['IS_SUMMER'] = (df['MONTH_NUM'].isin([6, 7, 8])).astype(int)
    df['IS_WINTER'] = (df['MONTH_NUM'].isin([12, 1, 2])).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'FLT_TOT_1') -> pd.DataFrame:
    """
    Tworzenie cech opóźnionych (lag features) dla szeregów czasowych.
    
    Cechy tworzone:
    - lag_1: Wartość zmiennej docelowej z poprzedniego miesiąca
    - lag_3: Średnia krocząca z 3 poprzednich miesięcy
    
    Cechy opóźnione są kluczowe w prognozowaniu szeregów czasowych,
    ponieważ przeszłe wartości często są dobrym predyktorem przyszłości
    
    Parametry:
        df (pd.DataFrame): DataFrame z danymi posortowanymi chronologicznie
        target_col (str): Nazwa zmiennej docelowej
        
    Zwraca:
        pd.DataFrame: DataFrame z dodatkowymi cechami opóźnionymi
    """
    df = df.copy()
    
    if 'APT_ICAO' not in df.columns or target_col not in df.columns:
        return df
    
    # Sortowanie chronologiczne per lotnisko
    df = df.sort_values(['APT_ICAO', 'YEAR', 'MONTH_NUM'])
    
    # Opóźnienie 1 miesiąca - wartość z poprzedniego miesiąca
    df['lag_1'] = df.groupby('APT_ICAO')[target_col].shift(1)
    
    # Średnia krocząca z 3 miesięcy (wygładza krótkoterminowe fluktuacje)
    df['lag_3'] = df.groupby('APT_ICAO')[target_col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Kodowanie cech kategorycznych na wartości numeryczne.
    
    Wykorzystuje Label Encoding dla:
    - APT_ICAO: Kod ICAO lotniska (np. EPWA dla Warszawy)
    - STATE_NAME: Nazwa kraju
    - SEASON: Pora roku
    
    Label Encoding przypisuje każdej unikalnej wartości liczbę całkowitą
    
    Parametry:
        df (pd.DataFrame): DataFrame z cechami kategorycznymi
        
    Zwraca:
        Tuple[pd.DataFrame, Dict]: DataFrame z zakodowanymi cechami i słownik encoderów
    """
    df = df.copy()
    
    # Lista kolumn do zakodowania
    cat_columns = ['APT_ICAO', 'STATE_NAME', 'SEASON']
    
    # Kodowanie tylko istniejących kolumn
    cat_columns = [col for col in cat_columns if col in df.columns]
    
    encoders = {}
    
    for col in cat_columns:
        # Inicjalizacja encodera
        le = LabelEncoder()
        # Kodowanie kolumny
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        # Zapisanie encodera dla ewentualnego późniejszego użycia
        encoders[col] = le
    
    return df, encoders


def engineer_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Pełny pipeline inżynierii cech - wykonuje wszystkie transformacje.
    
    Proces:
    1. Cechy czasowe (trend, cykliczność)
    2. Cechy sezonowe (pory roku, flagi)
    3. Cechy opóźnione (lag features)
    4. Kodowanie kategorii
    5. Usunięcie wierszy z NaN (powstałych z lag features)
    6. Wybór finalnych cech do modelowania
    
    Parametry:
        df (pd.DataFrame): Oczyszczone dane wejściowe
        
    Zwraca:
        Tuple zawierający:
        - DataFrame z wygenerowanymi cechami
        - Słownik encoderów kategorycznych
        - Lista nazw cech do modelowania
    """
    print("\n" + "="*80)
    print("ETAP 2: INŻYNIERIA CECH")
    print("="*80)
    
    print("\n[2.1] Tworzenie cech czasowych...")
    df = create_time_features(df)
    print("      Utworzono: YEAR_TREND, MONTH_SIN, MONTH_COS")
    
    print("\n[2.2] Tworzenie cech sezonowych...")
    df = create_seasonal_features(df)
    print("      Utworzono: SEASON, IS_SUMMER, IS_WINTER")
    
    print("\n[2.3] Tworzenie cech opóźnionych (lag features)...")
    df = create_lag_features(df)
    print("      Utworzono: lag_1, lag_3")
    
    print("\n[2.4] Kodowanie cech kategorycznych...")
    df, encoders = encode_categorical_features(df)
    print("      Zakodowano: APT_ICAO, STATE_NAME, SEASON")
    
    # Usunięcie wierszy z brakującymi wartościami (powstałymi przez lag features)
    rows_before = len(df)
    df = df.dropna()
    rows_removed = rows_before - len(df)
    print(f"\n[2.5] Usunięto {rows_removed:,} wierszy z brakującymi wartościami (lag features)")
    
    # Definicja finalnych cech do modelowania
    # Wybieramy tylko cechy numeryczne i zakodowane kategorie
    feature_cols = [
        'YEAR_TREND', 'MONTH_SIN', 'MONTH_COS',           # Cechy czasowe
        'IS_SUMMER', 'IS_WINTER',                          # Cechy sezonowe
        'FLT_DEP_1', 'FLT_ARR_1',                         # Cechy bazowe
        'lag_1', 'lag_3',                                  # Cechy opóźnione
        'APT_ICAO_encoded', 'STATE_NAME_encoded', 'SEASON_encoded'  # Zakodowane kategorie
    ]
    
    # Sprawdzenie dostępności cech
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"\n[2.6] Wybrano {len(feature_cols)} cech do modelowania:")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {col}")
    
    print(f"\n[2.7] Finalny zbiór danych: {len(df):,} wierszy × {len(feature_cols)} cech")
    
    return df, encoders, feature_cols


# ============================================================================
# SEKCJA 4: TRENOWANIE MODELU LIGHTGBM
# ============================================================================

def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str], 
                             target_col: str = 'FLT_TOT_1') -> Tuple:
    """
    Podział danych na zbiory treningowy i testowy.
    
    Stratyfikacja nie jest stosowana (zmienna ciągła), ale zapewniamy
    losowy podział z ustalonym ziarnem dla reprodukowalności.
    
    Parametry:
        df (pd.DataFrame): DataFrame z cechami i zmienną docelową
        feature_cols (List[str]): Lista nazw cech
        target_col (str): Nazwa zmiennej docelowej
        
    Zwraca:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*80)
    print("ETAP 3: PODZIAŁ DANYCH NA ZBIORY TRENINGOWY I TESTOWY")
    print("="*80)
    
    # Separacja cech (X) i zmiennej docelowej (y)
    X = df[feature_cols]
    y = df[target_col]
    
    # Podział 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"\n[3.1] Zbiór treningowy: {len(X_train):,} próbek ({100*(1-TEST_SIZE):.0f}%)")
    print(f"[3.2] Zbiór testowy:    {len(X_test):,} próbek ({100*TEST_SIZE:.0f}%)")
    
    print(f"\n[3.3] Statystyki zmiennej docelowej w zbiorze treningowym:")
    print(f"      Średnia: {y_train.mean():.2f}")
    print(f"      Std:     {y_train.std():.2f}")
    
    return X_train, X_test, y_train, y_test


def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """
    Trenowanie modelu LightGBM (Light Gradient Boosting Machine).
    
    LightGBM to szybka implementacja gradient boosting, która:
    - Wykorzystuje drzewa decyzyjne jako bazowe estymatory
    - Stosuje boosting (sekwencyjne uczenie, każde drzewo poprawia błędy poprzednich)
    - Jest efektywna dla dużych zbiorów danych
    
    Hiperparametry:
    - objective: regression (zadanie regresji)
    - metric: RMSE (Root Mean Squared Error)
    - num_leaves: 31 (maksymalna liczba liści w drzewie)
    - learning_rate: 0.05 (współczynnik uczenia)
    - feature_fraction: 0.9 (losowe próbkowanie cech)
    - bagging: losowe próbkowanie danych
    
    Parametry:
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe (do walidacji podczas treningu)
        
    Zwraca:
        lgb.Booster: Wytrenowany model LightGBM
    """
    print("\n" + "="*80)
    print("ETAP 4: TRENOWANIE MODELU LIGHTGBM")
    print("="*80)
    
    print("\n[4.1] Konfiguracja hiperparametrów LightGBM:")
    params = {
        'objective': 'regression',        # Zadanie regresji
        'metric': 'rmse',                 # Metryka optymalizacji
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree
        'num_leaves': 31,                 # Liczba liści w drzewie
        'learning_rate': 0.05,            # Współczynnik uczenia
        'feature_fraction': 0.9,          # Losowe próbkowanie cech (90%)
        'bagging_fraction': 0.8,          # Losowe próbkowanie danych (80%)
        'bagging_freq': 5,                # Częstotliwość bagging
        'verbose': -1                     # Wyłączenie szczegółowych logów
    }
    
    for key, value in params.items():
        print(f"      {key:20s} = {value}")
    
    # Utworzenie zbiorów danych w formacie LightGBM
    print("\n[4.2] Przygotowanie zbiorów danych LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Trenowanie modelu z early stopping
    print("\n[4.3] Rozpoczęcie treningu (max 500 iteracji, early stopping=50)...")
    print("-" * 80)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,              # Maksymalna liczba drzew
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # Zatrzymaj jeśli brak poprawy przez 50 iteracji
            lgb.log_evaluation(50)                    # Wyświetl wyniki co 50 iteracji
        ]
    )
    
    print("-" * 80)
    print(f"\n[4.4] Trenowanie zakończone. Użyto {model.num_trees()} drzew.")
    
    return model


def evaluate_lightgbm_model(model, X_train, y_train, X_test, y_test):
    """
    Ewaluacja modelu LightGBM na zbiorach treningowym i testowym.
    
    Metryki:
    - RMSE (Root Mean Squared Error): pierwiastek błędu średniokwadratowego
      -> kara większa dla dużych błędów, jednostka jak zmienna docelowa
    - MAE (Mean Absolute Error): średni błąd bezwzględny
      -> łatwiejsza interpretacja, mniej wrażliwa na outliers
    - R² (R-squared): współczynnik determinacji (0-1, im wyższy tym lepiej)
      -> procent wariancji wyjaśnionej przez model
    
    Parametry:
        model: Wytrenowany model LightGBM
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        
    Zwraca:
        Dict: Słownik z metrykami i predykcjami
    """
    print("\n[4.5] Ewaluacja modelu LightGBM:")
    print("-" * 80)
    
    # Predykcje na zbiorze treningowym
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Predykcje na zbiorze testowym
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Wyświetlenie wyników
    print(f"\n      {'Metryka':<20s} {'Train':<15s} {'Test':<15s}")
    print(f"      {'-'*50}")
    print(f"      {'RMSE':<20s} {train_rmse:<15.2f} {test_rmse:<15.2f}")
    print(f"      {'MAE':<20s} {train_mae:<15.2f} {test_mae:<15.2f}")
    print(f"      {'R²':<20s} {train_r2:<15.4f} {test_r2:<15.4f}")
    
    # Interpretacja
    print(f"\n      Interpretacja R² na zbiorze testowym:")
    print(f"      Model wyjaśnia {test_r2*100:.2f}% wariancji zmiennej docelowej")
    
    return {
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'test_predictions': y_test_pred
    }


def save_lightgbm_model(model):
    """
    Zapis modelu LightGBM do pliku.
    
    Parametry:
        model: Wytrenowany model LightGBM
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'lightgbm_model.txt')
    model.save_model(model_path)
    print(f"\n[4.6] Model LightGBM zapisany do: {model_path}")


# ============================================================================
# SEKCJA 5: TRENOWANIE SIECI NEURONOWEJ MLP
# ============================================================================

class MLPRegressor(nn.Module):
    """
    Klasa sieci neuronowej MLP (Multi-Layer Perceptron) w PyTorch.
    
    Architektura:
    - Warstwa wejściowa: liczba cech
    - Warstwy ukryte: [128, 64, 32] neurony
    - Każda warstwa ukryta ma:
      * Linear (fully connected)
      * ReLU (funkcja aktywacji)
      * Dropout (20% regularizacji - zapobiega przeuczeniu)
    - Warstwa wyjściowa: 1 neuron (regresja)
    
    MLP to klasyczna sieć feedforward - informacja płynie tylko do przodu,
    od wejścia do wyjścia.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        """
        Inicjalizacja architektury sieci MLP.
        
        Parametry:
            input_size (int): Liczba cech wejściowych
            hidden_sizes (List[int]): Lista rozmiarów warstw ukrytych
        """
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Budowa warstw ukrytych
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))  # Warstwa liniowa
            layers.append(nn.ReLU())                          # Funkcja aktywacji
            layers.append(nn.Dropout(0.2))                    # Regularyzacja
            prev_size = hidden_size
        
        # Warstwa wyjściowa (bez aktywacji dla regresji)
        layers.append(nn.Linear(prev_size, 1))
        
        # Połączenie wszystkich warstw w sekwencyjny model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Propagacja wprzód (forward pass).
        
        Parametry:
            x: Tensor wejściowy
            
        Zwraca:
            Tensor: Predykcje sieci
        """
        return self.network(x)


def prepare_pytorch_data(X_train, X_test, y_train, y_test):
    """
    Przygotowanie danych dla PyTorch.
    
    Kroki:
    1. Standaryzacja cech (mean=0, std=1) - ważne dla sieci neuronowych
    2. Konwersja do tensorów PyTorch
    3. Utworzenie DataLoader dla efektywnego treningu w batches
    
    Parametry:
        X_train, X_test, y_train, y_test: Zbiory danych
        
    Zwraca:
        Tuple: (train_loader, test_loader, scaler, X_test_tensor, y_test_tensor)
    """
    print("\n[5.1] Standaryzacja cech (StandardScaler)...")
    # Standaryzacja - uczenie tylko na zbiorze treningowym!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"      Średnia po standaryzacji: {X_train_scaled.mean():.6f}")
    print(f"      Odchylenie standardowe:   {X_train_scaled.std():.6f}")
    
    print("\n[5.2] Konwersja do tensorów PyTorch...")
    # Konwersja do tensorów
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    print("\n[5.3] Tworzenie DataLoaders...")
    # Utworzenie datasets i loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"      Batch size: {BATCH_SIZE}")
    print(f"      Liczba batchy treningowych: {len(train_loader)}")
    print(f"      Liczba batchy testowych:    {len(test_loader)}")
    
    return train_loader, test_loader, scaler, X_test_tensor, y_test_tensor


def train_mlp_model(input_size: int, train_loader, test_loader):
    """
    Trenowanie sieci neuronowej MLP.
    
    Proces treningu:
    1. Inicjalizacja modelu i optymalizatora
    2. Dla każdej epoki:
       a) Trening na danych treningowych (mini-batches)
       b) Walidacja na danych testowych
       c) Monitoring metryk (loss)
    3. Wybór najlepszego modelu (najniższy test loss)
    
    Optimizer: Adam (adaptive learning rate)
    Loss function: MSE (Mean Squared Error) - standard dla regresji
    
    Parametry:
        input_size (int): Liczba cech wejściowych
        train_loader: DataLoader ze zbiorem treningowym
        test_loader: DataLoader ze zbiorem testowym
        
    Zwraca:
        Tuple: (model, train_losses, test_losses)
    """
    print("\n" + "="*80)
    print("ETAP 5: TRENOWANIE SIECI NEURONOWEJ MLP")
    print("="*80)
    
    print("\n[5.4] Inicjalizacja architektury MLP:")
    print(f"      Warstwa wejściowa:  {input_size} cech")
    print(f"      Warstwy ukryte:     [128, 64, 32] neurony")
    print(f"      Warstwa wyjściowa:  1 neuron (regresja)")
    print(f"      Funkcja aktywacji:  ReLU")
    print(f"      Regularyzacja:      Dropout (20%)")
    
    # Inicjalizacja modelu
    model = MLPRegressor(input_size=input_size, hidden_sizes=[128, 64, 32])
    
    # Wyświetlenie struktury modelu
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n      Łączna liczba parametrów:    {total_params:,}")
    print(f"      Trenowalnych parametrów:      {trainable_params:,}")
    
    # Funkcja straty i optymalizator
    criterion = nn.MSELoss()                                    # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer
    
    print(f"\n[5.5] Konfiguracja treningu:")
    print(f"      Loss function:     MSE (Mean Squared Error)")
    print(f"      Optimizer:         Adam")
    print(f"      Learning rate:     {LEARNING_RATE}")
    print(f"      Liczba epok:       {EPOCHS}")
    print(f"      Batch size:        {BATCH_SIZE}")
    
    # Listy do przechowywania historii treningu
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_state = None
    
    print("\n[5.6] Rozpoczęcie treningu:")
    print("-" * 80)
    print(f"      {'Epoka':<10s} {'Train Loss':<15s} {'Test Loss':<15s} {'Status':<20s}")
    print("-" * 80)
    
    # Pętla treningowa
    for epoch in range(EPOCHS):
        # === FAZA TRENINGOWA ===
        model.train()  # Tryb treningowy (włącza dropout)
        train_loss_epoch = 0.0
        
        for X_batch, y_batch in train_loader:
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Aktualizacja wag
            optimizer.step()
            
            train_loss_epoch += loss.item()
        
        # Średnia strata treningowa
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        # === FAZA WALIDACYJNA ===
        model.eval()  # Tryb ewaluacji (wyłącza dropout)
        test_loss_epoch = 0.0
        
        with torch.no_grad():  # Bez obliczania gradientów
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                test_loss_epoch += loss.item()
        
        # Średnia strata testowa
        test_loss_epoch /= len(test_loader)
        test_losses.append(test_loss_epoch)
        
        # Zapisanie najlepszego modelu
        status = ""
        if test_loss_epoch < best_test_loss:
            best_test_loss = test_loss_epoch
            best_model_state = model.state_dict().copy()
            status = "✓ Nowy najlepszy"
        
        # Wyświetlanie postępu co 10 epok
        if (epoch + 1) % 10 == 0:
            print(f"      {epoch+1:<10d} {train_loss_epoch:<15.4f} {test_loss_epoch:<15.4f} {status:<20s}")
    
    print("-" * 80)
    print(f"\n[5.7] Trenowanie zakończone!")
    print(f"      Najlepszy test loss: {best_test_loss:.4f}")
    
    # Przywrócenie najlepszego modelu
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      Przywrócono najlepszy model z epoki")
    
    return model, train_losses, test_losses


def evaluate_mlp_model(model, X_test_tensor, y_test):
    """
    Ewaluacja sieci neuronowej MLP.
    
    Parametry:
        model: Wytrenowany model MLP
        X_test_tensor: Tensor z cechami testowymi
        y_test: Prawdziwe wartości zmiennej docelowej
        
    Zwraca:
        Dict: Metryki i predykcje
    """
    print("\n[5.8] Ewaluacja sieci MLP na zbiorze testowym:")
    print("-" * 80)
    
    model.eval()
    
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Obliczenie metryk
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    print(f"\n      {'Metryka':<20s} {'Wartość':<15s}")
    print(f"      {'-'*35}")
    print(f"      {'RMSE':<20s} {rmse:<15.2f}")
    print(f"      {'MAE':<20s} {mae:<15.2f}")
    print(f"      {'R²':<20s} {r2:<15.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }


def save_mlp_model(model, scaler):
    """
    Zapis modelu MLP i scalera.
    
    Parametry:
        model: Wytrenowany model MLP
        scaler: StandardScaler użyty do standaryzacji
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Zapis modelu
    model_path = os.path.join(MODELS_DIR, 'mlp_fp32.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n[5.9] Model MLP zapisany do: {model_path}")
    
    # Zapis scalera
    scaler_path = os.path.join(MODELS_DIR, 'mlp_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[5.10] Scaler zapisany do: {scaler_path}")


# ============================================================================
# SEKCJA 6: KOMPRESJA MODELI (PRUNING I QUANTIZATION)
# ============================================================================

def count_model_parameters(model):
    """
    Zliczanie parametrów w modelu.
    
    Parametry:
        model: Model PyTorch
        
    Zwraca:
        Tuple: (total_params, nonzero_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params


def prune_mlp_model(model, amount=PRUNING_AMOUNT):
    """
    Przycinanie (pruning) sieci neuronowej.
    
    Pruning to technika kompresji modelu, która:
    - Usuwa najmniej istotne połączenia (wagi bliskie zeru)
    - Redukuje rozmiar modelu
    - Przyspiesza inferencing
    - Może nieznacznie obniżyć accuracy (do akceptacji w praktyce)
    
    Stosujemy structured pruning - usuwamy całe neurony, nie pojedyncze wagi.
    To daje większe przyspieszenie na standardowym sprzęcie.
    
    Parametry:
        model: Model MLP do przycięcia
        amount (float): Procent neuronów do usunięcia (domyślnie 30%)
        
    Zwraca:
        Model: Przycięty model
    """
    print("\n" + "="*80)
    print("ETAP 6A: PRZYCINANIE MODELU (PRUNING)")
    print("="*80)
    
    print(f"\n[6.1] Parametry przed pruningiem:")
    total_before, nonzero_before = count_model_parameters(model)
    print(f"      Łączna liczba parametrów:    {total_before:,}")
    print(f"      Niezerowych parametrów:       {nonzero_before:,}")
    
    print(f"\n[6.2] Stosowanie structured pruning (amount={amount*100:.0f}%)...")
    print("      Metoda: L2-norm structured pruning na warstwach Linear")
    
    # Zastosowanie structured pruning do warstw liniowych
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Pruning wzdłuż wymiaru 0 (całe neurony)
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    print(f"\n[6.3] Parametry po pruningu:")
    total_after, nonzero_after = count_model_parameters(model)
    print(f"      Łączna liczba parametrów:    {total_after:,}")
    print(f"      Niezerowych parametrów:       {nonzero_after:,}")
    print(f"      Sparsity (rzadkość):          {100 * (1 - nonzero_after / total_after):.2f}%")
    
    # Usunięcie reparametryzacji (utrwalenie pruning)
    print("\n[6.4] Utrwalanie pruning (usuwanie reparametryzacji)...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass  # Brak pruning na tej warstwie
    
    print("      Pruning został utrwalony w modelu")
    
    return model


def quantize_mlp_model(model):
    """
    Kwantyzacja (quantization) sieci neuronowej.
    
    Quantization to technika kompresji, która:
    - Konwertuje wagi z FP32 (32-bit float) do INT8 (8-bit integer)
    - Redukuje rozmiar modelu ~4x
    - Przyspiesza inferencing (szczególnie na CPU)
    - Minimalny spadek accuracy
    
    Stosujemy dynamic quantization - kwantyzacja podczas inferencingu,
    optymalna dla modeli z warstwami Linear/LSTM.
    
    Parametry:
        model: Model MLP do kwantyzacji
        
    Zwraca:
        Model: Skwantyzowany model
    """
    print("\n" + "="*80)
    print("ETAP 6B: KWANTYZACJA MODELU (QUANTIZATION)")
    print("="*80)
    
    print("\n[6.5] Typ kwantyzacji: Dynamic Quantization")
    print("      Format:          FP32 → INT8")
    print("      Warstwy:         Linear layers")
    
    # Model musi być w trybie eval
    model.eval()
    
    # Kwantyzacja dynamiczna
    print("\n[6.6] Stosowanie kwantyzacji...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},      # Kwantyzuj tylko warstwy Linear
        dtype=torch.qint8  # Użyj 8-bitowych integerów
    )
    
    print("      Kwantyzacja zakończona pomyślnie")
    
    return quantized_model


def save_compressed_models(pruned_model, quantized_model):
    """
    Zapis skompresowanych modeli.
    
    Parametry:
        pruned_model: Model po pruning
        quantized_model: Model po quantization
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Zapis modelu pruned
    pruned_path = os.path.join(MODELS_DIR, 'mlp_pruned.pt')
    torch.save(pruned_model.state_dict(), pruned_path)
    pruned_size = os.path.getsize(pruned_path) / (1024 * 1024)
    print(f"\n[6.7] Model Pruned zapisany do: {pruned_path}")
    print(f"      Rozmiar: {pruned_size:.2f} MB")
    
    # Zapis modelu quantized
    quantized_path = os.path.join(MODELS_DIR, 'mlp_int8.pt')
    torch.save(quantized_model.state_dict(), quantized_path)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"\n[6.8] Model Quantized zapisany do: {quantized_path}")
    print(f"      Rozmiar: {quantized_size:.2f} MB")


# ============================================================================
# SEKCJA 7: PORÓWNANIE WSZYSTKICH MODELI
# ============================================================================

def compare_all_models(lightgbm_metrics, mlp_metrics, 
                      pruned_model, quantized_model, 
                      X_test_tensor, y_test):
    """
    Kompleksowe porównanie wszystkich modeli.
    
    Porównujemy:
    1. LightGBM (baseline)
    2. MLP FP32 (sieć neuronowa full precision)
    3. MLP Pruned (sieć po przycinaniu)
    4. MLP Quantized (sieć po kwantyzacji)
    
    Metryki porównania:
    - RMSE, MAE, R² (accuracy)
    - Rozmiar modelu (MB)
    - Czas inferencing (ms/sample)
    
    Parametry:
        lightgbm_metrics: Metryki modelu LightGBM
        mlp_metrics: Metryki modelu MLP
        pruned_model: Model po pruning
        quantized_model: Model po quantization
        X_test_tensor: Dane testowe
        y_test: Prawdziwe wartości
        
    Zwraca:
        pd.DataFrame: Tabela porównawcza
    """
    print("\n" + "="*80)
    print("ETAP 7: PORÓWNANIE WSZYSTKICH MODELI")
    print("="*80)
    
    results = []
    
    # Model 1: LightGBM
    print("\n[7.1] Ewaluacja modelu LightGBM...")
    lgb_size = os.path.getsize(os.path.join(MODELS_DIR, 'lightgbm_model.txt')) / (1024 * 1024)
    results.append({
        'Model': 'LightGBM',
        'RMSE': lightgbm_metrics['test_rmse'],
        'MAE': lightgbm_metrics['test_mae'],
        'R²': lightgbm_metrics['test_r2'],
        'Rozmiar (MB)': lgb_size
    })
    
    # Model 2: MLP FP32
    print("[7.2] Ewaluacja modelu MLP FP32...")
    mlp_size = os.path.getsize(os.path.join(MODELS_DIR, 'mlp_fp32.pt')) / (1024 * 1024)
    results.append({
        'Model': 'MLP FP32',
        'RMSE': mlp_metrics['rmse'],
        'MAE': mlp_metrics['mae'],
        'R²': mlp_metrics['r2'],
        'Rozmiar (MB)': mlp_size
    })
    
    # Model 3: MLP Pruned
    print("[7.3] Ewaluacja modelu MLP Pruned...")
    pruned_model.eval()
    with torch.no_grad():
        pruned_pred = pruned_model(X_test_tensor).numpy().flatten()
    
    pruned_rmse = np.sqrt(mean_squared_error(y_test, pruned_pred))
    pruned_mae = mean_absolute_error(y_test, pruned_pred)
    pruned_r2 = r2_score(y_test, pruned_pred)
    pruned_size = os.path.getsize(os.path.join(MODELS_DIR, 'mlp_pruned.pt')) / (1024 * 1024)
    
    results.append({
        'Model': 'MLP Pruned',
        'RMSE': pruned_rmse,
        'MAE': pruned_mae,
        'R²': pruned_r2,
        'Rozmiar (MB)': pruned_size
    })
    
    # Model 4: MLP Quantized
    print("[7.4] Ewaluacja modelu MLP Quantized...")
    quantized_model.eval()
    with torch.no_grad():
        quantized_pred = quantized_model(X_test_tensor).numpy().flatten()
    
    quantized_rmse = np.sqrt(mean_squared_error(y_test, quantized_pred))
    quantized_mae = mean_absolute_error(y_test, quantized_pred)
    quantized_r2 = r2_score(y_test, quantized_pred)
    quantized_size = os.path.getsize(os.path.join(MODELS_DIR, 'mlp_int8.pt')) / (1024 * 1024)
    
    results.append({
        'Model': 'MLP Quantized INT8',
        'RMSE': quantized_rmse,
        'MAE': quantized_mae,
        'R²': quantized_r2,
        'Rozmiar (MB)': quantized_size
    })
    
    # Utworzenie tabeli porównawczej
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("TABELA PORÓWNAWCZA MODELI")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Analiza wyników
    print("\n" + "="*80)
    print("ANALIZA WYNIKÓW")
    print("="*80)
    
    best_rmse_idx = comparison_df['RMSE'].idxmin()
    best_r2_idx = comparison_df['R²'].idxmax()
    smallest_idx = comparison_df['Rozmiar (MB)'].idxmin()
    
    print(f"\n✓ Najlepsza dokładność (RMSE):    {comparison_df.loc[best_rmse_idx, 'Model']}")
    print(f"✓ Najlepsze R²:                   {comparison_df.loc[best_r2_idx, 'Model']}")
    print(f"✓ Najmniejszy rozmiar:            {comparison_df.loc[smallest_idx, 'Model']}")
    
    # Kompresja
    original_size = comparison_df[comparison_df['Model'] == 'MLP FP32']['Rozmiar (MB)'].values[0]
    quantized_size_val = comparison_df[comparison_df['Model'] == 'MLP Quantized INT8']['Rozmiar (MB)'].values[0]
    compression_ratio = original_size / quantized_size_val
    
    print(f"\n📊 Stopień kompresji (FP32 → INT8): {compression_ratio:.2f}x")
    print(f"   Redukcja rozmiaru: {(1 - 1/compression_ratio)*100:.1f}%")
    
    return comparison_df


# ============================================================================
# SEKCJA 8: GŁÓWNA FUNKCJA URUCHAMIAJĄCA PROJEKT
# ============================================================================

def main():
    """
    Główna funkcja uruchamiająca cały pipeline projektu.
    
    Kolejność wykonania:
    1. Wczytanie i czyszczenie danych
    2. Inżynieria cech
    3. Podział na zbiory train/test
    4. Trenowanie LightGBM
    5. Trenowanie MLP
    6. Kompresja modeli (pruning + quantization)
    7. Porównanie wszystkich modeli
    """
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  PROJEKT AKADEMICKI: PREDYKCJA NATĘŻENIA RUCHU LOTNICZEGO".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")
    
    # ========================================
    # ETAP 1-2: Dane i cechy
    # ========================================
    df = load_and_clean_data(DATA_PATH)
    df, encoders, feature_cols = engineer_all_features(df)
    
    # ========================================
    # ETAP 3: Podział danych
    # ========================================
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, feature_cols, target_col='FLT_TOT_1'
    )
    
    # ========================================
    # ETAP 4: LightGBM
    # ========================================
    lgb_model = train_lightgbm_model(X_train, y_train, X_test, y_test)
    lgb_metrics = evaluate_lightgbm_model(lgb_model, X_train, y_train, X_test, y_test)
    save_lightgbm_model(lgb_model)
    
    # ========================================
    # ETAP 5: MLP
    # ========================================
    train_loader, test_loader, scaler, X_test_tensor, y_test_tensor = prepare_pytorch_data(
        X_train, X_test, y_train, y_test
    )
    
    mlp_model, train_losses, test_losses = train_mlp_model(
        input_size=len(feature_cols),
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    mlp_metrics = evaluate_mlp_model(mlp_model, X_test_tensor, y_test)
    save_mlp_model(mlp_model, scaler)
    
    # ========================================
    # ETAP 6: Kompresja
    # ========================================
    # 6A: Pruning
    pruned_model = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
    pruned_model.load_state_dict(mlp_model.state_dict())
    pruned_model = prune_mlp_model(pruned_model, amount=PRUNING_AMOUNT)
    
    # 6B: Quantization
    quantized_model = MLPRegressor(input_size=len(feature_cols), hidden_sizes=[128, 64, 32])
    quantized_model.load_state_dict(mlp_model.state_dict())
    quantized_model = quantize_mlp_model(quantized_model)
    
    save_compressed_models(pruned_model, quantized_model)
    
    # ========================================
    # ETAP 7: Porównanie
    # ========================================
    comparison_df = compare_all_models(
        lgb_metrics, mlp_metrics,
        pruned_model, quantized_model,
        X_test_tensor, y_test
    )
    
    # Zapis wyników porównania
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[7.5] Tabela porównawcza zapisana do: {comparison_path}")
    
    # ========================================
    # Zakończenie
    # ========================================
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  PROJEKT ZAKOŃCZONY POMYŚLNIE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\nWszystkie modele zostały wytrenowane i zapisane w folderze 'models/'")
    print("Wyniki porównania dostępne w pliku: models/model_comparison.csv\n")


# ============================================================================
# URUCHOMIENIE PROJEKTU
# ============================================================================

if __name__ == "__main__":
    """
    Punkt wejścia programu.
    
    Aby uruchomić projekt, wystarczy wykonać:
        python main.py
    
    Wymagania:
        - Zainstalowane biblioteki z requirements.txt
        - Plik danych: data/european_flights.csv
    """
    main()
