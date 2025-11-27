# ✈️ Air Traffic Volume Prediction

Machine learning project for predicting air traffic volume at European airports using **LightGBM** and **PyTorch MLP** models, with **model compression** techniques (pruning & quantization).

## Project Overview

This project predicts **IFR (Instrument Flight Rules) movements** at European airports using monthly aggregated flight data. It implements:

- **Baseline Model**: LightGBM regressor
- **Neural Network**: Multi-Layer Perceptron (MLP) in PyTorch
- **Model Compression**: Structured pruning and INT8 quantization
- **Comprehensive Evaluation**: Performance comparison across all models

## Project Structure

```
Air_Traffic_Volume_Prediction/
├── data/
│   └── european_flights.csv          # Dataset (monthly IFR statistics)
├── models/                            # Trained models and results
│   ├── lightgbm_model.txt            # LightGBM model
│   ├── mlp_fp32.pt                   # MLP FP32 model
│   ├── mlp_pruned.pt                 # Pruned MLP model
│   ├── mlp_int8.pt                   # Quantized MLP model
│   └── mlp_scaler.pkl                # Feature scaler
├── src/
│   ├── preprocessing.py              # Data loading and cleaning
│   ├── feature_engineering.py        # Feature creation
│   ├── train_lightgbm.py            # LightGBM training
│   ├── train_mlp.py                 # MLP training
│   ├── pruning.py                   # Model pruning
│   ├── quantization.py              # Model quantization
│   └── evaluate.py                  # Model comparison
├── notebooks/                         # Jupyter notebooks for exploration
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## **Quick Start**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `european_flights.csv` file in the `data/` directory.

### 3. Run the Complete Pipeline

Execute the scripts in order:

```bash
# Train LightGBM baseline
cd src
python train_lightgbm.py

# Train MLP model
python train_mlp.py

# Apply pruning
python pruning.py

# Apply quantization
python quantization.py

# Compare all models
python evaluate.py
```

## Dataset

**Source**: European Flights Dataset  
**Format**: Monthly aggregated IFR statistics per airport

**Key Columns**:
- `YEAR`, `MONTH_NUM` - Time features
- `APT_ICAO` - Airport ICAO code
- `STATE_NAME` - Country
- `FLT_TOT_1` - **Target variable** (total IFR movements)
- `FLT_DEP_1`, `FLT_ARR_1` - Departures and arrivals

## Features

The feature engineering pipeline creates:

### Time Features
- `YEAR_TREND` - Normalized year index
- `MONTH_SIN`, `MONTH_COS` - Cyclical month encoding

### Seasonal Features
- `SEASON` - Winter/Spring/Summer/Fall
- `IS_SUMMER`, `IS_WINTER` - Binary seasonal flags

### Lag Features
- `lag_1` - Previous month traffic
- `lag_3` - 3-month rolling average

### Categorical Encodings
- Encoded `APT_ICAO` (airport)
- Encoded `STATE_NAME` (country)
- Encoded `SEASON`

## Models

### 1. LightGBM Regressor (Baseline)
- **Type**: Gradient boosting trees
- **Hyperparameters**: 
  - Learning rate: 0.05
  - Num leaves: 31
  - Early stopping: 50 rounds
- **Output**: Feature importance plot

### 2. MLP (Multi-Layer Perceptron)
- **Architecture**: Input → [128, 64, 32] → Output(1)
- **Activation**: ReLU
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE
- **Epochs**: 50
- **Dropout**: 0.2

### 3. Pruned MLP
- **Method**: Structured L-n pruning
- **Pruning amount**: 30% of neurons
- **Fine-tuning**: 10 epochs at lr=0.0001

### 4. Quantized MLP
- **Method**: Dynamic quantization
- **Type**: INT8 (from FP32)
- **Backend**: fbgemm (CPU)
- **Target layers**: Linear layers

## Evaluation Metrics

Each model is evaluated on:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of determination)
- **Model Size** (MB)
- **Inference Time** (ms/sample)

## Visualizations

The evaluation script generates:

1. **Comparison Table** - Summary of all metrics
2. **Predictions vs Actual** - Scatter plots for each model
3. **Residuals Distribution** - Histogram of prediction errors
4. **Metrics Comparison** - Bar charts comparing RMSE, R², size, and speed

All plots are saved in the `models/` directory.

## Model Files

After training, the following files are saved:

- `lightgbm_model.txt` - LightGBM model (~1-5 MB)
- `mlp_fp32.pt` - Full precision MLP (~0.5 MB)
- `mlp_pruned.pt` - Pruned MLP (~0.5 MB, 30% sparse)
- `mlp_int8.pt` - Quantized MLP (~0.1-0.2 MB, 4x smaller)
- `mlp_scaler.pkl` - StandardScaler for MLP inputs
- `evaluation_results.csv` - Detailed comparison table

## Expected Results

Typical performance (will vary based on your dataset):

| Model | RMSE | R² | Size | Speed |
|-------|------|-----|------|-------|
| LightGBM | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP FP32 | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP Pruned | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP INT8 | ~XXX | ~0.XX | ~X MB | ~0.00X ms |

**Compression Benefits**:
- **Size reduction**: ~75% (FP32 → INT8)
- **Speed improvement**: ~2-4x faster inference
- **Accuracy retention**: Minimal loss (<2% RMSE increase)

## Methodology

### Data Preprocessing
1. Load CSV data
2. Remove duplicates
3. Handle missing values
4. Keep relevant columns

### Feature Engineering
1. Create time-based features
2. Add seasonal indicators
3. Generate lag features per airport
4. Encode categorical variables

### Model Training
1. **LightGBM**: Direct training with early stopping
2. **MLP**: 
   - Standardize features
   - Train with MSE loss
   - Save best model

### Model Compression
1. **Pruning**:
   - Apply L-n structured pruning (30%)
   - Fine-tune for 10 epochs
   - Remove pruning masks
2. **Quantization**:
   - Dynamic quantization FP32 → INT8
   - Evaluate accuracy retention
   - Measure size/speed improvements

## Notes

- **Data Quality**: Ensure your CSV has no major data quality issues
- **Memory**: Large datasets may require chunking or sampling
- **GPU**: MLP training can use CUDA if available
- **Quantization**: INT8 models run best on CPU (fbgemm backend)

## Contributing

This project was created as a demonstration of:
- Machine learning regression pipeline
- Model compression techniques
- PyTorch neural network implementation
- LightGBM gradient boosting
- Comprehensive model evaluation

## License

This project is open source and available for educational purposes.

## Acknowledgments

- **Dataset**: European Aviation Safety Agency (EASA) / Eurocontrol
- **Libraries**: PyTorch, LightGBM, scikit-learn, pandas
- **Model Compression**: PyTorch quantization and pruning APIs

---

**Author**: Wojciech Domino & Mateusz Maj 
**Date**: November 2025  
**Purpose**: Air traffic volume prediction with model compression