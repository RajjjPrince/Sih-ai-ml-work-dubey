"""
Train O3 and NO2 models for all sites (2-7)
- O3 models: Using approach from train_final_no2_o3.py
- NO2 models: Using approach from train_robust_no2.py
- Save models in separate folders for each site
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import warnings
import pickle
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING O3 AND NO2 MODELS FOR ALL SITES (2-7)")
print("="*80)

# ==================== FEATURE ENGINEERING FUNCTIONS ====================

def create_o3_features(df):
    """Create O3-specific features"""
    # Time features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    df['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5, 6]:
            return 'summer'
        elif month in [7, 8, 9]:
            return 'monsoon'
        else:
            return 'post_monsoon'
    
    df['season'] = df['month'].apply(get_season)
    df['is_winter'] = (df['season'] == 'winter').astype(int)
    df['is_summer'] = (df['season'] == 'summer').astype(int)
    df['is_monsoon'] = (df['season'] == 'monsoon').astype(int)
    df['is_post_monsoon'] = (df['season'] == 'post_monsoon').astype(int)
    
    # Derived meteorological features
    if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
    if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
        df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
    if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
        df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)
    
    # O3 photochemical features
    if 'solar_elevation' in df.columns:
        df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
        df['solar_elevation_squared'] = df['solar_elevation']**2
        df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
        df['solar_elevation_positive'] = np.maximum(0, df['solar_elevation'])
    if 'SZA_deg' in df.columns:
        df['sza_rad'] = np.radians(df['SZA_deg'])
        df['cos_sza'] = np.cos(df['sza_rad'])
        df['photolysis_rate_approx'] = np.maximum(0, df['cos_sza'])
    if 't2m_era5' in df.columns and 'solar_elevation' in df.columns:
        df['temp_solar_interaction'] = df['t2m_era5'] * np.abs(df['solar_elevation'])
        df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
        df['temp_solar_elevation_squared'] = df['t2m_era5'] * df['solar_elevation_squared']
    if 't2m_era5' in df.columns and 'photolysis_rate_approx' in df.columns:
        df['temp_photolysis'] = df['t2m_era5'] * df['photolysis_rate_approx']
    if 't2m_era5' in df.columns and 'cos_sza' in df.columns:
        df['temp_cos_sza'] = df['t2m_era5'] * df['cos_sza']
    if 'blh_era5' in df.columns:
        if 'solar_elevation' in df.columns:
            df['pbl_solar_elevation'] = df['blh_era5'] * df['solar_elevation_abs']
            df['pbl_solar_elevation_squared'] = df['blh_era5'] * df['solar_elevation_squared']
        if 'photolysis_rate_approx' in df.columns:
            df['pbl_photolysis'] = df['blh_era5'] * df['photolysis_rate_approx']
        if 'cos_sza' in df.columns:
            df['pbl_cos_sza'] = df['blh_era5'] * df['cos_sza']
        if 't2m_era5' in df.columns:
            df['pbl_temp'] = df['blh_era5'] * df['t2m_era5']
    if 'wind_speed' in df.columns and 'blh_era5' in df.columns:
        df['pbl_wind_product'] = df['blh_era5'] * df['wind_speed']
    if 'relative_humidity_approx' in df.columns and 't2m_era5' in df.columns:
        df['rh_temp_interaction'] = df['relative_humidity_approx'] * df['t2m_era5']
    if 'is_weekend' in df.columns and 'solar_elevation_abs' in df.columns:
        df['weekend_solar'] = df['is_weekend'] * df['solar_elevation_abs']
    
    # O3 lags and rolling means
    for col in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
        if col in df.columns:
            for lag in [1, 3, 6]:
                if f'{col}_lag_{lag}h' not in df.columns:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    for window in [3, 6, 12]:
        for feat in ['O3_target', 'no2', 't2m_era5']:
            if feat in df.columns:
                df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()
    
    return df

def create_no2_features(df):
    """Create NO2-specific features"""
    # Time features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Derived meteorological features
    if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
    
    # Core lags
    core_lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
    for col in core_lag_features:
        if col in df.columns:
            for lag in [1, 3, 6, 24]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    # BLH lag
    if 'blh_era5' in df.columns:
        df['blh_lag_1h'] = df['blh_era5'].shift(1)
    
    # Physically-motivated features
    if 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
        df['inversion_strength'] = df['t2m_era5'] - df['d2m_era5']
    
    if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
        df['ventilation_index'] = df['blh_era5'] * df['wind_speed']
        if 'inversion_strength' in df.columns:
            df['stability_index'] = df['inversion_strength'] / (df['blh_era5'] + 1e-6)
    
    # Traffic proxies
    df['morning_peak'] = df['hour'].isin([7, 8, 9]).astype(int)
    df['evening_peak'] = df['hour'].isin([17, 18, 19]).astype(int)
    df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
    
    # Event indicators
    df['stubble_burning_flag'] = df['month'].isin([10, 11]).astype(int)
    df['diwali_flag'] = ((df['month'] == 10) & (df['datetime'].dt.day >= 20) & (df['datetime'].dt.day <= 24)) | \
                        ((df['month'] == 11) & (df['datetime'].dt.day >= 1) & (df['datetime'].dt.day <= 5))
    df['diwali_flag'] = df['diwali_flag'].astype(int)
    
    # Stagnation flags
    if 'wind_speed' in df.columns:
        df['low_wind_flag'] = (df['wind_speed'] < 1.0).astype(int)
    if 'blh_era5' in df.columns:
        df['low_blh_flag'] = (df['blh_era5'] < 100).astype(int)
    
    # Simple interactions
    if 'pm2p5' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
    
    # Wind components
    if 'u10_era5' in df.columns:
        df['wind_u'] = df['u10_era5']
    if 'v10_era5' in df.columns:
        df['wind_v'] = df['v10_era5']
    
    # Solar elevation (if available)
    if 'solar_elevation' in df.columns:
        df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    
    # NO2 lifetime proxy
    if 'O3_target' in df.columns and 't2m_era5' in df.columns:
        df['no2_lifetime_proxy'] = 1.0 / (df['O3_target'] + df['t2m_era5'] / 10.0 + 1e-6)
    elif 't2m_era5' in df.columns:
        df['no2_lifetime_proxy'] = 1.0 / (df['t2m_era5'] / 10.0 + 1e-6)
    
    return df

def get_o3_features(df):
    """Get O3 feature set"""
    features = []
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    solar_features = ['solar_elevation', 'solar_elevation_abs', 'solar_elevation_squared',
                     'solar_elevation_positive', 'is_daytime', 'SZA_deg', 'sza_rad',
                     'cos_sza', 'photolysis_rate_approx']
    features.extend([f for f in solar_features if f in df.columns])
    photo_interactions = ['temp_solar_elevation', 'temp_solar_elevation_squared',
                         'temp_photolysis', 'temp_cos_sza']
    features.extend([f for f in photo_interactions if f in df.columns])
    pbl_solar_features = ['pbl_solar_elevation', 'pbl_solar_elevation_squared',
                         'pbl_photolysis', 'pbl_cos_sza', 'pbl_temp']
    features.extend([f for f in pbl_solar_features if f in df.columns])
    other_interactions = ['ventilation_rate', 'pbl_wind_product', 'rh_temp_interaction',
                         'weekend_solar']
    features.extend([f for f in other_interactions if f in df.columns])
    for lag in [1, 3, 6]:
        for feat in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    for window in [3, 6, 12]:
        for feat in ['O3_target', 'no2', 't2m_era5']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    # Remove duplicates
    seen = set()
    unique_features = []
    for f in features:
        if f in df.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features

def get_no2_features(df):
    """Get NO2 feature set (simplified core set)"""
    features = []
    
    # Core pollutants
    core_pollutants = ['pm2p5', 'pm10', 'so2', 'no2']
    features.extend([f for f in core_pollutants if f in df.columns])
    
    # Core meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5']
    features.extend([f for f in meteo if f in df.columns])
    
    # Core lags
    for lag in [1, 3, 6, 24]:
        for feat in ['no2', 'pm2p5', 'pm10', 't2m_era5', 'wind_speed']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    
    # BLH lag
    if 'blh_lag_1h' in df.columns:
        features.append('blh_lag_1h')
    
    # Physically-motivated features
    phys_features = ['inversion_strength', 'ventilation_index', 'stability_index', 'hour_weekend_interaction']
    features.extend([f for f in phys_features if f in df.columns])
    
    # Time features
    time_features = ['hour', 'month', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Traffic proxies
    traffic_features = ['morning_peak', 'evening_peak']
    features.extend([f for f in traffic_features if f in df.columns])
    
    # Event indicators
    event_features = ['stubble_burning_flag', 'diwali_flag', 'low_wind_flag', 'low_blh_flag']
    features.extend([f for f in event_features if f in df.columns])
    
    # Simple interactions
    if 'pm25_pm10_ratio' in df.columns:
        features.append('pm25_pm10_ratio')
    
    # Wind components
    if 'wind_u' in df.columns:
        features.append('wind_u')
    if 'wind_v' in df.columns:
        features.append('wind_v')
    
    # Solar
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    if 'solar_elevation_abs' in df.columns:
        features.append('solar_elevation_abs')
    
    # NO2 lifetime proxy
    if 'no2_lifetime_proxy' in df.columns:
        features.append('no2_lifetime_proxy')
    
    # Remove duplicates
    seen = set()
    unique_features = []
    for f in features:
        if f in df.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features

# ==================== DATA PREPARATION ====================

def prepare_data(df, target_col, features, train_mask, val_mask, test_mask=None):
    """Prepare data"""
    valid_mask = ~df[target_col].isna()
    train_idx = valid_mask & train_mask
    val_idx = valid_mask & val_mask
    
    X_train = df[train_idx][features].copy()
    y_train = df[train_idx][target_col].copy()
    X_val = df[val_idx][features].copy()
    y_val = df[val_idx][target_col].copy()
    
    if test_mask is not None:
        test_idx = valid_mask & test_mask
        X_test = df[test_idx][features].copy()
        y_test = df[test_idx][target_col].copy()
    else:
        X_test = None
        y_test = None
    
    # Convert to numeric
    for col in features:
        if col in X_train.columns:
            col_series = X_train[col]
            if isinstance(col_series, pd.Series):
                col_dtype = col_series.dtype
            else:
                continue
            
            if col_dtype == 'object':
                X_train[col] = pd.Categorical(X_train[col]).codes
                if col in X_val.columns:
                    X_val[col] = pd.Categorical(X_val[col], categories=pd.Categorical(X_train[col]).categories).codes
                if test_mask is not None and col in X_test.columns:
                    X_test[col] = pd.Categorical(X_test[col], categories=pd.Categorical(X_train[col]).categories).codes
            elif col_dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype(int)
                if test_mask is not None and col in X_test.columns:
                    X_test[col] = X_test[col].astype(int)
    
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    if test_mask is not None:
        X_test = X_test.select_dtypes(include=[np.number])
    features = [f for f in features if f in X_train.columns]
    
    # Fill NaN
    for col in X_train.columns:
        col_series = X_train[col]
        if isinstance(col_series, pd.Series):
            null_count = int(col_series.isnull().sum())
        else:
            null_count = 0
        
        if null_count > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            if col in X_val.columns:
                X_val[col].fillna(median_val, inplace=True)
            if test_mask is not None and col in X_test.columns:
                X_test[col].fillna(median_val, inplace=True)
    
    if test_mask is not None:
        return X_train, y_train, X_val, y_val, X_test, y_test, features
    else:
        return X_train, y_train, X_val, y_val, features

# ==================== TRAIN O3 MODEL ====================

def train_o3_model(df, train_mask, val_mask, test_mask):
    """Train O3 model"""
    print(f"\n{'='*80}")
    print("TRAINING O3 MODEL")
    print(f"{'='*80}")
    
    features = get_o3_features(df)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, 'O3_target', features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred)
    }
    val_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred),
        'R2': r2_score(y_val, y_val_pred)
    }
    test_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== TRAIN NO2 MODEL ====================

def calculate_sample_weights(y, percentile=75, weight_factor=2.0):
    """Calculate sample weights"""
    threshold = np.percentile(y, percentile)
    weights = np.ones(len(y))
    weights[y >= threshold] = weight_factor
    return weights

def train_residual_calibrator(y_true, y_pred, method='isotonic'):
    """Train residual calibrator"""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred, y_true)
    else:
        calibrator = LinearRegression()
        calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator

def train_no2_model(df, train_mask, val_mask, test_mask):
    """Train NO2 model"""
    print(f"\n{'='*80}")
    print("TRAINING NO2 MODEL")
    print(f"{'='*80}")
    
    features = get_no2_features(df)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, 'NO2_target', features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Peak-weighted training
    sample_weights = calculate_sample_weights(y_train, percentile=75, weight_factor=2.0)
    print(f"   Using peak-weighted loss (2.0x weight for >75th percentile)")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 6,
        'max_depth': 3,
        'learning_rate': 0.008,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 150,
        'lambda_l1': 2.5,
        'lambda_l2': 3.0,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=600,
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Train residual calibrator on validation set
    print("   Training residual calibrator on validation set...")
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    calibrator = train_residual_calibrator(y_val, val_pred, method='isotonic')
    
    # Calculate metrics
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
        'MAE': mean_absolute_error(y_train, train_pred),
        'R2': r2_score(y_train, train_pred)
    }
    val_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_val, val_pred)),
        'MAE': mean_absolute_error(y_val, val_pred),
        'R2': r2_score(y_val, val_pred)
    }
    
    # Apply calibration to test set
    test_pred_calibrated = calibrator.predict(test_pred)
    
    test_metrics_uncalibrated = {
        'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
        'MAE': mean_absolute_error(y_test, test_pred),
        'R2': r2_score(y_test, test_pred)
    }
    
    test_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, test_pred_calibrated)),
        'MAE': mean_absolute_error(y_test, test_pred_calibrated),
        'R2': r2_score(y_test, test_pred_calibrated)
    }
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE (uncalibrated): {test_metrics_uncalibrated['RMSE']:.4f}, R²: {test_metrics_uncalibrated['R2']:.4f}")
    print(f"   Test RMSE (calibrated):   {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    print(f"   Calibration improvement: {((test_metrics_uncalibrated['RMSE'] - test_metrics['RMSE']) / test_metrics_uncalibrated['RMSE'] * 100):.2f}%")
    
    return model, calibrator, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== MAIN TRAINING LOOP ====================

sites = [2, 3, 4, 5, 6, 7]
all_results = []

for site_num in sites:
    print("\n" + "="*80)
    print(f"PROCESSING SITE {site_num}")
    print("="*80)
    
    # Load site data
    site_file = f'sites/site_{site_num}_final_cleaned.csv'
    if not os.path.exists(site_file):
        print(f"   ⚠️  File not found: {site_file}, skipping...")
        continue
    
    print(f"\n1. Loading data from {site_file}...")
    df = pd.read_csv(site_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Check for required targets
    has_o3 = 'O3_target' in df.columns and df['O3_target'].notna().sum() > 100
    has_no2 = 'NO2_target' in df.columns and df['NO2_target'].notna().sum() > 100
    
    if not has_o3 and not has_no2:
        print(f"   ⚠️  No valid targets found, skipping site {site_num}...")
        continue
    
    # Create output directory
    site_folder = f'site_{site_num}_models'
    os.makedirs(site_folder, exist_ok=True)
    os.makedirs(f'{site_folder}/results', exist_ok=True)
    
    # Create features
    print("\n2. Creating features...")
    if has_o3:
        df = create_o3_features(df)
    if has_no2:
        df = create_no2_features(df)
    
    # Define train/val/test splits (similar to site 1)
    # Use temporal splits based on available data
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    date_range = max_date - min_date
    
    # If we have enough data, use similar splits as site 1
    if date_range.days > 730:  # At least 2 years
        # Train: first 2 years
        train_end = min_date + pd.DateOffset(days=int(date_range.days * 0.6))
        val_start = train_end
        val_end = min_date + pd.DateOffset(days=int(date_range.days * 0.8))
        test_start = val_end
        
        train_mask = (df['datetime'] >= min_date) & (df['datetime'] < train_end)
        val_mask = (df['datetime'] >= val_start) & (df['datetime'] < val_end)
        test_mask = (df['datetime'] >= test_start) & (df['datetime'] <= max_date)
    else:
        # For shorter datasets, use 60/20/20 split
        train_end = min_date + pd.DateOffset(days=int(date_range.days * 0.6))
        val_end = min_date + pd.DateOffset(days=int(date_range.days * 0.8))
        
        train_mask = (df['datetime'] >= min_date) & (df['datetime'] < train_end)
        val_mask = (df['datetime'] >= train_end) & (df['datetime'] < val_end)
        test_mask = (df['datetime'] >= val_end) & (df['datetime'] <= max_date)
    
    print(f"\n3. Data splits:")
    print(f"   Train: {train_mask.sum()} samples ({df[train_mask]['datetime'].min()} to {df[train_mask]['datetime'].max()})")
    print(f"   Val:   {val_mask.sum()} samples ({df[val_mask]['datetime'].min()} to {df[val_mask]['datetime'].max()})")
    print(f"   Test:  {test_mask.sum()} samples ({df[test_mask]['datetime'].min()} to {df[test_mask]['datetime'].max()})")
    
    site_results = {'site': site_num}
    
    # Train O3 model
    if has_o3:
        try:
            o3_model, o3_train_metrics, o3_val_metrics, o3_test_metrics, o3_baseline = train_o3_model(
                df, train_mask, val_mask, test_mask
            )
            
            # Save O3 model
            o3_model.save_model(f'{site_folder}/o3_model.txt')
            with open(f'{site_folder}/o3_model.pkl', 'wb') as f:
                pickle.dump(o3_model, f)
            
            site_results['O3'] = {
                'train_rmse': o3_train_metrics['RMSE'],
                'train_r2': o3_train_metrics['R2'],
                'val_rmse': o3_val_metrics['RMSE'],
                'val_r2': o3_val_metrics['R2'],
                'test_rmse': o3_test_metrics['RMSE'],
                'test_r2': o3_test_metrics['R2'],
                'baseline_rmse': o3_baseline
            }
            
            print(f"\n   ✓ O3 model saved to {site_folder}/")
        except Exception as e:
            print(f"   ✗ Error training O3 model: {e}")
            site_results['O3'] = {'error': str(e)}
    
    # Train NO2 model
    if has_no2:
        try:
            no2_model, no2_calibrator, no2_train_metrics, no2_val_metrics, no2_test_metrics, no2_baseline = train_no2_model(
                df, train_mask, val_mask, test_mask
            )
            
            # Save NO2 model
            no2_model.save_model(f'{site_folder}/no2_model.txt')
            with open(f'{site_folder}/no2_model.pkl', 'wb') as f:
                pickle.dump((no2_model, no2_calibrator), f)
            
            site_results['NO2'] = {
                'train_rmse': no2_train_metrics['RMSE'],
                'train_r2': no2_train_metrics['R2'],
                'val_rmse': no2_val_metrics['RMSE'],
                'val_r2': no2_val_metrics['R2'],
                'test_rmse': no2_test_metrics['RMSE'],
                'test_r2': no2_test_metrics['R2'],
                'baseline_rmse': no2_baseline
            }
            
            print(f"\n   ✓ NO2 model saved to {site_folder}/")
        except Exception as e:
            print(f"   ✗ Error training NO2 model: {e}")
            site_results['NO2'] = {'error': str(e)}
    
    all_results.append(site_results)
    
    # Save site-specific results
    results_df = pd.DataFrame([site_results])
    results_df.to_csv(f'{site_folder}/results/performance_summary.csv', index=False)
    
    print(f"\n   ✓ Results saved to {site_folder}/results/")

# ==================== SAVE OVERALL SUMMARY ====================

print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)

summary_data = []
for result in all_results:
    row = {'Site': result['site']}
    if 'O3' in result and 'error' not in result['O3']:
        row['O3_Test_RMSE'] = result['O3']['test_rmse']
        row['O3_Test_R2'] = result['O3']['test_r2']
    else:
        row['O3_Test_RMSE'] = 'N/A'
        row['O3_Test_R2'] = 'N/A'
    
    if 'NO2' in result and 'error' not in result['NO2']:
        row['NO2_Test_RMSE'] = result['NO2']['test_rmse']
        row['NO2_Test_R2'] = result['NO2']['test_r2']
    else:
        row['NO2_Test_RMSE'] = 'N/A'
        row['NO2_Test_R2'] = 'N/A'
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('all_sites_performance_summary.csv', index=False)

print("\nOverall Performance Summary:")
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: all_sites_performance_summary.csv")
print("\n" + "="*80)
print("All models saved in site-specific folders:")
for site_num in sites:
    print(f"  - site_{site_num}_models/")
print("="*80)

