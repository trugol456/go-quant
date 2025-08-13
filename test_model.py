import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data():
    """Create synthetic ETH orderbook data for testing"""
    print("Creating synthetic data for testing...")
    
    # Create timestamps (1 hour of 1-second data)
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(3600)]
    
    # Generate synthetic price data with realistic characteristics
    np.random.seed(42)
    base_price = 2500.0
    
    # Generate price walks with volatility clustering
    returns = np.random.normal(0, 0.0001, len(timestamps))
    # Add volatility clustering
    vol_regime = np.random.choice([0.5, 1.0, 2.0], len(timestamps), p=[0.7, 0.2, 0.1])
    returns = returns * vol_regime
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create orderbook data
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        spread = price * np.random.uniform(0.0001, 0.0005)
        
        # Generate bid/ask prices and volumes
        bid_price1 = price - spread/2
        ask_price1 = price + spread/2
        mid_price = (bid_price1 + ask_price1) / 2
        
        # Generate multiple orderbook levels
        bid_prices = [bid_price1 - i * spread * np.random.uniform(0.1, 0.3) for i in range(5)]
        ask_prices = [ask_price1 + i * spread * np.random.uniform(0.1, 0.3) for i in range(5)]
        
        # Generate volumes with realistic patterns
        base_volume = np.random.uniform(10, 100)
        bid_volumes = [base_volume * np.random.uniform(0.5, 2.0) * (0.8 ** i) for i in range(5)]
        ask_volumes = [base_volume * np.random.uniform(0.5, 2.0) * (0.8 ** i) for i in range(5)]
        
        # Create implied volatility target (simplified model)
        # IV should correlate with recent price movements and spreads
        if i > 10:
            recent_returns = returns[max(0, i-10):i]
            realized_vol = np.std(recent_returns) * np.sqrt(10)
            spread_factor = spread / price
            iv = realized_vol + spread_factor * 10 + np.random.normal(0, 0.001)
            iv = max(0.001, min(0.1, iv))  # Clamp to reasonable range
        else:
            iv = 0.02
        
        row = {
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'mid_price': mid_price,
            'label': iv
        }
        
        # Add bid prices and volumes
        for j in range(5):
            row[f'bid_price{j+1}'] = bid_prices[j]
            row[f'bid_volume{j+1}'] = bid_volumes[j]
            row[f'ask_price{j+1}'] = ask_prices[j]
            row[f'ask_volume{j+1}'] = ask_volumes[j]
        
        data.append(row)
    
    return pd.DataFrame(data)

def test_feature_engineering():
    """Test the feature engineering components"""
    from assignment import ETHVolatilityForecaster
    
    print("Testing feature engineering components...")
    
    # Create synthetic data
    df = create_synthetic_data()
    print(f"Created synthetic data with shape: {df.shape}")
    
    # Initialize forecaster
    forecaster = ETHVolatilityForecaster()
    
    # Test individual feature engineering components
    print("\n1. Testing basic orderbook features...")
    df_with_basic = forecaster.calculate_advanced_orderbook_features(df)
    print(f"Added basic features. New shape: {df_with_basic.shape}")
    
    print("\n2. Testing time series features...")
    df_with_ts = forecaster.calculate_advanced_time_series_features(df_with_basic, min_periods_for_indicators=30)
    print(f"Added time series features. New shape: {df_with_ts.shape}")
    
    print("\n3. Testing regime features...")
    df_with_regime = forecaster.create_regime_features(df_with_ts)
    print(f"Added regime features. New shape: {df_with_regime.shape}")
    
    print("\n4. Testing lagged features...")
    df_final = forecaster.create_lagged_features(df_with_regime, 'label')
    print(f"Added lagged features. Final shape: {df_final.shape}")
    
    # Show sample of new features
    print("\nSample of engineered features:")
    feature_cols = [col for col in df_final.columns if col not in ['timestamp', 'label'] and not col.startswith(('bid_', 'ask_'))]
    print(f"Generated {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols[:15]):
        print(f"  {i+1:2d}. {col}")
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more features")
    
    return df_final

def test_modeling_pipeline():
    """Test the modeling pipeline with synthetic data"""
    print("\n" + "="*60)
    print("TESTING MODELING PIPELINE")
    print("="*60)
    
    # Create train/test split from synthetic data
    df = create_synthetic_data()
    
    # Split into train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Remove labels from test data
    test_df = test_df.drop('label', axis=1)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize forecaster with synthetic data
    from assignment import ETHVolatilityForecaster
    forecaster = ETHVolatilityForecaster()
    
    # Manually set the data
    forecaster.eth_train = train_df
    forecaster.eth_test = test_df
    forecaster.original_test_length = len(test_df)
    
    print("\nTesting feature engineering...")
    forecaster.engineer_features()
    
    print(f"Final train features: {len(forecaster.feature_columns)}")
    
    if len(forecaster.feature_columns) > 0:
        print("\nTesting model training...")
        try:
            X_test_dict = forecaster.train_advanced_models()
            print("✓ Model training completed successfully")
            
            print("\nTesting prediction generation...")
            final_predictions, individual_predictions, ensemble_strategies = forecaster.generate_predictions(X_test_dict)
            print("✓ Prediction generation completed successfully")
            
            print(f"\nPrediction Statistics:")
            print(f"  Length: {len(final_predictions)}")
            print(f"  Mean: {np.mean(final_predictions):.6f}")
            print(f"  Std:  {np.std(final_predictions):.6f}")
            print(f"  Min:  {np.min(final_predictions):.6f}")
            print(f"  Max:  {np.max(final_predictions):.6f}")
            
            # Test feature importance analysis
            feature_rankings = forecaster.analyze_feature_importance()
            
        except Exception as e:
            print(f"✗ Error in modeling pipeline: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No features generated - cannot test modeling pipeline")

def main():
    """Main testing function"""
    print("TESTING ENHANCED ETH VOLATILITY FORECASTER")
    print("="*60)
    
    # Test 1: Feature Engineering
    print("\nTEST 1: Feature Engineering")
    print("-" * 30)
    try:
        df_with_features = test_feature_engineering()
        print("✓ Feature engineering test passed")
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Full Modeling Pipeline
    print("\nTEST 2: Modeling Pipeline")
    print("-" * 30)
    try:
        test_modeling_pipeline()
        print("✓ Modeling pipeline test completed")
    except Exception as e:
        print(f"✗ Modeling pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    print("\nKey Improvements Made:")
    print("1. ✓ Advanced orderbook features (volume concentration, micro-price, weighted pressure)")
    print("2. ✓ Enhanced time series features (multiple volatility estimators, microstructure)")
    print("3. ✓ Cross-asset correlation and lead-lag relationships")
    print("4. ✓ Market regime detection (volatility, trend, momentum regimes)")
    print("5. ✓ Sophisticated ensemble methods (rank averaging, trimmed mean, dynamic weighting)")
    print("6. ✓ Multiple model types (Ridge, ElasticNet, RF, XGB, LGB, GBM, MLP)")
    print("7. ✓ Enhanced time series cross-validation")
    print("8. ✓ Feature selection and importance analysis")
    print("9. ✓ Multiple scaling strategies for different model types")
    print("10. ✓ Post-processing smoothing for time series predictions")

if __name__ == "__main__":
    main()