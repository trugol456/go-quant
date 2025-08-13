import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

# Optimized imports for Kaggle environment
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
import xgboost as xgb
import lightgbm as lgb

# Statistical imports
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import joblib

class KaggleOptimizedETHForecaster:
    """
    Kaggle P100-Optimized ETH Implied Volatility Forecasting Model
    
    Optimizations:
    - Reduced memory footprint
    - GPU-accelerated where possible
    - Efficient feature selection
    - Streamlined ensemble
    - Fast execution for Kaggle time limits
    """
    
    def __init__(self, data_path_train="/kaggle/input/gq-implied-volatility-forecasting/train/",
                 data_path_test="/kaggle/input/gq-implied-volatility-forecasting/test/",
                 max_features=80, n_models=4):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.max_features = max_features  # Reduced for memory efficiency
        self.n_models = n_models
        self.eth_train = None
        self.eth_test = None
        self.cross_assets = {}
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Memory optimization settings
        self.dtype = np.float32  # Use float32 instead of float64
        self.chunk_size = 10000  # Process data in chunks if needed
        
    def optimize_memory(self, df):
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype(np.int32)
        return df
    
    def load_data(self):
        """Load data with memory optimization"""
        print("Loading ETH training data...")
        self.eth_train = pd.read_csv(f"{self.data_path_train}ETH.csv")
        self.eth_train['timestamp'] = pd.to_datetime(self.eth_train['timestamp'])
        self.eth_train = self.eth_train.sort_values('timestamp').reset_index(drop=True)
        self.eth_train = self.optimize_memory(self.eth_train)
        
        print("Loading ETH test data...")
        self.eth_test = pd.read_csv(f"{self.data_path_test}ETH.csv")
        self.eth_test['timestamp'] = pd.to_datetime(self.eth_test['timestamp'])
        self.eth_test = self.eth_test.sort_values('timestamp').reset_index(drop=True)
        self.eth_test = self.optimize_memory(self.eth_test)
        
        self.original_test_length = len(self.eth_test)
        
        # Load only essential cross-asset data (BTC for correlation)
        try:
            print("Loading BTC training data...")
            btc_train = pd.read_csv(f"{self.data_path_train}BTC.csv")
            btc_train['timestamp'] = pd.to_datetime(btc_train['timestamp'])
            btc_train = btc_train.sort_values('timestamp').reset_index(drop=True)
            self.cross_assets['BTC_train'] = self.optimize_memory(btc_train)
            
            # Try BTC test data
            try:
                btc_test = pd.read_csv(f"{self.data_path_test}BTC.csv")
                btc_test['timestamp'] = pd.to_datetime(btc_test['timestamp'])
                btc_test = btc_test.sort_values('timestamp').reset_index(drop=True)
                self.cross_assets['BTC_test'] = self.optimize_memory(btc_test)
            except FileNotFoundError:
                print("BTC test data not found")
        except FileNotFoundError:
            print("BTC training data not found")
        
        print(f"Data loaded - ETH Train: {self.eth_train.shape}, ETH Test: {self.eth_test.shape}")
        
    def calculate_essential_features(self, df, prefix=""):
        """Calculate only the most essential features for speed"""
        df_features = df.copy()
        
        # Basic orderbook features
        df_features[f'{prefix}spread'] = df_features['ask_price1'] - df_features['bid_price1']
        df_features[f'{prefix}spread_pct'] = df_features[f'{prefix}spread'] / (df_features['mid_price'] + 1e-8)
        
        # Volume imbalance (only level 1 for speed)
        df_features[f'{prefix}volume_imbalance'] = (
            (df_features['bid_volume1'] - df_features['ask_volume1']) / 
            (df_features['bid_volume1'] + df_features['ask_volume1'] + 1e-8)
        )
        
        # Total volumes
        bid_cols = [f'bid_volume{i}' for i in range(1, 6) if f'bid_volume{i}' in df_features.columns]
        ask_cols = [f'ask_volume{i}' for i in range(1, 6) if f'ask_volume{i}' in df_features.columns]
        
        if bid_cols and ask_cols:
            df_features[f'{prefix}total_bid_volume'] = df_features[bid_cols].sum(axis=1)
            df_features[f'{prefix}total_ask_volume'] = df_features[ask_cols].sum(axis=1)
            df_features[f'{prefix}total_volume'] = df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume']
            
            # Weighted orderbook pressure (simplified)
            df_features[f'{prefix}orderbook_pressure'] = (
                (df_features[f'{prefix}total_bid_volume'] - df_features[f'{prefix}total_ask_volume']) / 
                (df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume'] + 1e-8)
            )
        
        # Price impact (simplified)
        if 'ask_price5' in df_features.columns and 'bid_price5' in df_features.columns:
            df_features[f'{prefix}price_impact'] = (
                (df_features['ask_price5'] - df_features['bid_price5']) / df_features['mid_price']
            )
        
        return df_features
    
    def calculate_time_series_features(self, df, prefix=""):
        """Calculate essential time series features"""
        df_features = df.copy()
        
        # Essential windows only
        windows = [5, 10, 30, 60, 300]  # Reduced from more extensive list
        
        for window in windows:
            if len(df_features) > window:
                # Returns
                df_features[f'{prefix}return_{window}s'] = df_features['mid_price'].pct_change(window)
                
                # Realized volatility
                returns = df_features['mid_price'].pct_change()
                df_features[f'{prefix}realized_vol_{window}s'] = returns.rolling(window, min_periods=1).std()
                
                # Price momentum
                df_features[f'{prefix}momentum_{window}s'] = (
                    df_features['mid_price'] / df_features['mid_price'].shift(window) - 1
                )
                
                # Volume momentum (if available)
                if f'{prefix}total_volume' in df_features.columns:
                    df_features[f'{prefix}volume_momentum_{window}s'] = df_features[f'{prefix}total_volume'].pct_change(window)
        
        # Simple technical indicators
        if len(df_features) >= 60:
            # Moving averages
            df_features[f'{prefix}sma_20'] = df_features['mid_price'].rolling(20, min_periods=1).mean()
            df_features[f'{prefix}ema_20'] = df_features['mid_price'].ewm(span=20, min_periods=1).mean()
            
            # RSI (simplified)
            delta = df_features['mid_price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df_features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands position
            sma = df_features['mid_price'].rolling(20, min_periods=1).mean()
            std = df_features['mid_price'].rolling(20, min_periods=1).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            df_features[f'{prefix}bb_position'] = (df_features['mid_price'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        return df_features
    
    def calculate_cross_asset_features(self, eth_df, is_test=False):
        """Calculate essential cross-asset features (BTC only for speed)"""
        df_features = eth_df.copy()
        
        suffix = "_test" if is_test else "_train"
        btc_key = f"BTC{suffix}"
        
        if btc_key in self.cross_assets:
            btc_df = self.cross_assets[btc_key]
            
            # Merge BTC data
            merged = pd.merge_asof(
                df_features.sort_values('timestamp'), 
                btc_df[['timestamp', 'mid_price']].sort_values('timestamp'),
                on='timestamp', 
                suffixes=('', '_BTC'),
                direction='backward',
                tolerance=pd.Timedelta('10s')
            )
            
            if 'mid_price_BTC' in merged.columns:
                # Essential cross-asset features only
                eth_returns = merged['mid_price'].pct_change()
                btc_returns = merged['mid_price_BTC'].pct_change()
                
                # Rolling correlation (60s window)
                df_features['BTC_correlation_60s'] = eth_returns.rolling(60, min_periods=10).corr(btc_returns)
                
                # BTC returns
                df_features['BTC_return_30s'] = merged['mid_price_BTC'].pct_change(30)
                
                # Relative performance
                df_features['BTC_relative_performance'] = eth_returns - btc_returns
        
        return df_features
    
    def create_lagged_features(self, df, target_col=None):
        """Create essential lagged features"""
        df_features = df.copy()
        
        # Core features for lagging (reduced set)
        core_features = ['mid_price', 'spread_pct', 'volume_imbalance', 'realized_vol_60s']
        
        # Add target lags only for training
        if target_col and target_col in df.columns:
            core_features.append(target_col)
        
        # Reduced lag periods for speed
        lags = [1, 2, 5, 10, 30]
        
        for feature in core_features:
            if feature in df_features.columns:
                for lag in lags:
                    df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
        
        return df_features
    
    def fast_feature_selection(self, X_train, y_train):
        """Fast feature selection optimized for Kaggle"""
        print(f"Fast feature selection from {X_train.shape[1]} features...")
        
        # Remove low variance features
        var_selector = VarianceThreshold(threshold=0.001)
        X_var = var_selector.fit_transform(X_train)
        selected_features = X_train.columns[var_selector.get_support()]
        
        # Quick statistical selection
        if len(selected_features) > self.max_features:
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
            selector.fit(X_train[selected_features], y_train)
            final_features = selected_features[selector.get_support()]
        else:
            final_features = selected_features
        
        print(f"Selected {len(final_features)} features for modeling")
        return final_features.tolist()
    
    def engineer_features(self):
        """Streamlined feature engineering for Kaggle"""
        print("Starting streamlined feature engineering...")
        
        # Calculate essential features
        self.eth_train = self.calculate_essential_features(self.eth_train)
        self.eth_test = self.calculate_essential_features(self.eth_test)
        
        # Time series features
        self.eth_train = self.calculate_time_series_features(self.eth_train)
        self.eth_test = self.calculate_time_series_features(self.eth_test)
        
        # Cross-asset features
        self.eth_train = self.calculate_cross_asset_features(self.eth_train, is_test=False)
        self.eth_test = self.calculate_cross_asset_features(self.eth_test, is_test=True)
        
        # Lagged features
        self.eth_train = self.create_lagged_features(self.eth_train, 'label')
        self.eth_test = self.create_lagged_features(self.eth_test)
        
        # Memory cleanup
        gc.collect()
        
        # Find common features
        train_features = set(self.eth_train.columns) - {'timestamp', 'label'}
        test_features = set(self.eth_test.columns) - {'timestamp'}
        common_features = train_features.intersection(test_features)
        
        # Clean data
        self.eth_train = self.eth_train.dropna()
        
        # Fill test data NaN values
        for col in common_features:
            if col in self.eth_test.columns:
                self.eth_test[col] = self.eth_test[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Feature selection
        if len(common_features) > 0:
            X_temp = self.eth_train[list(common_features)].fillna(0)
            y_temp = self.eth_train['label']
            self.feature_columns = self.fast_feature_selection(X_temp, y_temp)
        else:
            self.feature_columns = []
        
        print(f"Feature engineering completed. Using {len(self.feature_columns)} features")
    
    def train_kaggle_optimized_models(self):
        """Train GPU-optimized models for Kaggle P100"""
        print("Training Kaggle-optimized models...")
        
        X_train = self.eth_train[self.feature_columns].copy()
        y_train = self.eth_train['label'].copy()
        X_test = self.eth_test[self.feature_columns].copy()
        
        # Handle remaining NaN and inf values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Optimize data types
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        # Scalers
        standard_scaler = StandardScaler()
        robust_scaler = RobustScaler()
        
        X_train_standard = standard_scaler.fit_transform(X_train)
        X_train_robust = robust_scaler.fit_transform(X_train)
        
        X_test_standard = standard_scaler.transform(X_test)
        X_test_robust = robust_scaler.transform(X_test)
        
        self.scalers['standard'] = standard_scaler
        self.scalers['robust'] = robust_scaler
        
        # Model 1: Ridge (fast and stable)
        print("Training Ridge...")
        ridge = Ridge(alpha=0.5, random_state=42)
        ridge.fit(X_train_standard, y_train)
        self.models['ridge'] = ridge
        
        # Model 2: ElasticNet
        print("Training ElasticNet...")
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        elastic.fit(X_train_robust, y_train)
        self.models['elastic'] = elastic
        
        # Model 3: XGBoost (GPU accelerated if available)
        print("Training XGBoost...")
        try:
            # Try GPU acceleration
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0,
                random_state=42
            )
        except:
            # Fallback to CPU
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Model 4: LightGBM (GPU accelerated if available)
        print("Training LightGBM...")
        try:
            # Try GPU acceleration
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                device='gpu',
                random_state=42,
                verbose=-1
            )
        except:
            # Fallback to CPU
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # Memory cleanup
        del X_train_standard, X_train_robust
        gc.collect()
        
        print("Model training completed!")
        return X_test_standard, X_test_robust, X_test
    
    def generate_fast_predictions(self, X_test_standard, X_test_robust, X_test):
        """Generate optimized ensemble predictions"""
        print("Generating ensemble predictions...")
        
        predictions = {}
        predictions['ridge'] = self.models['ridge'].predict(X_test_standard)
        predictions['elastic'] = self.models['elastic'].predict(X_test_robust)
        predictions['xgboost'] = self.models['xgboost'].predict(X_test)
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_test)
        
        # Simple but effective ensemble
        weights = {
            'ridge': 0.15,
            'elastic': 0.15,
            'xgboost': 0.35,
            'lightgbm': 0.35
        }
        
        ensemble_pred = np.zeros(len(X_test), dtype=np.float32)
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        # Light smoothing for time series
        if len(ensemble_pred) > 3:
            smoothed = np.convolve(ensemble_pred, np.ones(3)/3, mode='same')
            ensemble_pred = 0.8 * ensemble_pred + 0.2 * smoothed
        
        return ensemble_pred, predictions
    
    def create_submission(self, predictions):
        """Create submission file"""
        submission = pd.DataFrame({
            'timestamp': range(1, len(predictions) + 1),
            'labels': predictions
        })
        
        submission.to_csv("submission.csv", index=False)
        print(f"Submission saved: {submission.shape}")
        return submission
    
    def run_kaggle_pipeline(self):
        """Run the complete Kaggle-optimized pipeline"""
        print("Starting Kaggle-Optimized ETH Volatility Forecasting...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        if len(self.feature_columns) == 0:
            print("ERROR: No features generated!")
            return None, None
        
        # Train models
        X_test_standard, X_test_robust, X_test = self.train_kaggle_optimized_models()
        
        # Generate predictions
        final_predictions, individual_predictions = self.generate_fast_predictions(
            X_test_standard, X_test_robust, X_test
        )
        
        # Ensure correct length
        if len(final_predictions) != self.original_test_length:
            print(f"Adjusting prediction length from {len(final_predictions)} to {self.original_test_length}")
            if len(final_predictions) < self.original_test_length:
                # Pad with mean
                pad_length = self.original_test_length - len(final_predictions)
                pad_values = np.full(pad_length, np.mean(final_predictions))
                final_predictions = np.concatenate([final_predictions, pad_values])
            else:
                # Truncate
                final_predictions = final_predictions[:self.original_test_length]
        
        # Create submission
        submission = self.create_submission(final_predictions)
        
        print("=" * 60)
        print("Kaggle Pipeline Completed!")
        print(f"Prediction stats: Mean={np.mean(final_predictions):.6f}, Std={np.std(final_predictions):.6f}")
        
        return submission, final_predictions

# Kaggle execution
if __name__ == "__main__":
    # Initialize Kaggle-optimized forecaster
    forecaster = KaggleOptimizedETHForecaster(max_features=80, n_models=4)
    
    # Run optimized pipeline
    submission, predictions = forecaster.run_kaggle_pipeline()
    
    if submission is not None:
        print("\nSubmission preview:")
        print(submission.head())
        print("...")
        print(submission.tail())
        print("\n✅ Ready for Kaggle submission!")
    else:
        print("❌ Pipeline failed")