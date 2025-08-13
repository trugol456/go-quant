import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr

class ETHVolatilityForecaster:
    """
    
    ETH Implied Volatility Forecasting Model
    
    This class implements a comprehensive approach to forecasting 10-second ahead
    implied volatility for Ethereum using high-frequency orderbook data and 
    cross-asset signals.
    """
    
    def __init__(self, data_path_train="/kaggle/input/gq2data/train/",
                 data_path_test="/kaggle/input/gq2data/test/"):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.eth_train = None
        self.eth_test = None
        self.cross_assets = {}
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_data(self):
        """Load all training and test data"""
        print("Loading ETH training data...")
        self.eth_train = pd.read_csv(f"{self.data_path_train}ETH.csv")
        self.eth_train['timestamp'] = pd.to_datetime(self.eth_train['timestamp'])
        self.eth_train = self.eth_train.sort_values('timestamp').reset_index(drop=True)
        
        print("Loading ETH test data...")
        self.eth_test = pd.read_csv(f"{self.data_path_test}ETH.csv")
        self.eth_test['timestamp'] = pd.to_datetime(self.eth_test['timestamp'])
        self.eth_test = self.eth_test.sort_values('timestamp').reset_index(drop=True)
        
        # Store original test length - this is what submission should match
        self.original_test_length = len(self.eth_test)
        print(f"IMPORTANT: Original test data length = {self.original_test_length} (this is what submission should have)")
        
        # Store original test indices for submission
        self.original_test_indices = self.eth_test.index.copy()
        
        # Load cross-asset data for training
        cross_asset_files = ['BTC.csv', 'SOL.csv']
        for asset_file in cross_asset_files:
            try:
                asset_name = asset_file.replace('.csv', '')
                print(f"Loading {asset_name} training data...")
                self.cross_assets[f"{asset_name}_train"] = pd.read_csv(f"{self.data_path_train}{asset_file}")
                self.cross_assets[f"{asset_name}_train"]['timestamp'] = pd.to_datetime(
                    self.cross_assets[f"{asset_name}_train"]['timestamp'])
                self.cross_assets[f"{asset_name}_train"] = self.cross_assets[f"{asset_name}_train"].sort_values('timestamp').reset_index(drop=True)
                
                # Try to load test data for cross-assets
                try:
                    print(f"Loading {asset_name} test data...")
                    self.cross_assets[f"{asset_name}_test"] = pd.read_csv(f"{self.data_path_test}{asset_file}")
                    self.cross_assets[f"{asset_name}_test"]['timestamp'] = pd.to_datetime(
                        self.cross_assets[f"{asset_name}_test"]['timestamp'])
                    self.cross_assets[f"{asset_name}_test"] = self.cross_assets[f"{asset_name}_test"].sort_values('timestamp').reset_index(drop=True)
                except FileNotFoundError:
                    print(f"Warning: {asset_file} test data not found")
                    
            except FileNotFoundError:
                print(f"Warning: {asset_file} training data not found")
                
        print(f"Loaded data shapes - ETH Train: {self.eth_train.shape}, ETH Test: {self.eth_test.shape}")
        print(f"Expected submission rows: {self.original_test_length}")
        
    def clean_data(self, df):
        """Clean data by handling missing values and outliers"""
        df_clean = df.copy()
        is_test_data = 'label' not in df_clean.columns
        
        print(f"Input data shape: {df_clean.shape}")
        
        # Handle missing timestamps more carefully - don't resample test data
        if 'label' in df_clean.columns:  # Only for training data
            df_clean = df_clean.set_index('timestamp').resample('1S').first().reset_index()
            print(f"After resampling (train only): {df_clean.shape}")
        
        # Forward fill missing values for small gaps (up to 10 seconds)
        for col in df_clean.columns:
            if col != 'timestamp':
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=10)
        
        # For test data, be much more conservative about dropping rows
        if is_test_data:
            # For test data, only drop rows where ALL features are missing
            # Fill missing values instead of dropping rows
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill').fillna(df_clean[col].median())
            print(f"Test data after filling NaN: {df_clean.shape}")
        else:
            # For training data, we can be more aggressive about dropping bad rows
            missing_threshold = 0.5
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(thresh=len(df_clean.columns) * missing_threshold)
            print(f"Training data: dropped {initial_rows - len(df_clean)} rows with excessive missing data")
        
        # Handle outliers using IQR method (more conservative)
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'label':
                Q1 = df_clean[col].quantile(0.05)  # More conservative
                Q3 = df_clean[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        print(f"Final cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def calculate_orderbook_features(self, df, prefix=""):
        """Calculate orderbook-based features"""
        df_features = df.copy()
        
        # Basic price features
        df_features[f'{prefix}spread'] = df_features['ask_price1'] - df_features['bid_price1']
        df_features[f'{prefix}spread_pct'] = df_features[f'{prefix}spread'] / (df_features['mid_price'] + 1e-8)
        
        # Orderbook imbalance features
        df_features[f'{prefix}volume_imbalance_1'] = (df_features['bid_volume1'] - df_features['ask_volume1']) / (df_features['bid_volume1'] + df_features['ask_volume1'] + 1e-8)
        
        # Total volume features
        bid_volumes = [f'bid_volume{i}' for i in range(1, 6) if f'bid_volume{i}' in df_features.columns]
        ask_volumes = [f'ask_volume{i}' for i in range(1, 6) if f'ask_volume{i}' in df_features.columns]
        
        if bid_volumes and ask_volumes:
            df_features[f'{prefix}total_bid_volume'] = df_features[bid_volumes].sum(axis=1)
            df_features[f'{prefix}total_ask_volume'] = df_features[ask_volumes].sum(axis=1)
            df_features[f'{prefix}total_volume'] = df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume']
            
            # Volume-weighted prices
            bid_prices = [f'bid_price{i}' for i in range(1, 6) if f'bid_price{i}' in df_features.columns]
            ask_prices = [f'ask_price{i}' for i in range(1, 6) if f'ask_price{i}' in df_features.columns]
            
            if bid_prices and ask_prices:
                df_features[f'{prefix}vwap_bid'] = (df_features[bid_prices] * df_features[bid_volumes]).sum(axis=1) / (df_features[bid_volumes].sum(axis=1) + 1e-8)
                df_features[f'{prefix}vwap_ask'] = (df_features[ask_prices] * df_features[ask_volumes]).sum(axis=1) / (df_features[ask_volumes].sum(axis=1) + 1e-8)
        
        # Price impact features (if all levels available)
        if 'ask_price5' in df_features.columns and 'bid_price5' in df_features.columns:
            df_features[f'{prefix}price_impact_buy'] = (df_features['ask_price5'] - df_features['ask_price1']) / (df_features['ask_price1'] + 1e-8)
            df_features[f'{prefix}price_impact_sell'] = (df_features['bid_price1'] - df_features['bid_price5']) / (df_features['bid_price1'] + 1e-8)
        
        return df_features
    
    def calculate_time_series_features(self, df, prefix="", min_periods_for_indicators=50):
        """Calculate time-series based features"""
        df_features = df.copy()
        
        # Price-based features
        windows = [5, 10, 30, 60, 300]  # 5s, 10s, 30s, 1min, 5min
        
        for window in windows:
            if len(df_features) > window:  # Only calculate if we have enough data
                # Returns
                df_features[f'{prefix}return_{window}s'] = df_features['mid_price'].pct_change(window)
                
                # Realized volatility
                returns = df_features['mid_price'].pct_change()
                df_features[f'{prefix}realized_vol_{window}s'] = returns.rolling(window, min_periods=1).std() * np.sqrt(window)
                
                # Price momentum
                df_features[f'{prefix}momentum_{window}s'] = (df_features['mid_price'] / df_features['mid_price'].shift(window) - 1)
                
                # Volume momentum (if total_volume exists)
                if f'{prefix}total_volume' in df_features.columns:
                    df_features[f'{prefix}volume_momentum_{window}s'] = df_features[f'{prefix}total_volume'].pct_change(window)
                
                # Spread momentum
                if f'{prefix}spread' in df_features.columns:
                    df_features[f'{prefix}spread_momentum_{window}s'] = df_features[f'{prefix}spread'].pct_change(window)
        
        # Technical indicators (only if we have enough data)
        if len(df_features) >= min_periods_for_indicators:
            # Moving averages
            min_window = min(20, len(df_features) // 2)
            df_features[f'{prefix}sma_20'] = df_features['mid_price'].rolling(min_window, min_periods=1).mean()
            df_features[f'{prefix}ema_20'] = df_features['mid_price'].ewm(span=min_window, min_periods=1).mean()
            
            # Bollinger Bands
            sma = df_features['mid_price'].rolling(min_window, min_periods=1).mean()
            std = df_features['mid_price'].rolling(min_window, min_periods=1).std()
            df_features[f'{prefix}bb_upper'] = sma + (std * 2)
            df_features[f'{prefix}bb_lower'] = sma - (std * 2)
            df_features[f'{prefix}bb_position'] = (df_features['mid_price'] - df_features[f'{prefix}bb_lower']) / (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower'] + 1e-8)
            
            # RSI (simplified)
            delta = df_features['mid_price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=min(14, min_window), min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, min_window), min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df_features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def calculate_cross_asset_features(self, eth_df, is_test=False):
        """Calculate cross-asset correlation and momentum features"""
        df_features = eth_df.copy()
        
        suffix = "_test" if is_test else "_train"
        
        for asset_name in ['BTC', 'SOL']:
            asset_key = f"{asset_name}{suffix}"
            if asset_key in self.cross_assets:
                asset_df = self.cross_assets[asset_key]
                
                # Merge on timestamp
                merged = pd.merge_asof(df_features.sort_values('timestamp'), 
                                     asset_df[['timestamp', 'mid_price']].sort_values('timestamp'),
                                     on='timestamp', 
                                     suffixes=('', f'_{asset_name}'),
                                     direction='backward')
                
                if f'mid_price_{asset_name}' in merged.columns:
                    # Cross-asset returns
                    eth_returns = merged['mid_price'].pct_change()
                    asset_returns = merged[f'mid_price_{asset_name}'].pct_change()
                    
                    # Rolling correlation (shorter window to ensure we have data)
                    corr_window = min(60, len(merged) // 4) if len(merged) > 10 else 10
                    df_features[f'{asset_name}_correlation_60s'] = eth_returns.rolling(corr_window, min_periods=5).corr(asset_returns)
                    
                    # Cross-asset returns
                    df_features[f'{asset_name}_return_5s'] = merged[f'mid_price_{asset_name}'].pct_change(5)
                    df_features[f'{asset_name}_return_30s'] = merged[f'mid_price_{asset_name}'].pct_change(30)
                    
                    # Relative performance
                    df_features[f'{asset_name}_relative_performance'] = eth_returns - asset_returns
        
        return df_features
    
    def create_lagged_features(self, df, target_col=None):
        """Create lagged features for time series prediction"""
        df_features = df.copy()
        
        # Base lag features (available in both train and test)
        base_lag_features = ['mid_price', 'spread_pct', 'volume_imbalance_1']
        
        # Add realized volatility if it exists
        realized_vol_cols = [col for col in df_features.columns if 'realized_vol_60s' in col]
        if realized_vol_cols:
            base_lag_features.extend(realized_vol_cols)
        
        # Only add target lags for training data
        if target_col and target_col in df.columns:
            base_lag_features.append(target_col)
        
        lags = [1, 2, 3, 5, 10]  # 1s, 2s, 3s, 5s, 10s lags
        
        for feature in base_lag_features:
            if feature in df_features.columns:
                for lag in lags:
                    df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
        
        return df_features
    
    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Store original test data length
        original_test_length = len(self.eth_test)
        print(f"Original test data length: {original_test_length}")
        
        # Clean data
        self.eth_train = self.clean_data(self.eth_train)
        self.eth_test = self.clean_data(self.eth_test)
        
        print(f"Test data length after cleaning: {len(self.eth_test)}")
        
        # Calculate orderbook features
        self.eth_train = self.calculate_orderbook_features(self.eth_train)
        self.eth_test = self.calculate_orderbook_features(self.eth_test)
        
        # Calculate time series features
        self.eth_train = self.calculate_time_series_features(self.eth_train, min_periods_for_indicators=100)
        self.eth_test = self.calculate_time_series_features(self.eth_test, min_periods_for_indicators=50)
        
        # Calculate cross-asset features
        self.eth_train = self.calculate_cross_asset_features(self.eth_train, is_test=False)
        self.eth_test = self.calculate_cross_asset_features(self.eth_test, is_test=True)
        
        # Create lagged features
        self.eth_train = self.create_lagged_features(self.eth_train, 'label')
        self.eth_test = self.create_lagged_features(self.eth_test)  # No target column
        
        print(f"Test data length after feature engineering: {len(self.eth_test)}")
        
        # Find common features between train and test
        train_features = set(self.eth_train.columns) - {'timestamp', 'label'}
        test_features = set(self.eth_test.columns) - {'timestamp'}
        common_features = train_features.intersection(test_features)
        
        self.feature_columns = list(common_features)
        
        print(f"Total train features: {len(train_features)}")
        print(f"Total test features: {len(test_features)}")
        print(f"Common features for modeling: {len(self.feature_columns)}")
        
        # Remove rows with NaN values created by feature engineering - ONLY for training data
        initial_train_size = len(self.eth_train)
        self.eth_train = self.eth_train.dropna()
        print(f"Removed {initial_train_size - len(self.eth_train)} rows with NaN values from training data")
        
        # For test data, fill NaN instead of dropping - preserve ALL rows
        test_length_before_fill = len(self.eth_test)
        for col in self.feature_columns:
            if col in self.eth_test.columns:
                self.eth_test[col] = self.eth_test[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Test data length after filling NaN: {len(self.eth_test)} (should be {original_test_length})")
        
        # If we still lost rows, this is a problem we need to address
        if len(self.eth_test) != original_test_length:
            print(f"ERROR: Lost {original_test_length - len(self.eth_test)} test rows during feature engineering!")
            print("This will cause submission length mismatch.")
        
        print("Feature engineering completed!")
    
    def prepare_data_for_modeling(self):
        """Prepare features and target for modeling"""
        X_train = self.eth_train[self.feature_columns].copy()
        y_train = self.eth_train['label'].copy()
        X_test = self.eth_test[self.feature_columns].copy()
        
        # Handle any remaining infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with median (use training data median for both)
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)
        
        print(f"Final data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def time_series_cross_validation(self, X, y, n_splits=5):
        """Perform time series cross validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        # Use a simple model for CV to save time
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            # Train and predict
            model.fit(X_train_scaled, y_train_cv)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate correlation
            corr, _ = pearsonr(y_val_cv, y_pred)
            cv_scores.append(corr)
        
        return cv_scores
    
    def train_models(self):
        """Train multiple models and ensemble them"""
        print("Preparing data for modeling...")
        X_train, y_train, X_test = self.prepare_data_for_modeling()
        
        print("Performing time series cross validation...")
        cv_scores = self.time_series_cross_validation(X_train, y_train)
        print(f"CV Correlation scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"Mean CV Correlation: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        print("Training models...")
        
        # Model 1: Ridge Regression
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        self.models['ridge'] = ridge
        
        # Model 2: Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Model 3: XGBoost
        print("Training XGBoost...")
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
        
        # Model 4: LightGBM
        print("Training LightGBM...")
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
        
        # Evaluate models on training data
        print("\nModel evaluation on training data:")
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                y_pred = model.predict(X_train_scaled)
            else:
                y_pred = model.predict(X_train)
            
            corr, _ = pearsonr(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print(f"{model_name}: Correlation = {corr:.4f}, RMSE = {rmse:.6f}")
        
        print("Model training completed!")
        
        return X_test_scaled, X_test
    
    def generate_predictions(self, X_test_scaled, X_test):
        """Generate ensemble predictions"""
        print("Generating predictions...")
        
        predictions = {}
        
        # Get predictions from each model
        predictions['ridge'] = self.models['ridge'].predict(X_test_scaled)
        predictions['random_forest'] = self.models['random_forest'].predict(X_test)
        predictions['xgboost'] = self.models['xgboost'].predict(X_test)
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_test)
        
        # Ensemble prediction (weighted average)
        weights = {
            'ridge': 0.2,
            'random_forest': 0.2,
            'xgboost': 0.3,
            'lightgbm': 0.3
        }
        
        ensemble_pred = np.zeros(len(X_test))
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        return ensemble_pred, predictions
    
    def create_submission(self, predictions):
        """Create submission file with correct format"""
        # Create submission with sequential timestamps starting from 1
        submission = pd.DataFrame({
            'timestamp': range(1, len(predictions) + 1),
            'labels': predictions
        })
        
        submission.to_csv("submission.csv", index=False)
        print("Submission file created: submission.csv")
        print(f"Submission shape: {submission.shape}")
        print(f"Expected test rows: {len(self.eth_test)}")
        
        # Verify submission format
        if len(submission) != len(self.eth_test):
            print(f"WARNING: Submission length ({len(submission)}) doesn't match test data length ({len(self.eth_test)})")
        
        return submission
    
    def run_full_pipeline(self):
        """Run the complete forecasting pipeline"""
        print("Starting ETH Implied Volatility Forecasting Pipeline...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        # Train models
        X_test_scaled, X_test = self.train_models()
        
        # Generate predictions
        final_predictions, individual_predictions = self.generate_predictions(X_test_scaled, X_test)
        
        # Ensure predictions match test data length
        if len(final_predictions) != len(self.eth_test):
            print(f"WARNING: Predictions length ({len(final_predictions)}) != test data length ({len(self.eth_test)})")
            # If predictions are shorter, pad with mean
            if len(final_predictions) < len(self.eth_test):
                pad_length = len(self.eth_test) - len(final_predictions)
                pad_values = np.full(pad_length, np.mean(final_predictions))
                final_predictions = np.concatenate([final_predictions, pad_values])
            # If predictions are longer, truncate
            elif len(final_predictions) > len(self.eth_test):
                final_predictions = final_predictions[:len(self.eth_test)]
        
        # Create submission
        submission = self.create_submission(final_predictions)
        
        print("=" * 60)
        print("Pipeline completed successfully!")
        print(f"Final predictions shape: {final_predictions.shape}")
        print(f"Test data shape: {self.eth_test.shape}")
        print(f"Prediction statistics:")
        print(f"  Mean: {np.mean(final_predictions):.6f}")
        print(f"  Std:  {np.std(final_predictions):.6f}")
        print(f"  Min:  {np.min(final_predictions):.6f}")
        print(f"  Max:  {np.max(final_predictions):.6f}")
        
        return submission, final_predictions

# Main execution
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = ETHVolatilityForecaster()
    
    # Run the complete pipeline
    submission, predictions = forecaster.run_full_pipeline()
    
    # Display submission preview
    print("\nSubmission preview:")
    print(submission.head(10))
    print("...")
    print(submission.tail(5))