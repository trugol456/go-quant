import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
import talib  # Technical Analysis Library - install with pip install TA-Lib

# Advanced modeling
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class ETHVolatilityForecaster:
    """
    Advanced ETH Implied Volatility Forecasting Model
    
    This class implements a sophisticated approach to forecasting 10-second ahead
    implied volatility for Ethereum using high-frequency orderbook data, 
    cross-asset signals, and advanced feature engineering.
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
        self.feature_importance = {}
        
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
    
    def calculate_advanced_orderbook_features(self, df, prefix=""):
        """Calculate advanced orderbook-based features"""
        df_features = df.copy()
        
        # Basic price features
        df_features[f'{prefix}spread'] = df_features['ask_price1'] - df_features['bid_price1']
        df_features[f'{prefix}spread_pct'] = df_features[f'{prefix}spread'] / (df_features['mid_price'] + 1e-8)
        
        # Orderbook imbalance features (multiple levels)
        for level in range(1, 6):
            if f'bid_volume{level}' in df_features.columns and f'ask_volume{level}' in df_features.columns:
                df_features[f'{prefix}volume_imbalance_{level}'] = (
                    (df_features[f'bid_volume{level}'] - df_features[f'ask_volume{level}']) / 
                    (df_features[f'bid_volume{level}'] + df_features[f'ask_volume{level}'] + 1e-8)
                )
        
        # Weighted orderbook pressure
        bid_volumes = [f'bid_volume{i}' for i in range(1, 6) if f'bid_volume{i}' in df_features.columns]
        ask_volumes = [f'ask_volume{i}' for i in range(1, 6) if f'ask_volume{i}' in df_features.columns]
        
        if bid_volumes and ask_volumes:
            # Weight by inverse distance from mid price
            weights = np.array([1/i for i in range(1, len(bid_volumes)+1)])
            
            weighted_bid_vol = sum(df_features[vol] * weight for vol, weight in zip(bid_volumes, weights))
            weighted_ask_vol = sum(df_features[vol] * weight for vol, weight in zip(ask_volumes, weights))
            
            df_features[f'{prefix}weighted_orderbook_pressure'] = (
                (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol + 1e-8)
            )
            
            # Total volume features
            df_features[f'{prefix}total_bid_volume'] = df_features[bid_volumes].sum(axis=1)
            df_features[f'{prefix}total_ask_volume'] = df_features[ask_volumes].sum(axis=1)
            df_features[f'{prefix}total_volume'] = df_features[f'{prefix}total_bid_volume'] + df_features[f'{prefix}total_ask_volume']
            
            # Volume concentration (Gini coefficient approximation)
            bid_vol_array = df_features[bid_volumes].values
            ask_vol_array = df_features[ask_volumes].values
            
            df_features[f'{prefix}bid_concentration'] = self._calculate_concentration(bid_vol_array)
            df_features[f'{prefix}ask_concentration'] = self._calculate_concentration(ask_vol_array)
            
            # Volume-weighted prices
            bid_prices = [f'bid_price{i}' for i in range(1, 6) if f'bid_price{i}' in df_features.columns]
            ask_prices = [f'ask_price{i}' for i in range(1, 6) if f'ask_price{i}' in df_features.columns]
            
            if bid_prices and ask_prices:
                df_features[f'{prefix}vwap_bid'] = (df_features[bid_prices] * df_features[bid_volumes]).sum(axis=1) / (df_features[bid_volumes].sum(axis=1) + 1e-8)
                df_features[f'{prefix}vwap_ask'] = (df_features[ask_prices] * df_features[ask_volumes]).sum(axis=1) / (df_features[ask_volumes].sum(axis=1) + 1e-8)
                
                # Micro-price (more accurate than mid price)
                total_bid_vol = df_features[bid_volumes].sum(axis=1)
                total_ask_vol = df_features[ask_volumes].sum(axis=1)
                df_features[f'{prefix}micro_price'] = (
                    (df_features['ask_price1'] * total_bid_vol + df_features['bid_price1'] * total_ask_vol) /
                    (total_bid_vol + total_ask_vol + 1e-8)
                )
        
        # Price impact features (all levels)
        if 'ask_price5' in df_features.columns and 'bid_price5' in df_features.columns:
            df_features[f'{prefix}price_impact_buy'] = (df_features['ask_price5'] - df_features['ask_price1']) / (df_features['ask_price1'] + 1e-8)
            df_features[f'{prefix}price_impact_sell'] = (df_features['bid_price1'] - df_features['bid_price5']) / (df_features['bid_price1'] + 1e-8)
            
            # Market depth
            df_features[f'{prefix}market_depth_buy'] = df_features[ask_volumes].sum(axis=1) if ask_volumes else 0
            df_features[f'{prefix}market_depth_sell'] = df_features[bid_volumes].sum(axis=1) if bid_volumes else 0
        
        return df_features
    
    def _calculate_concentration(self, volume_array):
        """Calculate volume concentration (simplified Gini coefficient)"""
        if volume_array.shape[1] == 0:
            return np.zeros(volume_array.shape[0])
        
        # Normalize volumes to sum to 1 for each row
        volume_array = volume_array + 1e-8  # Avoid division by zero
        normalized = volume_array / volume_array.sum(axis=1, keepdims=True)
        
        # Calculate Gini coefficient for each row
        n = volume_array.shape[1]
        concentration = np.zeros(volume_array.shape[0])
        
        for i in range(volume_array.shape[0]):
            sorted_vols = np.sort(normalized[i])
            cumsum = np.cumsum(sorted_vols)
            concentration[i] = (2 * np.sum((np.arange(1, n+1) - (n+1)/2) * sorted_vols)) / (n * np.sum(sorted_vols))
        
        return concentration
    
    def calculate_advanced_time_series_features(self, df, prefix="", min_periods_for_indicators=50):
        """Calculate advanced time-series based features"""
        df_features = df.copy()
        
        # Price-based features with multiple timeframes
        windows = [5, 10, 30, 60, 120, 300]  # 5s, 10s, 30s, 1min, 2min, 5min
        
        for window in windows:
            if len(df_features) > window:
                # Returns
                df_features[f'{prefix}return_{window}s'] = df_features['mid_price'].pct_change(window)
                
                # Log returns (more stable)
                df_features[f'{prefix}log_return_{window}s'] = np.log(df_features['mid_price'] / df_features['mid_price'].shift(window))
                
                # Realized volatility (multiple estimators)
                returns = df_features['mid_price'].pct_change()
                df_features[f'{prefix}realized_vol_{window}s'] = returns.rolling(window, min_periods=1).std() * np.sqrt(window)
                
                # Garman-Klass volatility estimator (if we had OHLC data)
                # Using mid-price as proxy
                high_proxy = df_features['mid_price'].rolling(window, min_periods=1).max()
                low_proxy = df_features['mid_price'].rolling(window, min_periods=1).min()
                df_features[f'{prefix}gk_vol_{window}s'] = np.sqrt(
                    0.5 * np.log(high_proxy / low_proxy)**2 - 
                    (2*np.log(2) - 1) * np.log(df_features['mid_price'] / df_features['mid_price'].shift(window))**2
                )
                
                # Range-based volatility
                df_features[f'{prefix}range_vol_{window}s'] = (high_proxy - low_proxy) / df_features['mid_price']
                
                # Price momentum with different measures
                df_features[f'{prefix}momentum_{window}s'] = (df_features['mid_price'] / df_features['mid_price'].shift(window) - 1)
                df_features[f'{prefix}roc_{window}s'] = (df_features['mid_price'] - df_features['mid_price'].shift(window)) / df_features['mid_price'].shift(window)
                
                # Volatility clustering features
                if window >= 30:
                    vol_series = returns.rolling(window, min_periods=1).std()
                    df_features[f'{prefix}vol_momentum_{window}s'] = vol_series.pct_change()
                    df_features[f'{prefix}vol_mean_reversion_{window}s'] = vol_series / vol_series.rolling(window*2, min_periods=1).mean() - 1
                
                # Volume-based features
                if f'{prefix}total_volume' in df_features.columns:
                    df_features[f'{prefix}volume_momentum_{window}s'] = df_features[f'{prefix}total_volume'].pct_change(window)
                    df_features[f'{prefix}volume_ma_{window}s'] = df_features[f'{prefix}total_volume'].rolling(window, min_periods=1).mean()
                    
                    # Volume-price relationship
                    price_vol_corr = df_features['mid_price'].pct_change().rolling(window, min_periods=5).corr(
                        df_features[f'{prefix}total_volume'].pct_change()
                    )
                    df_features[f'{prefix}price_volume_corr_{window}s'] = price_vol_corr
                
                # Spread dynamics
                if f'{prefix}spread' in df_features.columns:
                    df_features[f'{prefix}spread_momentum_{window}s'] = df_features[f'{prefix}spread'].pct_change(window)
                    df_features[f'{prefix}spread_volatility_{window}s'] = df_features[f'{prefix}spread'].pct_change().rolling(window, min_periods=1).std()
        
        # Advanced technical indicators (only if we have enough data)
        if len(df_features) >= min_periods_for_indicators:
            try:
                # Convert to numpy arrays for TA-Lib
                prices = df_features['mid_price'].values.astype(np.float64)
                
                # Moving averages and bands
                df_features[f'{prefix}sma_20'] = talib.SMA(prices, timeperiod=min(20, len(prices)//2))
                df_features[f'{prefix}ema_20'] = talib.EMA(prices, timeperiod=min(20, len(prices)//2))
                df_features[f'{prefix}bb_upper'], df_features[f'{prefix}bb_middle'], df_features[f'{prefix}bb_lower'] = talib.BBANDS(prices, timeperiod=min(20, len(prices)//2))
                
                # Position in Bollinger Bands
                df_features[f'{prefix}bb_position'] = (prices - df_features[f'{prefix}bb_lower']) / (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower'] + 1e-8)
                df_features[f'{prefix}bb_width'] = (df_features[f'{prefix}bb_upper'] - df_features[f'{prefix}bb_lower']) / df_features[f'{prefix}bb_middle']
                
                # Momentum indicators
                df_features[f'{prefix}rsi'] = talib.RSI(prices, timeperiod=min(14, len(prices)//4))
                df_features[f'{prefix}macd'], df_features[f'{prefix}macd_signal'], df_features[f'{prefix}macd_hist'] = talib.MACD(prices)
                df_features[f'{prefix}cci'] = talib.CCI(prices, prices, prices, timeperiod=min(14, len(prices)//4))
                
                # Volatility indicators
                df_features[f'{prefix}atr'] = talib.ATR(prices, prices, prices, timeperiod=min(14, len(prices)//4))
                
                # Trend indicators
                df_features[f'{prefix}adx'] = talib.ADX(prices, prices, prices, timeperiod=min(14, len(prices)//4))
                
            except Exception as e:
                print(f"Warning: TA-Lib indicators failed: {e}")
                # Fallback to simple indicators
                min_window = min(20, len(df_features) // 2)
                df_features[f'{prefix}sma_20'] = df_features['mid_price'].rolling(min_window, min_periods=1).mean()
                df_features[f'{prefix}ema_20'] = df_features['mid_price'].ewm(span=min_window, min_periods=1).mean()
                
                # Simple RSI
                delta = df_features['mid_price'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=min(14, min_window), min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, min_window), min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                df_features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
        
        # Microstructure features
        if len(df_features) > 10:
            # Tick direction and intensity
            price_changes = df_features['mid_price'].diff()
            df_features[f'{prefix}tick_direction'] = np.sign(price_changes)
            df_features[f'{prefix}tick_intensity'] = np.abs(price_changes)
            
            # Price acceleration
            df_features[f'{prefix}price_acceleration'] = price_changes.diff()
            
            # Smoothed price using Savitzky-Golay filter
            try:
                window_length = min(11, len(df_features) // 2)
                if window_length % 2 == 0:
                    window_length -= 1
                if window_length >= 3:
                    df_features[f'{prefix}smoothed_price'] = savgol_filter(df_features['mid_price'], window_length, 2)
                    df_features[f'{prefix}price_noise'] = df_features['mid_price'] - df_features[f'{prefix}smoothed_price']
            except:
                pass
        
        return df_features
    
    def calculate_cross_asset_features(self, eth_df, is_test=False):
        """Calculate advanced cross-asset correlation and momentum features"""
        df_features = eth_df.copy()
        
        suffix = "_test" if is_test else "_train"
        
        for asset_name in ['BTC', 'SOL']:
            asset_key = f"{asset_name}{suffix}"
            if asset_key in self.cross_assets:
                asset_df = self.cross_assets[asset_key]
                
                # Merge on timestamp with tolerance
                merged = pd.merge_asof(df_features.sort_values('timestamp'), 
                                     asset_df[['timestamp', 'mid_price']].sort_values('timestamp'),
                                     on='timestamp', 
                                     suffixes=('', f'_{asset_name}'),
                                     direction='backward',
                                     tolerance=pd.Timedelta('5s'))  # 5-second tolerance
                
                if f'mid_price_{asset_name}' in merged.columns:
                    # Returns for different timeframes
                    eth_returns = merged['mid_price'].pct_change()
                    asset_returns = merged[f'mid_price_{asset_name}'].pct_change()
                    
                    windows = [10, 30, 60, 300]
                    for window in windows:
                        if len(merged) > window:
                            # Rolling correlation
                            df_features[f'{asset_name}_correlation_{window}s'] = eth_returns.rolling(window, min_periods=5).corr(asset_returns)
                            
                            # Rolling beta (ETH vs asset)
                            eth_ret_window = eth_returns.rolling(window, min_periods=5)
                            asset_ret_window = asset_returns.rolling(window, min_periods=5)
                            covariance = eth_ret_window.cov(asset_ret_window)
                            asset_variance = asset_ret_window.var()
                            df_features[f'{asset_name}_beta_{window}s'] = covariance / (asset_variance + 1e-8)
                            
                            # Cross-asset momentum
                            df_features[f'{asset_name}_return_{window}s'] = merged[f'mid_price_{asset_name}'].pct_change(window)
                            
                            # Relative strength
                            eth_momentum = merged['mid_price'].pct_change(window)
                            asset_momentum = merged[f'mid_price_{asset_name}'].pct_change(window)
                            df_features[f'{asset_name}_relative_strength_{window}s'] = eth_momentum - asset_momentum
                    
                    # Cross-asset volatility spillover
                    eth_vol = eth_returns.rolling(60, min_periods=5).std()
                    asset_vol = asset_returns.rolling(60, min_periods=5).std()
                    df_features[f'{asset_name}_vol_spillover'] = eth_vol.rolling(30, min_periods=5).corr(asset_vol)
                    
                    # Lead-lag relationships
                    for lag in [1, 2, 3, 5]:
                        if len(merged) > lag:
                            # Asset returns leading ETH
                            df_features[f'{asset_name}_lead_corr_{lag}s'] = eth_returns.rolling(30, min_periods=5).corr(asset_returns.shift(lag))
                            # ETH returns leading asset
                            df_features[f'{asset_name}_lag_corr_{lag}s'] = eth_returns.shift(lag).rolling(30, min_periods=5).corr(asset_returns)
                    
                    # Price level relationships
                    df_features[f'{asset_name}_price_ratio'] = merged['mid_price'] / merged[f'mid_price_{asset_name}']
                    df_features[f'{asset_name}_price_ratio_ma'] = df_features[f'{asset_name}_price_ratio'].rolling(60, min_periods=1).mean()
                    df_features[f'{asset_name}_price_ratio_std'] = df_features[f'{asset_name}_price_ratio'].rolling(60, min_periods=1).std()
        
        return df_features
    
    def create_regime_features(self, df):
        """Create market regime-based features"""
        df_features = df.copy()
        
        if len(df_features) < 60:
            return df_features
        
        # Volatility regimes
        returns = df_features['mid_price'].pct_change()
        vol_30s = returns.rolling(30, min_periods=1).std()
        vol_300s = returns.rolling(300, min_periods=1).std()
        
        # High/low volatility regime
        vol_quantiles = vol_300s.rolling(1800, min_periods=60).quantile([0.33, 0.67])  # 30-minute window
        df_features['vol_regime'] = pd.cut(vol_300s, bins=[0, vol_quantiles.iloc[:, 0].fillna(vol_300s.quantile(0.33)).values,
                                                          vol_quantiles.iloc[:, 1].fillna(vol_300s.quantile(0.67)).values,
                                                          np.inf], labels=[0, 1, 2])
        
        # Trend regimes
        sma_short = df_features['mid_price'].rolling(60, min_periods=1).mean()
        sma_long = df_features['mid_price'].rolling(300, min_periods=1).mean()
        df_features['trend_regime'] = np.where(sma_short > sma_long, 1, 0)  # 1 = uptrend, 0 = downtrend
        
        # Momentum regimes
        momentum_10s = df_features['mid_price'].pct_change(10)
        momentum_60s = df_features['mid_price'].pct_change(60)
        df_features['momentum_regime'] = np.where((momentum_10s > 0) & (momentum_60s > 0), 2,  # Strong up
                                                 np.where((momentum_10s < 0) & (momentum_60s < 0), 0,  # Strong down
                                                         1))  # Mixed/sideways
        
        return df_features
    
    def create_lagged_features(self, df, target_col=None):
        """Create comprehensive lagged features for time series prediction"""
        df_features = df.copy()
        
        # Core features for lagging
        core_features = ['mid_price', 'spread_pct', 'volume_imbalance_1', 'weighted_orderbook_pressure']
        
        # Add volatility features if they exist
        vol_features = [col for col in df_features.columns if 'realized_vol_60s' in col or 'gk_vol_60s' in col]
        core_features.extend(vol_features)
        
        # Add cross-asset features
        cross_features = [col for col in df_features.columns if any(asset in col for asset in ['BTC', 'SOL']) and 'return_30s' in col]
        core_features.extend(cross_features[:5])  # Limit to avoid too many features
        
        # Only add target lags for training data
        if target_col and target_col in df.columns:
            core_features.append(target_col)
        
        lags = [1, 2, 3, 5, 10, 20, 30]  # Multiple lag periods
        
        for feature in core_features:
            if feature in df_features.columns:
                for lag in lags:
                    df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
        
        # Create interaction features between key lags
        if 'mid_price_lag_1' in df_features.columns and 'mid_price_lag_10' in df_features.columns:
            df_features['price_momentum_lag'] = df_features['mid_price_lag_1'] / df_features['mid_price_lag_10'] - 1
        
        return df_features
    
    def feature_selection(self, X_train, y_train, max_features=100):
        """Select the most important features"""
        print(f"Performing feature selection from {X_train.shape[1]} features...")
        
        # Remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.001)
        X_train_var = variance_selector.fit_transform(X_train)
        selected_features = X_train.columns[variance_selector.get_support()]
        
        print(f"After variance filtering: {len(selected_features)} features")
        
        # Select top features using f_regression
        if len(selected_features) > max_features:
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_train_selected = selector.fit_transform(X_train[selected_features], y_train)
            final_features = selected_features[selector.get_support()]
            print(f"After statistical selection: {len(final_features)} features")
        else:
            final_features = selected_features
        
        return final_features.tolist()

    def engineer_features(self):
        """Enhanced feature engineering pipeline"""
        print("Starting enhanced feature engineering...")
        
        # Store original test data length
        original_test_length = len(self.eth_test)
        print(f"Original test data length: {original_test_length}")
        
        # Clean data
        self.eth_train = self.clean_data(self.eth_train)
        self.eth_test = self.clean_data(self.eth_test)
        
        print(f"Test data length after cleaning: {len(self.eth_test)}")
        
        # Calculate advanced orderbook features
        self.eth_train = self.calculate_advanced_orderbook_features(self.eth_train)
        self.eth_test = self.calculate_advanced_orderbook_features(self.eth_test)
        
        # Calculate advanced time series features
        self.eth_train = self.calculate_advanced_time_series_features(self.eth_train, min_periods_for_indicators=200)
        self.eth_test = self.calculate_advanced_time_series_features(self.eth_test, min_periods_for_indicators=100)
        
        # Calculate cross-asset features
        self.eth_train = self.calculate_cross_asset_features(self.eth_train, is_test=False)
        self.eth_test = self.calculate_cross_asset_features(self.eth_test, is_test=True)
        
        # Create regime features
        self.eth_train = self.create_regime_features(self.eth_train)
        self.eth_test = self.create_regime_features(self.eth_test)
        
        # Create lagged features
        self.eth_train = self.create_lagged_features(self.eth_train, 'label')
        self.eth_test = self.create_lagged_features(self.eth_test)
        
        print(f"Test data length after feature engineering: {len(self.eth_test)}")
        
        # Find common features between train and test
        train_features = set(self.eth_train.columns) - {'timestamp', 'label'}
        test_features = set(self.eth_test.columns) - {'timestamp'}
        common_features = train_features.intersection(test_features)
        
        print(f"Total train features: {len(train_features)}")
        print(f"Total test features: {len(test_features)}")
        print(f"Common features before selection: {len(common_features)}")
        
        # Remove rows with NaN values created by feature engineering - ONLY for training data
        initial_train_size = len(self.eth_train)
        self.eth_train = self.eth_train.dropna()
        print(f"Removed {initial_train_size - len(self.eth_train)} rows with NaN values from training data")
        
        # For test data, fill NaN instead of dropping - preserve ALL rows
        for col in common_features:
            if col in self.eth_test.columns:
                self.eth_test[col] = self.eth_test[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Feature selection
        if len(common_features) > 100:
            X_temp = self.eth_train[list(common_features)].fillna(0)
            y_temp = self.eth_train['label']
            selected_features = self.feature_selection(X_temp, y_temp, max_features=150)
            self.feature_columns = selected_features
        else:
            self.feature_columns = list(common_features)
        
        print(f"Final selected features for modeling: {len(self.feature_columns)}")
        print(f"Test data length after filling NaN: {len(self.eth_test)} (should be {original_test_length})")
        
        # If we still lost rows, this is a problem
        if len(self.eth_test) != original_test_length:
            print(f"ERROR: Lost {original_test_length - len(self.eth_test)} test rows during feature engineering!")
        
        print("Enhanced feature engineering completed!")
    
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
        """Perform enhanced time series cross validation with walk-forward approach"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'correlations': [],
            'rmse_scores': [],
            'mae_scores': [],
            'spearman_correlations': []
        }
        
        # Use multiple models for robust CV
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'xgb': xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        }
        
        scaler = RobustScaler()  # More robust to outliers
        
        model_cv_scores = {name: [] for name in models.keys()}
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            fold_predictions = []
            
            for model_name, model in models.items():
                if model_name == 'xgb':
                    model.fit(X_train_cv, y_train_cv)
                    y_pred = model.predict(X_val_cv)
                else:
                    model.fit(X_train_scaled, y_train_cv)
                    y_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                corr, _ = pearsonr(y_val_cv, y_pred)
                model_cv_scores[model_name].append(corr)
                fold_predictions.append(y_pred)
            
            # Ensemble prediction for this fold
            ensemble_pred = np.mean(fold_predictions, axis=0)
            
            # Calculate ensemble metrics
            corr, _ = pearsonr(y_val_cv, ensemble_pred)
            spearman_corr, _ = spearmanr(y_val_cv, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val_cv, ensemble_pred))
            mae = mean_absolute_error(y_val_cv, ensemble_pred)
            
            cv_results['correlations'].append(corr)
            cv_results['spearman_correlations'].append(spearman_corr)
            cv_results['rmse_scores'].append(rmse)
            cv_results['mae_scores'].append(mae)
        
        # Print individual model performance
        for model_name, scores in model_cv_scores.items():
            print(f"{model_name.upper()} CV Correlation: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        return cv_results
    
    def train_advanced_models(self):
        """Train multiple advanced models with optimized hyperparameters"""
        print("Preparing data for modeling...")
        X_train, y_train, X_test = self.prepare_data_for_modeling()
        
        print("Performing enhanced time series cross validation...")
        cv_results = self.time_series_cross_validation(X_train, y_train)
        print(f"Ensemble CV Pearson Correlation: {np.mean(cv_results['correlations']):.4f} (+/- {np.std(cv_results['correlations']) * 2:.4f})")
        print(f"Ensemble CV Spearman Correlation: {np.mean(cv_results['spearman_correlations']):.4f} (+/- {np.std(cv_results['spearman_correlations']) * 2:.4f})")
        print(f"Ensemble CV RMSE: {np.mean(cv_results['rmse_scores']):.6f} (+/- {np.std(cv_results['rmse_scores']) * 2:.6f})")
        
        # Multiple scalers for different model types
        standard_scaler = StandardScaler()
        robust_scaler = RobustScaler()
        minmax_scaler = MinMaxScaler()
        
        X_train_standard = standard_scaler.fit_transform(X_train)
        X_train_robust = robust_scaler.fit_transform(X_train)
        X_train_minmax = minmax_scaler.fit_transform(X_train)
        
        X_test_standard = standard_scaler.transform(X_test)
        X_test_robust = robust_scaler.transform(X_test)
        X_test_minmax = minmax_scaler.transform(X_test)
        
        self.scalers['standard'] = standard_scaler
        self.scalers['robust'] = robust_scaler
        self.scalers['minmax'] = minmax_scaler
        
        print("Training advanced models...")
        
        # Model 1: Ridge with L2 regularization (standard scaling)
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=0.5, random_state=42)
        ridge.fit(X_train_standard, y_train)
        self.models['ridge'] = ridge
        
        # Model 2: ElasticNet with L1+L2 regularization (robust scaling)
        print("Training ElasticNet...")
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        elastic.fit(X_train_robust, y_train)
        self.models['elasticnet'] = elastic
        
        # Model 3: Random Forest with optimized parameters
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Model 4: XGBoost with early stopping
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Model 5: LightGBM with dart boosting
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            boosting_type='dart',
            drop_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # Model 6: Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # Model 7: Neural Network (MLP)
        print("Training Neural Network...")
        try:
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            mlp.fit(X_train_standard, y_train)
            self.models['mlp'] = mlp
        except Exception as e:
            print(f"Warning: MLP training failed: {e}")
        
        # Store feature importance from tree-based models
        self.feature_importance['xgboost'] = dict(zip(X_train.columns, xgb_model.feature_importances_))
        self.feature_importance['lightgbm'] = dict(zip(X_train.columns, lgb_model.feature_importances_))
        self.feature_importance['random_forest'] = dict(zip(X_train.columns, rf.feature_importances_))
        
        # Evaluate models on training data
        print("\nAdvanced model evaluation on training data:")
        scaled_data = {
            'ridge': X_train_standard,
            'elasticnet': X_train_robust,
            'mlp': X_train_standard
        }
        
        for model_name, model in self.models.items():
            if model_name in scaled_data:
                y_pred = model.predict(scaled_data[model_name])
            else:
                y_pred = model.predict(X_train)
            
            corr, _ = pearsonr(y_train, y_pred)
            spearman_corr, _ = spearmanr(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print(f"{model_name}: Pearson = {corr:.4f}, Spearman = {spearman_corr:.4f}, RMSE = {rmse:.6f}")
        
        print("Advanced model training completed!")
        
        return {
            'standard': X_test_standard,
            'robust': X_test_robust,
            'minmax': X_test_minmax,
            'raw': X_test
        }
    
    def generate_predictions(self, X_test_dict):
        """Generate sophisticated ensemble predictions with multiple strategies"""
        print("Generating advanced ensemble predictions...")
        
        predictions = {}
        
        # Get predictions from each model with appropriate scaling
        predictions['ridge'] = self.models['ridge'].predict(X_test_dict['standard'])
        predictions['elasticnet'] = self.models['elasticnet'].predict(X_test_dict['robust'])
        predictions['random_forest'] = self.models['random_forest'].predict(X_test_dict['raw'])
        predictions['xgboost'] = self.models['xgboost'].predict(X_test_dict['raw'])
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_test_dict['raw'])
        predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(X_test_dict['raw'])
        
        if 'mlp' in self.models:
            predictions['mlp'] = self.models['mlp'].predict(X_test_dict['standard'])
        
        # Multiple ensemble strategies
        
        # Strategy 1: Weighted average based on CV performance (baseline)
        basic_weights = {
            'ridge': 0.10,
            'elasticnet': 0.10,
            'random_forest': 0.15,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'gradient_boosting': 0.15
        }
        
        if 'mlp' in predictions:
            basic_weights['mlp'] = 0.10
            # Renormalize weights
            total_weight = sum(basic_weights.values())
            basic_weights = {k: v/total_weight for k, v in basic_weights.items()}
        
        ensemble_basic = np.zeros(len(X_test_dict['raw']))
        for model_name, weight in basic_weights.items():
            if model_name in predictions:
                ensemble_basic += weight * predictions[model_name]
        
        # Strategy 2: Rank averaging (more robust to outliers)
        pred_matrix = np.column_stack([predictions[model] for model in predictions.keys()])
        ranks = np.apply_along_axis(lambda x: stats.rankdata(x), axis=0, arr=pred_matrix.T)
        ensemble_rank = np.mean(ranks, axis=0)
        
        # Convert ranks back to values using quantile mapping from basic ensemble
        ensemble_rank_mapped = np.percentile(ensemble_basic, 
                                           (ensemble_rank - 1) / (len(ensemble_rank) - 1) * 100)
        
        # Strategy 3: Median ensemble (robust to extreme predictions)
        ensemble_median = np.median(pred_matrix, axis=1)
        
        # Strategy 4: Trimmed mean (remove extreme 20% of predictions)
        ensemble_trimmed = stats.trim_mean(pred_matrix, 0.2, axis=1)
        
        # Strategy 5: Dynamic weighting based on recent performance (simplified)
        # Weight tree-based models higher for volatility prediction
        dynamic_weights = {
            'ridge': 0.08,
            'elasticnet': 0.08,
            'random_forest': 0.18,
            'xgboost': 0.28,
            'lightgbm': 0.28,
            'gradient_boosting': 0.18
        }
        
        if 'mlp' in predictions:
            dynamic_weights['mlp'] = 0.12
            # Renormalize
            total_weight = sum(dynamic_weights.values())
            dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        ensemble_dynamic = np.zeros(len(X_test_dict['raw']))
        for model_name, weight in dynamic_weights.items():
            if model_name in predictions:
                ensemble_dynamic += weight * predictions[model_name]
        
        # Final ensemble: Combine different strategies
        final_ensemble = (
            0.35 * ensemble_basic +      # Weighted average
            0.20 * ensemble_rank_mapped + # Rank-based
            0.15 * ensemble_median +      # Median
            0.15 * ensemble_trimmed +     # Trimmed mean
            0.15 * ensemble_dynamic       # Dynamic weighting
        )
        
        # Apply post-processing smoothing for time series
        if len(final_ensemble) > 5:
            # Light smoothing to reduce noise
            smoothed = savgol_filter(final_ensemble, window_length=min(5, len(final_ensemble)//2*2+1), polyorder=1)
            final_ensemble = 0.8 * final_ensemble + 0.2 * smoothed
        
        # Store all ensemble strategies for analysis
        ensemble_strategies = {
            'basic': ensemble_basic,
            'rank': ensemble_rank_mapped,
            'median': ensemble_median,
            'trimmed': ensemble_trimmed,
            'dynamic': ensemble_dynamic,
            'final': final_ensemble
        }
        
        print(f"Generated predictions using {len(predictions)} models")
        print(f"Prediction statistics:")
        print(f"  Mean: {np.mean(final_ensemble):.6f}")
        print(f"  Std:  {np.std(final_ensemble):.6f}")
        print(f"  Min:  {np.min(final_ensemble):.6f}")
        print(f"  Max:  {np.max(final_ensemble):.6f}")
        
        return final_ensemble, predictions, ensemble_strategies
    
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
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance from tree-based models"""
        print("\nFeature Importance Analysis:")
        print("=" * 50)
        
        # Combine importance scores from all tree-based models
        combined_importance = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                if feature not in combined_importance:
                    combined_importance[feature] = []
                combined_importance[feature].append(importance)
        
        # Calculate mean importance across models
        mean_importance = {feature: np.mean(scores) for feature, scores in combined_importance.items()}
        
        # Sort by importance
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 15 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:15]):
            print(f"{i+1:2d}. {feature:<40} {importance:.4f}")
        
        return sorted_features
    
    def run_full_pipeline(self):
        """Run the complete advanced forecasting pipeline"""
        print("Starting Advanced ETH Implied Volatility Forecasting Pipeline...")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        # Train advanced models
        X_test_dict = self.train_advanced_models()
        
        # Analyze feature importance
        feature_rankings = self.analyze_feature_importance()
        
        # Generate predictions
        final_predictions, individual_predictions, ensemble_strategies = self.generate_predictions(X_test_dict)
        
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
        
        print("=" * 70)
        print("Advanced Pipeline completed successfully!")
        print(f"Final predictions shape: {final_predictions.shape}")
        print(f"Test data shape: {self.eth_test.shape}")
        
        # Compare ensemble strategies
        print("\nEnsemble Strategy Comparison:")
        for strategy_name, predictions in ensemble_strategies.items():
            print(f"{strategy_name.upper():>12}: Mean={np.mean(predictions):.6f}, Std={np.std(predictions):.6f}")
        
        print(f"\nFinal Prediction Statistics:")
        print(f"  Mean: {np.mean(final_predictions):.6f}")
        print(f"  Std:  {np.std(final_predictions):.6f}")
        print(f"  Min:  {np.min(final_predictions):.6f}")
        print(f"  Max:  {np.max(final_predictions):.6f}")
        
        return submission, final_predictions, feature_rankings

# Main execution
if __name__ == "__main__":
    # Initialize the advanced forecaster
    forecaster = ETHVolatilityForecaster()
    
    # Run the complete advanced pipeline
    submission, predictions, feature_rankings = forecaster.run_full_pipeline()
    
    # Display submission preview
    print("\nSubmission preview:")
    print(submission.head(10))
    print("...")
    print(submission.tail(5))
    
    # Optional: Save additional analysis
    print(f"\nSaved submission.csv with {len(submission)} rows")
    print("Pipeline completed successfully!")