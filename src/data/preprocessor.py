"""
Data preprocessing module for feature engineering and normalization.
Handles sequence generation, feature extraction, and data scaling.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
from src.config import Config


class Preprocessor:
    """Handles feature engineering and data normalization."""
    
    def __init__(self):
        """Initialize preprocessor with scalers."""
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
        
    def add_rul_target(self, df: pd.DataFrame, max_rul: int = 130) -> pd.DataFrame:
        """
        Add Remaining Useful Life (RUL) target column.
        
        Args:
            df: DataFrame with unit_id and time_cycle
            max_rul: Maximum RUL value to cap
            
        Returns:
            DataFrame with RUL column
        """
        df = df.copy()
        
        # Calculate RUL for each unit
        rul_list = []
        for unit in df['unit_id'].unique():
            unit_df = df[df['unit_id'] == unit]
            max_cycle = unit_df['time_cycle'].max()
            
            unit_rul = max_cycle - unit_df['time_cycle']
            # Cap RUL at max_rul
            unit_rul = unit_rul.clip(upper=max_rul)
            rul_list.append(unit_rul)
        
        df['RUL'] = pd.concat(rul_list).values
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        # Select numeric sensor columns
        sensor_cols = [col for col in Config.FEATURE_COLUMNS if col in df.columns]
        
        for col in sensor_cols:
            # Rolling mean
            df[f'{col}_rolling_mean'] = df.groupby('unit_id')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            # Rolling std
            df[f'{col}_rolling_std'] = df.groupby('unit_id')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).fillna(0)
        
        return df
    
    def fit(self, df: pd.DataFrame, feature_cols: list = None):
        """
        Fit scalers on training data.
        
        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to scale
        """
        if feature_cols is None:
            feature_cols = Config.FEATURE_COLUMNS
        
        # Fit feature scaler
        self.feature_scaler.fit(df[feature_cols])
        
        # Fit target scaler if RUL exists
        if 'RUL' in df.columns:
            self.target_scaler.fit(df[['RUL']])
        
        self.is_fitted = True
        self.feature_cols = feature_cols
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Scale features
        df[self.feature_cols] = self.feature_scaler.transform(df[self.feature_cols])
        
        # Scale target if exists
        if 'RUL' in df.columns:
            df['RUL'] = self.target_scaler.transform(df[['RUL']])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            feature_cols: Feature columns to scale
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_cols)
        return self.transform(df)
    
    def inverse_transform_rul(self, rul_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform RUL predictions back to original scale.
        
        Args:
            rul_scaled: Scaled RUL values
            
        Returns:
            Original scale RUL values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        if len(rul_scaled.shape) == 1:
            rul_scaled = rul_scaled.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(rul_scaled).flatten()
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        sequence_length: int = None,
        feature_cols: list = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            df: Input DataFrame (must be for single unit)
            sequence_length: Length of sequences
            feature_cols: Feature columns to include
            
        Returns:
            Tuple of (X_sequences, y_targets) or (X_sequences, None)
        """
        if sequence_length is None:
            sequence_length = Config.SEQUENCE_LENGTH
        
        if feature_cols is None:
            feature_cols = self.feature_cols if self.is_fitted else Config.FEATURE_COLUMNS
        
        # Extract feature values
        feature_data = df[feature_cols].values
        
        # Create sequences
        X = []
        y = [] if 'RUL' in df.columns else None
        
        for i in range(len(feature_data) - sequence_length + 1):
            X.append(feature_data[i:i + sequence_length])
            
            if y is not None:
                # Target is RUL at the end of sequence
                y.append(df['RUL'].iloc[i + sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y) if y is not None else None
        
        return X, y
    
    def create_sequences_all_units(
        self,
        df: pd.DataFrame,
        sequence_length: int = None,
        feature_cols: list = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for all units in dataset.
        
        Args:
            df: Full DataFrame with multiple units
            sequence_length: Length of sequences
            feature_cols: Feature columns
            
        Returns:
            Combined sequences from all units
        """
        X_list = []
        y_list = [] if 'RUL' in df.columns else None
        
        for unit_id in df['unit_id'].unique():
            unit_df = df[df['unit_id'] == unit_id].sort_values('time_cycle')
            
            X_seq, y_seq = self.create_sequences(unit_df, sequence_length, feature_cols)
            
            X_list.append(X_seq)
            if y_list is not None and y_seq is not None:
                y_list.append(y_seq)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list) if y_list is not None else None
        
        return X, y
    
    def save(self, path: Path):
        """Save preprocessor state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'is_fitted': self.is_fitted,
            'feature_cols': self.feature_cols if self.is_fitted else None
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'Preprocessor':
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.feature_scaler = state['feature_scaler']
        preprocessor.target_scaler = state['target_scaler']
        preprocessor.is_fitted = state['is_fitted']
        preprocessor.feature_cols = state['feature_cols']
        
        print(f"✓ Preprocessor loaded from {path}")
        return preprocessor
