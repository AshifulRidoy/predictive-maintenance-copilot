"""
Data ingestion module for CMAPSS dataset and IoT sensor streams.
Handles data loading, validation, and schema enforcement.
"""
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from src.config import Config


class SensorReading(BaseModel):
    """Schema for individual sensor reading."""
    
    unit_id: int = Field(..., description="Equipment unit identifier")
    time_cycle: int = Field(..., ge=0, description="Operational cycle number")
    
    # Operational settings
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    
    # Sensor measurements
    T2: float = Field(..., description="Total temperature at fan inlet")
    T24: float = Field(..., description="Total temperature at LPC outlet")
    T30: float = Field(..., description="Total temperature at HPC outlet")
    T50: float = Field(..., description="Total temperature at LPT outlet")
    P2: float = Field(..., description="Pressure at fan inlet")
    P15: float = Field(..., description="Total pressure in bypass-duct")
    P30: float = Field(..., description="Total pressure at HPC outlet")
    Nf: float = Field(..., description="Physical fan speed")
    Nc: float = Field(..., description="Physical core speed")
    epr: float = Field(..., description="Engine pressure ratio")
    Ps30: float = Field(..., description="Static pressure at HPC outlet")
    phi: float = Field(..., description="Ratio of fuel flow to Ps30")
    NRf: float = Field(..., description="Corrected fan speed")
    NRc: float = Field(..., description="Corrected core speed")
    BPR: float = Field(..., description="Bypass Ratio")
    farB: float = Field(..., description="Burner fuel-air ratio")
    htBleed: float = Field(..., description="Bleed Enthalpy")
    Nf_dmd: float = Field(..., description="Demanded fan speed")
    PCNfR_dmd: float = Field(..., description="Demanded corrected fan speed")
    W31: float = Field(..., description="HPT coolant bleed")
    W32: float = Field(..., description="LPT coolant bleed")
    
    @validator('*', pre=True)
    def check_not_nan(cls, v):
        """Ensure no NaN values."""
        if isinstance(v, float) and np.isnan(v):
            raise ValueError("NaN values not allowed")
        return v


class Ingestor:
    """Handles data ingestion from CMAPSS dataset or simulated IoT streams."""
    
    COLUMN_NAMES = [
        "unit_id", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"
    ] + Config.FEATURE_COLUMNS
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data ingestor.
        
        Args:
            data_path: Path to CMAPSS dataset file. If None, uses Config.RAW_DATA_DIR
        """
        self.data_path = data_path or Config.RAW_DATA_DIR
        
    def load_cmapss_data(self, filename: str = "train_FD001.txt") -> pd.DataFrame:
        """
        Load CMAPSS dataset from text file.
        
        Args:
            filename: Name of the CMAPSS data file
            
        Returns:
            DataFrame with validated sensor readings
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            # Generate synthetic data for demonstration
            print(f"⚠️  {filename} not found. Generating synthetic data...")
            return self._generate_synthetic_data()
        
        # Load data without headers
        df = pd.read_csv(file_path, sep=" ", header=None)
        
        # Remove extra columns (CMAPSS has trailing spaces)
        df = df.dropna(axis=1, how='all')
        
        # Assign column names
        if len(df.columns) != len(self.COLUMN_NAMES):
            print(f"Expected {len(self.COLUMN_NAMES)} columns, got {len(df.columns)}")
            # Adjust column names to match
            df.columns = self.COLUMN_NAMES[:len(df.columns)]
        else:
            df.columns = self.COLUMN_NAMES
        
        print(f"✓ Loaded {len(df)} records from {filename}")
        print(f"  Units: {df['unit_id'].nunique()}")
        print(f"  Cycles: {df['time_cycle'].min()} - {df['time_cycle'].max()}")
        
        return self._validate_data(df)
    
    def _generate_synthetic_data(self, n_units: int = 10, cycles_per_unit: int = 200) -> pd.DataFrame:
        """
        Generate synthetic sensor data for testing.
        
        Args:
            n_units: Number of equipment units
            cycles_per_unit: Operating cycles per unit
            
        Returns:
            Synthetic DataFrame
        """
        np.random.seed(42)
        data = []
        
        for unit in range(1, n_units + 1):
            for cycle in range(1, cycles_per_unit + 1):
                # Simulate degradation over time
                degradation_factor = cycle / cycles_per_unit
                
                row = {
                    "unit_id": unit,
                    "time_cycle": cycle,
                    "op_setting_1": np.random.uniform(-0.0007, 0.0010),
                    "op_setting_2": np.random.uniform(0.0003, 0.0008),
                    "op_setting_3": np.random.uniform(100, 120),
                }
                
                # Sensor values with degradation
                row.update({
                    "T2": 642.0 + np.random.normal(0, 2) + degradation_factor * 5,
                    "T24": 642.0 + np.random.normal(0, 2) + degradation_factor * 5,
                    "T30": 1580.0 + np.random.normal(0, 10) + degradation_factor * 15,
                    "T50": 1400.0 + np.random.normal(0, 10) + degradation_factor * 12,
                    "P2": 518.0 + np.random.normal(0, 2),
                    "P15": 21.6 + np.random.normal(0, 0.5),
                    "P30": 553.0 + np.random.normal(0, 5) + degradation_factor * 10,
                    "Nf": 2388.0 + np.random.normal(0, 10),
                    "Nc": 9050.0 + np.random.normal(0, 50),
                    "epr": 1.3 + np.random.normal(0, 0.02),
                    "Ps30": 47.5 + np.random.normal(0, 2),
                    "phi": 520.0 + np.random.normal(0, 10),
                    "NRf": 2388.0 + np.random.normal(0, 10),
                    "NRc": 9050.0 + np.random.normal(0, 50),
                    "BPR": 8.4 + np.random.normal(0, 0.2),
                    "farB": 0.03 + np.random.normal(0, 0.001),
                    "htBleed": 390.0 + np.random.normal(0, 5),
                    "Nf_dmd": 2388.0 + np.random.normal(0, 10),
                    "PCNfR_dmd": 2388.0 + np.random.normal(0, 10),
                    "W31": 38.0 + np.random.normal(0, 1),
                    "W32": 23.0 + np.random.normal(0, 1),
                })
                
                data.append(row)
        
        df = pd.DataFrame(data)
        print(f"✓ Generated synthetic data: {len(df)} records for {n_units} units")
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data schema and quality.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"⚠️  Missing values detected:\n{missing[missing > 0]}")
            # Forward fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            print("⚠️  Infinite values detected and replaced with median")
            for col in df.select_dtypes(include=[np.number]).columns:
                df.loc[np.isinf(df[col]), col] = df[col].median()
        
        return df
    
    def get_unit_data(self, df: pd.DataFrame, unit_id: int) -> pd.DataFrame:
        """
        Extract data for a specific equipment unit.
        
        Args:
            df: Full dataset
            unit_id: Unit identifier
            
        Returns:
            DataFrame for specified unit
        """
        unit_df = df[df['unit_id'] == unit_id].sort_values('time_cycle').reset_index(drop=True)
        return unit_df
    
    def simulate_stream(self, df: pd.DataFrame, unit_id: int, window_size: int = 50):
        """
        Simulate a streaming data source by yielding windows of data.
        
        Args:
            df: Full dataset
            unit_id: Unit to simulate
            window_size: Size of sliding window
            
        Yields:
            DataFrame windows
        """
        unit_df = self.get_unit_data(df, unit_id)
        
        for i in range(window_size, len(unit_df) + 1):
            window = unit_df.iloc[i - window_size:i]
            yield window
