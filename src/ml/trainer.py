"""
Training script for ML models.
Handles model training, validation, and saving.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pickle

from src.config import Config
from src.ml.models import LSTMAutoEncoder, RULRegressor, IsolationForestModel, compute_reconstruction_error
from src.data import Ingestor, Preprocessor


class ModelTrainer:
    """Handles training of ML models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            device: Device to use (cuda/cpu). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
    
    def train_autoencoder(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        lr: float = None
    ) -> Tuple[LSTMAutoEncoder, dict]:
        """
        Train LSTM Autoencoder for anomaly detection.
        
        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            X_val: Validation sequences
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Trained model and training history
        """
        epochs = epochs or Config.AUTOENCODER_EPOCHS
        batch_size = batch_size or Config.AUTOENCODER_BATCH_SIZE
        lr = lr or Config.AUTOENCODER_LR
        
        # Initialize model
        input_dim = X_train.shape[2]
        sequence_length = X_train.shape[1]
        
        model = LSTMAutoEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None
        }
        
        print(f"\nTraining Autoencoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(self.device)
                        reconstructed = model(x)
                        loss = criterion(reconstructed, x)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        print("✓ Autoencoder training complete")
        return model, history
    
    def train_rul_predictor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        lr: float = None
    ) -> Tuple[RULRegressor, dict]:
        """
        Train RUL predictor.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Trained model and training history
        """
        epochs = epochs or Config.RUL_EPOCHS
        batch_size = batch_size or Config.RUL_BATCH_SIZE
        lr = lr or Config.RUL_LR
        
        # Initialize model
        input_dim = X_train.shape[2]
        
        model = RULRegressor(input_dim=input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).unsqueeze(1)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None
        }
        
        print(f"\nTraining RUL Predictor for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        predictions = model(x_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        print("✓ RUL Predictor training complete")
        return model, history
    
    def train_isolation_forest(self, X_train: np.ndarray) -> IsolationForestModel:
        """
        Train Isolation Forest.
        
        Args:
            X_train: Training data
            
        Returns:
            Trained model
        """
        print("\nTraining Isolation Forest...")
        
        model = IsolationForestModel()
        model.fit(X_train)
        
        print("✓ Isolation Forest training complete")
        return model
    
    def save_model(self, model, path: Path, metadata: dict = None):
        """
        Save trained model.
        
        Args:
            model: Model to save
            path: Save path
            metadata: Additional metadata
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, nn.Module):
            # PyTorch model
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'metadata': metadata or {}
            }
            torch.save(save_dict, path)
        else:
            # Scikit-learn model
            save_dict = {
                'model': model,
                'metadata': metadata or {}
            }
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)
        
        print(f"✓ Model saved to {path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ML Model Training Pipeline")
    print("=" * 60)
    
    # Initialize components
    ingestor = Ingestor()
    preprocessor = Preprocessor()
    trainer = ModelTrainer()
    
    # Load data
    print("\n1. Loading data...")
    df = ingestor.load_cmapss_data()
    
    # Add RUL target
    df = preprocessor.add_rul_target(df)
    
    # Split train/val (80/20 by units)
    units = df['unit_id'].unique()
    n_train = int(0.8 * len(units))
    train_units = units[:n_train]
    val_units = units[n_train:]
    
    df_train = df[df['unit_id'].isin(train_units)]
    df_val = df[df['unit_id'].isin(val_units)]
    
    print(f"  Train units: {len(train_units)}, Val units: {len(val_units)}")
    
    # Fit preprocessor on training data
    print("\n2. Preprocessing...")
    preprocessor.fit(df_train)
    df_train_scaled = preprocessor.transform(df_train)
    df_val_scaled = preprocessor.transform(df_val)
    
    # Create sequences
    print("  Creating sequences...")
    X_train, y_train = preprocessor.create_sequences_all_units(df_train_scaled)
    X_val, y_val = preprocessor.create_sequences_all_units(df_val_scaled)
    
    print(f"  Train sequences: {X_train.shape}")
    print(f"  Val sequences: {X_val.shape}")
    
    # Train models
    print("\n3. Training models...")
    
    # Autoencoder
    autoencoder, ae_history = trainer.train_autoencoder(X_train, X_val)
    trainer.save_model(
        autoencoder,
        Config.MODELS_DIR / 'autoencoder.pt',
        {'input_dim': X_train.shape[2], 'sequence_length': X_train.shape[1]}
    )
    
    # RUL Predictor
    rul_model, rul_history = trainer.train_rul_predictor(X_train, y_train, X_val, y_val)
    trainer.save_model(
        rul_model,
        Config.MODELS_DIR / 'rul_predictor.pt',
        {'input_dim': X_train.shape[2]}
    )
    
    # Isolation Forest
    iso_forest = trainer.train_isolation_forest(X_train)
    trainer.save_model(iso_forest, Config.MODELS_DIR / 'isolation_forest.pkl')
    
    # Save preprocessor
    preprocessor.save(Config.MODELS_DIR / 'preprocessor.pkl')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {Config.MODELS_DIR}")


if __name__ == "__main__":
    main()
