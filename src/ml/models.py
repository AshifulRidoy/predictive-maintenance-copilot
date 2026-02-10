"""
Machine Learning models for anomaly detection and RUL prediction.
Includes LSTM Autoencoder, RUL Regressor, and Isolation Forest.
"""
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import numpy as np
from src.config import Config


class LSTMAutoEncoder(nn.Module):
    """LSTM-based Autoencoder for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        latent_dim: int = None,
        sequence_length: int = None
    ):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            latent_dim: Latent representation dimension
            sequence_length: Length of input sequences
        """
        super(LSTMAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or Config.AUTOENCODER_HIDDEN_DIM
        self.latent_dim = latent_dim or Config.AUTOENCODER_LATENT_DIM
        self.sequence_length = sequence_length or Config.SEQUENCE_LENGTH
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, 
            self.hidden_dim, 
            batch_first=True
        )
        self.encoder_fc = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_lstm = nn.LSTM(
            self.hidden_dim,
            input_dim,
            batch_first=True
        )
        
    def encode(self, x):
        """Encode input to latent representation."""
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        # Use last hidden state
        latent = self.encoder_fc(h_n[-1])
        return latent
    
    def decode(self, latent):
        """Decode latent representation to reconstruction."""
        batch_size = latent.shape[0]
        
        # Project latent to hidden
        hidden = self.decoder_fc(latent)
        
        # Repeat for sequence length
        hidden_seq = hidden.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode
        reconstructed, _ = self.decoder_lstm(hidden_seq)
        
        return reconstructed
    
    def forward(self, x):
        """Forward pass: encode then decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


class RULRegressor(nn.Module):
    """LSTM-based Remaining Useful Life predictor."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = 0.2
    ):
        """
        Initialize RUL Regressor.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(RULRegressor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or Config.RUL_LSTM_HIDDEN_DIM
        self.num_layers = num_layers or Config.RUL_LSTM_LAYERS
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            RUL prediction (batch, 1)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = h_n[-1]
        
        # Fully connected
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class IsolationForestModel:
    """Isolation Forest for baseline anomaly detection."""
    
    def __init__(
        self,
        n_estimators: int = None,
        contamination: float = None,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.n_estimators = n_estimators or Config.ISO_FOREST_ESTIMATORS
        self.contamination = contamination or Config.ISO_FOREST_CONTAMINATION
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=random_state
        )
        
    def fit(self, X: np.ndarray):
        """
        Fit the model.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        # Flatten sequences if needed
        if len(X.shape) == 3:
            # (batch, seq, features) -> (batch, seq * features)
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly labels (1 for normal, -1 for anomaly)
        """
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Negative score (higher = more anomalous)
        scores = -self.model.score_samples(X)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores


def compute_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Compute reconstruction error for autoencoder.
    
    Args:
        original: Original input
        reconstructed: Reconstructed output
        
    Returns:
        Mean squared error per sample
    """
    mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2])
    return mse


def get_model_summary(model: nn.Module) -> dict:
    """
    Get summary of PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__
    }
