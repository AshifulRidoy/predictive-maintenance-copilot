"""
Inference engine for running predictions on sensor data.
Loads trained models and provides unified prediction interface.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle

from src.config import Config
from src.ml.models import LSTMAutoEncoder, RULRegressor, IsolationForestModel, compute_reconstruction_error
from src.data import Preprocessor


class InferenceEngine:
    """Unified inference engine for all ML models."""
    
    def __init__(
        self,
        models_dir: Path = None,
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            models_dir: Directory containing trained models
            device: Device to use (cuda/cpu)
        """
        self.models_dir = models_dir or Config.MODELS_DIR
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model placeholders
        self.autoencoder: Optional[LSTMAutoEncoder] = None
        self.rul_predictor: Optional[RULRegressor] = None
        self.iso_forest: Optional[IsolationForestModel] = None
        self.preprocessor: Optional[Preprocessor] = None
        
        # Model metadata
        self.ae_metadata = {}
        self.rul_metadata = {}
        
        # Anomaly score thresholds (to be calibrated)
        self.ae_threshold = None
        
    def load_models(self):
        """Load all trained models."""
        print("Loading models...")
        
        # Load preprocessor
        preprocessor_path = self.models_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = Preprocessor.load(preprocessor_path)
        else:
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        # Load autoencoder
        ae_path = self.models_dir / 'autoencoder.pt'
        if ae_path.exists():
            checkpoint = torch.load(ae_path, map_location=self.device)
            self.ae_metadata = checkpoint.get('metadata', {})
            
            self.autoencoder = LSTMAutoEncoder(
                input_dim=self.ae_metadata['input_dim'],
                sequence_length=self.ae_metadata['sequence_length']
            ).to(self.device)
            
            self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
            self.autoencoder.eval()
            print("  ✓ Autoencoder loaded")
        else:
            print("  ⚠️  Autoencoder not found")
        
        # Load RUL predictor
        rul_path = self.models_dir / 'rul_predictor.pt'
        if rul_path.exists():
            checkpoint = torch.load(rul_path, map_location=self.device)
            self.rul_metadata = checkpoint.get('metadata', {})
            
            self.rul_predictor = RULRegressor(
                input_dim=self.rul_metadata['input_dim']
            ).to(self.device)
            
            self.rul_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.rul_predictor.eval()
            print("  ✓ RUL Predictor loaded")
        else:
            print("  ⚠️  RUL Predictor not found")
        
        # Load Isolation Forest
        iso_path = self.models_dir / 'isolation_forest.pkl'
        if iso_path.exists():
            with open(iso_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.iso_forest = checkpoint['model']
            print("  ✓ Isolation Forest loaded")
        else:
            print("  ⚠️  Isolation Forest not found")
    
    def calibrate_thresholds(self, normal_data: np.ndarray):
        """
        Calibrate anomaly detection thresholds on normal data.
        
        Args:
            normal_data: Normal operational sequences
        """
        if self.autoencoder is None:
            print("Autoencoder not loaded, skipping calibration")
            return
        
        # Compute reconstruction errors on normal data
        with torch.no_grad():
            data_tensor = torch.FloatTensor(normal_data).to(self.device)
            reconstructed = self.autoencoder(data_tensor)
            errors = compute_reconstruction_error(data_tensor, reconstructed).cpu().numpy()
        
        # Set threshold at 95th percentile
        self.ae_threshold = np.percentile(errors, 95)
        print(f"✓ Autoencoder threshold calibrated: {self.ae_threshold:.6f}")
    
    def predict_anomaly(self, sequence: np.ndarray) -> Dict:
        """
        Detect anomalies in sensor sequence.
        
        Args:
            sequence: Input sequence (1, seq_len, features) or (seq_len, features)
            
        Returns:
            Dictionary with anomaly scores and predictions
        """
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, :]
        
        results = {}
        
        # Autoencoder-based detection
        if self.autoencoder is not None:
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(sequence).to(self.device)
                reconstructed = self.autoencoder(seq_tensor)
                error = compute_reconstruction_error(seq_tensor, reconstructed).item()
            
            results['ae_reconstruction_error'] = error
            
            if self.ae_threshold is not None:
                results['ae_is_anomaly'] = error > self.ae_threshold
                results['ae_anomaly_score'] = min(error / self.ae_threshold, 1.0)
            else:
                results['ae_anomaly_score'] = error
        
        # Isolation Forest detection
        if self.iso_forest is not None:
            iso_score = self.iso_forest.score_samples(sequence)[0]
            iso_pred = self.iso_forest.predict(sequence)[0]
            
            results['iso_anomaly_score'] = iso_score
            results['iso_is_anomaly'] = (iso_pred == -1)
        
        # Combine scores
        if 'ae_anomaly_score' in results and 'iso_anomaly_score' in results:
            results['combined_anomaly_score'] = (
                0.7 * results['ae_anomaly_score'] + 
                0.3 * results['iso_anomaly_score']
            )
        elif 'ae_anomaly_score' in results:
            results['combined_anomaly_score'] = results['ae_anomaly_score']
        elif 'iso_anomaly_score' in results:
            results['combined_anomaly_score'] = results['iso_anomaly_score']
        
        return results
    
    def predict_rul(self, sequence: np.ndarray) -> Dict:
        """
        Predict Remaining Useful Life.
        
        Args:
            sequence: Input sequence (1, seq_len, features) or (seq_len, features)
            
        Returns:
            Dictionary with RUL prediction
        """
        if self.rul_predictor is None:
            return {'error': 'RUL predictor not loaded'}
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, :]
        
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).to(self.device)
            rul_scaled = self.rul_predictor(seq_tensor).cpu().numpy()
        
        # Inverse transform to original scale
        rul = self.preprocessor.inverse_transform_rul(rul_scaled)[0]
        
        # Compute confidence (simplified)
        # In production, use ensemble or Bayesian methods
        confidence = 0.85
        
        # Compute failure probability based on RUL
        failure_prob = self._compute_failure_probability(rul)
        
        return {
            'rul_prediction': float(rul),
            'rul_confidence': confidence,
            'failure_probability': failure_prob
        }
    
    def _compute_failure_probability(self, rul: float) -> float:
        """
        Compute failure probability from RUL.
        
        Args:
            rul: Remaining Useful Life in cycles
            
        Returns:
            Failure probability [0, 1]
        """
        # Sigmoid function centered at threshold
        threshold = Config.HIGH_RISK_RUL_THRESHOLD
        steepness = 0.5
        
        prob = 1 / (1 + np.exp(steepness * (rul - threshold)))
        return float(prob)
    
    def run_full_inference(self, sequence: np.ndarray) -> Dict:
        """
        Run complete inference pipeline.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Complete prediction results
        """
        results = {}
        
        # Anomaly detection
        anomaly_results = self.predict_anomaly(sequence)
        results.update(anomaly_results)
        
        # RUL prediction
        rul_results = self.predict_rul(sequence)
        results.update(rul_results)
        
        # Determine risk level
        anomaly_score = results.get('combined_anomaly_score', 0)
        failure_prob = results.get('failure_probability', 0)
        rul = results.get('rul_prediction', 100)
        
        if failure_prob > Config.FAILURE_PROB_THRESHOLD or rul < Config.HIGH_RISK_RUL_THRESHOLD:
            results['risk_level'] = 'HIGH'
        elif anomaly_score > Config.ANOMALY_SCORE_THRESHOLD:
            results['risk_level'] = 'MEDIUM'
        else:
            results['risk_level'] = 'LOW'
        
        return results
    
    def get_feature_importance(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        Compute feature importance for the prediction.
        Simplified version - in production use SHAP or integrated gradients.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, :]
        
        # Compute variance across sequence for each feature
        feature_variance = np.var(sequence[0], axis=0)
        
        # Normalize to sum to 1
        importance = feature_variance / (feature_variance.sum() + 1e-8)
        
        # Map to feature names
        feature_names = self.preprocessor.feature_cols if self.preprocessor else Config.FEATURE_COLUMNS
        
        importance_dict = {
            name: float(score) 
            for name, score in zip(feature_names, importance)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
