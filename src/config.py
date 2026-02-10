"""
Configuration module for Predictive Maintenance Copilot.
Manages API keys, database paths, and model hyperparameters.
"""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the Predictive Maintenance Copilot."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    MANUALS_DIR = DATA_DIR / "manuals"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # API Configuration
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # LLM Configuration
    LLM_MODEL: str = "meta-llama/llama-3-8b-instruct"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1000
    
    # AWS Configuration (for production)
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # DynamoDB Tables
    DYNAMO_DECISIONS_TABLE: str = "MaintenanceDecisions"
    DYNAMO_MODEL_METADATA_TABLE: str = "ModelMetadata"
    DYNAMO_LLM_LOGS_TABLE: str = "LLMCallLogs"
    DYNAMO_EVIDENCE_TABLE: str = "RetrievedEvidence"
    
    # Use local mock for development
    USE_LOCAL_DYNAMO: bool = os.getenv("USE_LOCAL_DYNAMO", "true").lower() == "true"
    LOCAL_DYNAMO_PATH: Path = DATA_DIR / "local_db.json"
    
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = "maintenance_manuals"
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768
    
    # ML Model Hyperparameters
    # Anomaly Detection
    AUTOENCODER_HIDDEN_DIM: int = 32
    AUTOENCODER_LATENT_DIM: int = 16
    AUTOENCODER_EPOCHS: int = 50
    AUTOENCODER_BATCH_SIZE: int = 64
    AUTOENCODER_LR: float = 0.001
    
    # RUL Prediction
    RUL_LSTM_HIDDEN_DIM: int = 64
    RUL_LSTM_LAYERS: int = 2
    RUL_EPOCHS: int = 100
    RUL_BATCH_SIZE: int = 32
    RUL_LR: float = 0.001
    
    # Isolation Forest
    ISO_FOREST_ESTIMATORS: int = 100
    ISO_FOREST_CONTAMINATION: float = 0.1
    
    # Data Processing
    SEQUENCE_LENGTH: int = 50  # Time steps for LSTM input
    FEATURE_COLUMNS = [
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", 
        "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB",
        "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
    ]
    
    # Safety thresholds
    HIGH_RISK_RUL_THRESHOLD: int = 5  # cycles
    ANOMALY_SCORE_THRESHOLD: float = 0.7
    FAILURE_PROB_THRESHOLD: float = 0.8
    
    # RAG Configuration
    RAG_TOP_K: int = 3
    RAG_SIMILARITY_THRESHOLD: float = 0.7
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration."""
        errors = []
        
        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY not set")
        
        if not cls.USE_LOCAL_DYNAMO and not (cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY):
            errors.append("AWS credentials not set for production DynamoDB")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)
        cls.MANUALS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        
        if errors:
            print("⚠️  Configuration warnings:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Validate on import
Config.validate()
