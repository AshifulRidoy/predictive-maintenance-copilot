# Project File Structure

```
predictive-maintenance-copilot/
│
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── ARCHITECTURE.md              # Technical architecture documentation
├── README.md                    # Project overview and quick start
├── SETUP.md                     # Detailed setup instructions
├── FILE_STRUCTURE.md            # This file
│
├── docker-compose.yml           # Qdrant service definition
├── download_cmapss.py           # ← NASA dataset download script
├── pyproject.toml               # Python project configuration
├── quick_start.sh               # Automated setup script (Unix)
├── requirements.txt             # Python dependencies
├── verify_setup.py              # System verification script
│
├── data/                        # Data directory
│   ├── manuals/                 # Maintenance manuals (auto-created)
│   │   ├── turbine_maintenance_guide.txt
│   │   ├── troubleshooting_procedures.txt
│   │   └── safety_procedures.txt
│   └── raw/                     # Raw sensor data
│       ├── train_FD001.txt      # ← Downloaded by download_cmapss.py
│       ├── test_FD001.txt
│       ├── RUL_FD001.txt
│       └── ... (other CMAPSS files)
│
├── models/                      # Trained ML models (created by trainer)
│   ├── autoencoder.pt
│   ├── rul_predictor.pt
│   ├── isolation_forest.pkl
│   └── preprocessor.pkl
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration and settings
│   │
│   ├── agent/                   # LangGraph agent components
│   │   ├── __init__.py
│   │   ├── graph.py             # Workflow orchestration
│   │   ├── nodes.py             # Agent nodes (ML + LLM)
│   │   └── state.py             # State definition
│   │
│   ├── data/                    # Data ingestion and preprocessing
│   │   ├── __init__.py
│   │   ├── ingestor.py          # CMAPSS data loading
│   │   └── preprocessor.py      # Feature engineering
│   │
│   ├── ml/                      # Machine learning models
│   │   ├── __init__.py
│   │   ├── inference.py         # Inference engine
│   │   ├── models.py            # Model definitions
│   │   └── trainer.py           # Training pipeline
│   │
│   ├── rag/                     # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── index_builder.py     # Vector index creation
│   │   └── retriever.py         # Document retrieval
│   │
│   ├── ui/                      # User interface
│   │   └── app.py               # Streamlit dashboard
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── logger.py            # Audit logging
│
└── tests/                       # Test suite (future)
    ├── __init__.py
    ├── test_data.py
    ├── test_ml.py
    ├── test_rag.py
    └── test_agent.py
```

## File Descriptions

### Root Level

**Configuration Files:**
- `.env.example` - Template for environment variables (API keys, AWS credentials)
- `.gitignore` - Specifies files to exclude from version control
- `pyproject.toml` - Python project metadata and dependencies
- `requirements.txt` - Simplified dependency list for pip

**Documentation:**
- `README.md` - Quick start guide and project overview
- `SETUP.md` - Step-by-step installation instructions
- `ARCHITECTURE.md` - Technical architecture and design decisions
- `FILE_STRUCTURE.md` - This file

**Scripts:**
- `download_cmapss.py` - **NEW** Downloads NASA CMAPSS dataset
- `verify_setup.py` - Validates installation and configuration
- `quick_start.sh` - Automated setup for Unix systems
- `docker-compose.yml` - Qdrant service configuration

### Source Code (`src/`)

**Configuration:**
- `config.py` - Centralized configuration (API keys, hyperparameters, paths)

**Data Layer (`src/data/`):**
- `ingestor.py` - Loads CMAPSS data or generates synthetic data
- `preprocessor.py` - Feature engineering, normalization, sequence generation

**ML Layer (`src/ml/`):**
- `models.py` - PyTorch/sklearn model definitions (Autoencoder, RUL, IsoForest)
- `trainer.py` - Training pipeline with validation
- `inference.py` - Unified inference engine for predictions

**RAG Layer (`src/rag/`):**
- `index_builder.py` - Creates vector index in Qdrant, generates sample manuals
- `retriever.py` - Semantic search and context formatting

**Agent Layer (`src/agent/`):**
- `state.py` - TypedDict state definition for LangGraph
- `nodes.py` - Agent nodes (ML inference, LLM reasoning, planning)
- `graph.py` - LangGraph workflow orchestration

**UI Layer (`src/ui/`):**
- `app.py` - Streamlit dashboard with visualizations and controls

**Utilities (`src/utils/`):**
- `logger.py` - Audit trail logging (DynamoDB or local JSON)

### Data Directories

**`data/raw/`:**
- Contains CMAPSS dataset files downloaded from NASA
- Auto-populated by `download_cmapss.py`
- System falls back to synthetic data if empty

**`data/manuals/`:**
- Maintenance manuals for RAG
- Auto-created by `index_builder.py` if empty
- Can add custom manuals here

**`models/`:**
- Trained model artifacts (.pt, .pkl files)
- Created by `trainer.py`
- Required before running the system

## File Creation Order

When setting up the system, files are created in this order:

1. **Manual**: `.env` (copy from `.env.example`, add API key)
2. **Docker**: `docker-compose up` creates Qdrant container
3. **Download**: `python download_cmapss.py` creates `data/raw/*.txt`
4. **Training**: `python -m src.ml.trainer` creates `models/*.pt` and `models/*.pkl`
5. **RAG**: `python -m src.rag.index_builder` creates `data/manuals/*.txt`
6. **Runtime**: Streamlit creates cached data

## Git-Ignored Files

These files are excluded from version control (see `.gitignore`):

- `.env` - Contains secrets
- `data/raw/*.txt` - Large dataset files
- `models/*.pt`, `models/*.pkl` - Large model files  
- `qdrant_storage/` - Qdrant database files
- `__pycache__/`, `*.pyc` - Python bytecode
- `venv/`, `env/` - Virtual environments

## Key Entry Points


