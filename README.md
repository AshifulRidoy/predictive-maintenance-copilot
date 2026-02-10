# Predictive Maintenance Copilot

> **AI-Powered Equipment Monitoring with LangGraph + IoT**

A hybrid AI system combining ML models for numeric prediction and LangGraph agents for reasoning and explanation. Demonstrates engineering maturity through deterministic orchestration, evidence-backed recommendations via RAG, and complete audit trails.

## 🎯 Overview

The Predictive Maintenance Copilot is an autonomous AI-powered system that:
- **Monitors** IoT sensor streams from turbofan engines
- **Predicts** equipment failure using trained ML models (LSTM, Autoencoder, Isolation Forest)
- **Reasons** about root causes using LLM APIs (via OpenRouter)
- **Retrieves** maintenance knowledge from manuals using RAG (LlamaIndex + Qdrant)
- **Generates** actionable repair instructions with human-in-the-loop controls
- **Maintains** complete audit trails for compliance and debugging

### Key Features

✅ **Hybrid Architecture**: ML models for predictions, LLMs for reasoning  
✅ **Explainable AI**: Full transparency through LLM-generated explanations  
✅ **Deterministic Orchestration**: LangGraph state machine for predictable workflows  
✅ **Evidence-Backed**: RAG ensures recommendations are grounded in documentation  
✅ **Production-Ready**: Complete logging, safety gates, and human approval flows  

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IoT Sensor Streams                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Data Ingestion & Preprocessing                  │
│  • Schema validation  • Feature engineering  • Normalization │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   ML Inference Engine                        │
│  • LSTM Autoencoder (Anomaly)  • RUL Predictor (LSTM)       │
│  • Isolation Forest            • Feature Attribution         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Agent Orchestration                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   ML     │→ │ Diagnose │→ │   RAG    │→ │   Plan   │   │
│  │ Inference│  │  (LLM)   │  │ Retrieve │  │  (LLM)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         ↓            ↓             ↓             ↓          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Safety Check → Explain → Approval Gate       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                         │
│  • Real-time monitoring  • AI reasoning  • Human approval   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### Required
- **Python 3.9+**
- **Docker** (for Qdrant vector database)
- **OpenRouter API Key** ([Get free tier](https://openrouter.ai/))

### Optional (Production)
- AWS Account (for DynamoDB audit logs)
- AWS Credentials configured

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd predictive-maintenance-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get free key at: https://openrouter.ai/
OPENROUTER_API_KEY=your_key_here
```

### 3. Start Qdrant Vector Database

```bash
# Start Qdrant using Docker Compose
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 4. Train ML Models

```bash
# Train anomaly detection and RUL prediction models
python -m src.ml.trainer

# This will:
# - Generate synthetic CMAPSS data (or use real data if available)
# - Train LSTM Autoencoder for anomaly detection
# - Train RUL Predictor for failure prediction
# - Train Isolation Forest baseline
# - Save all models to models/ directory
```

Expected output:
```
✓ Loaded 2000 records from synthetic data
✓ Autoencoder training complete
✓ RUL Predictor training complete
✓ Isolation Forest training complete
Models saved to: models/
```

### 5. Build RAG Knowledge Base

```bash
# Index maintenance manuals into Qdrant
python -m src.rag.index_builder

# This will:
# - Create sample maintenance manuals
# - Generate embeddings using sentence-transformers
# - Build vector index in Qdrant
```

Expected output:
```
✓ Created turbine_maintenance_guide.txt
✓ Created troubleshooting_procedures.txt
✓ Created safety_procedures.txt
✓ Loaded 3 documents
✓ Index built successfully
```

### 6. Launch Dashboard

```bash
# Start Streamlit app
streamlit run src/ui/app.py
```

The dashboard will open at `http://localhost:8501`

## 🎮 Using the System

### Dashboard Walkthrough

1. **Select Equipment Unit**: Choose from available turbofan units
2. **Select Operating Cycle**: Use slider to navigate through operational history
3. **Run Analysis**: Click "Run Analysis" to invoke the AI agent

The system will:
- Run ML inference on sensor data
- Use LLM to diagnose root cause
- Retrieve relevant maintenance procedures via RAG
- Generate step-by-step maintenance plan
- Check safety conditions
- Provide human-readable explanation

### Understanding Results

**Risk Levels:**
- 🟢 **LOW**: Normal operation, continue monitoring
- 🟡 **MEDIUM**: Anomaly detected, schedule inspection
- 🔴 **HIGH**: Critical risk, immediate action required

**Human Approval:**
When safety thresholds are exceeded, the system requires human approval before executing maintenance plans.

## 📁 Project Structure

```
predictive-maintenance-copilot/
├── src/
│   ├── config.py              # Configuration and settings
│   ├── data/
│   │   ├── ingestor.py        # CMAPSS data loading
│   │   └── preprocessor.py    # Feature engineering
│   ├── ml/
│   │   ├── models.py          # PyTorch/sklearn models
│   │   ├── trainer.py         # Training pipeline
│   │   └── inference.py       # Inference engine
│   ├── rag/
│   │   ├── index_builder.py   # Vector index creation
│   │   └── retriever.py       # Document retrieval
│   ├── agent/
│   │   ├── state.py           # LangGraph state definition
│   │   ├── nodes.py           # Agent nodes (ML + LLM)
│   │   └── graph.py           # Workflow orchestration
│   ├── ui/
│   │   └── app.py             # Streamlit dashboard
│   └── utils/
│       └── logger.py          # Audit trail logging
├── data/
│   ├── raw/                   # Sensor data
│   └── manuals/               # Maintenance manuals
├── models/                    # Trained model artifacts
├── docker-compose.yml         # Qdrant service
├── pyproject.toml            # Dependencies
└── README.md
```

## 🔧 Configuration

Edit `src/config.py` to customize:

### ML Hyperparameters
- `AUTOENCODER_HIDDEN_DIM`: Autoencoder architecture
- `RUL_LSTM_HIDDEN_DIM`: RUL predictor architecture
- `SEQUENCE_LENGTH`: Input sequence length (default: 50)

### Safety Thresholds
- `HIGH_RISK_RUL_THRESHOLD`: RUL cycles requiring approval (default: 5)
- `ANOMALY_SCORE_THRESHOLD`: Anomaly detection threshold (default: 0.7)
- `FAILURE_PROB_THRESHOLD`: Failure probability threshold (default: 0.8)

### RAG Settings
- `RAG_TOP_K`: Number of documents to retrieve (default: 3)
- `EMBEDDING_MODEL`: HuggingFace embedding model

### LLM Settings
- `LLM_MODEL`: OpenRouter model (default: llama-3-8b-instruct)
- `LLM_TEMPERATURE`: Temperature for generation (default: 0.1)

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Manual Testing Scenarios

**Scenario 1: Normal Operation**
- Select Unit 1, Cycle 50
- Expect: LOW risk, no maintenance required

**Scenario 2: Degradation Detection**
- Select Unit 5, Cycle 180
- Expect: MEDIUM risk, scheduled maintenance

**Scenario 3: Critical Failure**
- Select Unit 3, Cycle 195
- Expect: HIGH risk, immediate action, approval required

## 📊 Audit Trail

All decisions are logged for compliance:

### Local Development
Logs stored in: `data/local_db.json`

### Production
Configure AWS credentials in `.env` for DynamoDB logging:
- `MaintenanceDecisions`: Complete decision records
- `LLMCallLogs`: All LLM API calls
- `RetrievedEvidence`: RAG retrieval results
- `ModelMetadata`: Model versions and metrics

## 🔐 Security Best Practices

1. **Never commit API keys**: Use `.env` file (gitignored)
2. **Rotate credentials**: Regularly update OpenRouter API keys
3. **Human approval**: Always require approval for HIGH risk actions
4. **Audit logs**: Enable CloudTrail in production
5. **Network security**: Use VPC for AWS services

## 🚨 Troubleshooting

### Qdrant Connection Failed
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant
```

### OpenRouter API Errors
- Verify API key in `.env`
- Check rate limits: [OpenRouter Dashboard](https://openrouter.ai/activity)
- Free tier: 20 requests/minute

### Model Not Found
```bash
# Retrain models
python -m src.ml.trainer
```

### Import Errors
```bash
# Reinstall package
pip install -e .
```

## 📚 References

- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **LlamaIndex**: https://docs.llamaindex.ai/
- **Qdrant**: https://qdrant.tech/documentation/
- **OpenRouter**: https://openrouter.ai/docs
- **CMAPSS Dataset**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## 🤝 Contributing

This is a portfolio/demonstration project. For production use:
1. Replace synthetic data with real sensor streams
2. Fine-tune models on domain-specific data
3. Implement streaming data pipeline (Kafka/Kinesis)
4. Add comprehensive test suite
5. Deploy on cloud infrastructure

## 📝 License

MIT License - See LICENSE file

## 👤 Author

Built as a demonstration of:
- Hybrid AI architecture (ML + LLM)
- Deterministic agent orchestration with LangGraph
- RAG for evidence-backed AI decisions
- Production-ready ML system design

---

**Note**: This system uses synthetic data for demonstration. In production, replace with real sensor telemetry and conduct thorough validation before deploying to critical infrastructure.
