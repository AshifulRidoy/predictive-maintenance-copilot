"""
Verification script to test the complete system.
Run this after setup to ensure everything is working.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data import Ingestor, Preprocessor
from src.ml import InferenceEngine
from src.rag import MaintenanceRetriever
from src.agent import MaintenanceAgent, create_initial_state
from src.utils import AuditLogger
import numpy as np


def test_data_layer():
    """Test data ingestion and preprocessing."""
    print("\n" + "="*60)
    print("Testing Data Layer")
    print("="*60)
    
    # Test ingestor
    ingestor = Ingestor()
    df = ingestor.load_cmapss_data()
    assert len(df) > 0, "Failed to load data"
    print("✓ Data ingestion successful")
    
    # Test preprocessor
    preprocessor = Preprocessor()
    df_with_rul = preprocessor.add_rul_target(df)
    assert 'RUL' in df_with_rul.columns, "Failed to add RUL"
    print("✓ Preprocessing successful")
    
    return df


def test_ml_layer():
    """Test ML models."""
    print("\n" + "="*60)
    print("Testing ML Layer")
    print("="*60)
    
    try:
        # Check if models exist
        ae_path = Config.MODELS_DIR / 'autoencoder.pt'
        rul_path = Config.MODELS_DIR / 'rul_predictor.pt'
        
        if not ae_path.exists() or not rul_path.exists():
            print("⚠️  Models not found. Run: python -m src.ml.trainer")
            return False
        
        # Load inference engine
        engine = InferenceEngine()
        engine.load_models()
        print("✓ ML models loaded successfully")
        
        # Test inference
        dummy_sequence = np.random.randn(1, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS))
        results = engine.run_full_inference(dummy_sequence)
        
        assert 'combined_anomaly_score' in results, "Anomaly detection failed"
        assert 'rul_prediction' in results, "RUL prediction failed"
        print("✓ ML inference successful")
        
        return engine
        
    except Exception as e:
        print(f"✗ ML layer error: {e}")
        return None


def test_rag_layer():
    """Test RAG components."""
    print("\n" + "="*60)
    print("Testing RAG Layer")
    print("="*60)
    
    try:
        # Test retriever
        retriever = MaintenanceRetriever()
        print("✓ RAG retriever initialized")
        
        # Test query
        docs = retriever.retrieve("vibration troubleshooting")
        if len(docs) > 0:
            print(f"✓ Retrieved {len(docs)} documents")
        else:
            print("⚠️  No documents retrieved. Run: python -m src.rag.index_builder")
            return None
        
        return retriever
        
    except Exception as e:
        print(f"✗ RAG layer error: {e}")
        print("  Make sure Qdrant is running: docker-compose up -d")
        return None


def test_agent_layer(engine, retriever):
    """Test LangGraph agent."""
    print("\n" + "="*60)
    print("Testing Agent Layer")
    print("="*60)
    
    if engine is None or retriever is None:
        print("⚠️  Skipping agent test due to missing dependencies")
        return False
    
    try:
        # Create agent
        agent = MaintenanceAgent(engine, retriever)
        print("✓ LangGraph agent created")
        
        # Create test state
        dummy_sequence = np.random.randn(1, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS))
        state = create_initial_state(
            sensor_data={'sequence': dummy_sequence.tolist()},
            unit_id=1
        )
        
        # Run workflow
        if not Config.OPENROUTER_API_KEY:
            print("⚠️  OPENROUTER_API_KEY not set. Skipping LLM tests.")
            print("   Set API key in .env to test full workflow")
            return False
        
        print("Running full workflow (this may take 10-20 seconds)...")
        final_state = agent.run(state)
        
        assert 'risk_level' in final_state, "Workflow incomplete"
        assert 'explanation' in final_state, "LLM explanation missing"
        print("✓ Agent workflow completed successfully")
        
        # Print summary
        print("\nWorkflow Summary:")
        print(f"  Risk Level: {final_state.get('risk_level')}")
        print(f"  Failure Mode: {final_state.get('failure_mode')}")
        print(f"  RUL: {final_state.get('rul_prediction', 0):.0f} cycles")
        print(f"  Nodes Executed: {len(final_state.get('node_execution_log', []))}")
        print(f"  LLM Calls: {len(final_state.get('llm_calls', []))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent layer error: {e}")
        return False


def test_logging_layer():
    """Test audit logging."""
    print("\n" + "="*60)
    print("Testing Logging Layer")
    print("="*60)
    
    try:
        logger = AuditLogger()
        print("✓ Audit logger initialized")
        
        # Test logging
        test_state = {
            'timestamp': '2024-01-01T00:00:00',
            'unit_id': 1,
            'risk_level': 'LOW',
            'workflow_start_time': '2024-01-01T00:00:00',
            'workflow_end_time': '2024-01-01T00:00:10'
        }
        
        decision_id = logger.log_decision(test_state)
        print(f"✓ Logged test decision: {decision_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging layer error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PREDICTIVE MAINTENANCE COPILOT - SYSTEM VERIFICATION")
    print("="*60)
    
    # Configuration check
    print("\nConfiguration:")
    print(f"  Models Dir: {Config.MODELS_DIR}")
    print(f"  Data Dir: {Config.DATA_DIR}")
    print(f"  Qdrant: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    print(f"  LLM Model: {Config.LLM_MODEL}")
    print(f"  API Key Set: {'Yes' if Config.OPENROUTER_API_KEY else 'No'}")
    
    # Run tests
    df = test_data_layer()
    engine = test_ml_layer()
    retriever = test_rag_layer()
    test_logging_layer()
    
    # Agent test (requires API key)
    agent_success = test_agent_layer(engine, retriever)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    checks = [
        ("Data Layer", df is not None),
        ("ML Layer", engine is not None),
        ("RAG Layer", retriever is not None),
        ("Agent Layer", agent_success),
    ]
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n🎉 All systems operational!")
        print("\nNext steps:")
        print("  1. Start the dashboard: streamlit run src/ui/app.py")
        print("  2. Select a unit and run analysis")
    else:
        print("\n⚠️  Some checks failed. Please fix issues above.")
        print("\nCommon fixes:")
        print("  - Run: python -m src.ml.trainer (if ML layer failed)")
        print("  - Run: python -m src.rag.index_builder (if RAG layer failed)")
        print("  - Run: docker-compose up -d (if Qdrant failed)")
        print("  - Set OPENROUTER_API_KEY in .env (if Agent layer failed)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
