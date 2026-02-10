"""
Logging utility for audit trails.
Supports both local JSON logging and DynamoDB for production.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from src.config import Config


class AuditLogger:
    """Structured logging for complete audit trails."""
    
    def __init__(self, use_dynamodb: bool = None):
        """
        Initialize audit logger.
        
        Args:
            use_dynamodb: Whether to use DynamoDB. Defaults to Config setting.
        """
        self.use_dynamodb = use_dynamodb if use_dynamodb is not None else not Config.USE_LOCAL_DYNAMO
        
        if self.use_dynamodb:
            try:
                import boto3
                self.dynamodb = boto3.resource(
                    'dynamodb',
                    region_name=Config.AWS_REGION,
                    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
                )
                self.tables = {
                    'decisions': self.dynamodb.Table(Config.DYNAMO_DECISIONS_TABLE),
                    'llm_logs': self.dynamodb.Table(Config.DYNAMO_LLM_LOGS_TABLE),
                    'evidence': self.dynamodb.Table(Config.DYNAMO_EVIDENCE_TABLE),
                    'model_metadata': self.dynamodb.Table(Config.DYNAMO_MODEL_METADATA_TABLE)
                }
                print("✓ Connected to DynamoDB for audit logging")
            except Exception as e:
                print(f"⚠️  DynamoDB connection failed: {e}")
                print("Falling back to local JSON logging")
                self.use_dynamodb = False
        
        if not self.use_dynamodb:
            # Use local JSON file
            Config.DATA_DIR.mkdir(exist_ok=True)
            self.log_file = Config.LOCAL_DYNAMO_PATH
            self._init_local_log()
    
    def _init_local_log(self):
        """Initialize local JSON log file."""
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                json.dump({
                    'decisions': [],
                    'llm_logs': [],
                    'evidence': [],
                    'model_metadata': []
                }, f, indent=2)
    
    def log_decision(self, state: Dict[str, Any], decision_id: str = None) -> str:
        """
        Log a maintenance decision.
        
        Args:
            state: Agent state after workflow execution
            decision_id: Optional decision ID (generated if None)
            
        Returns:
            Decision ID
        """
        decision_id = decision_id or str(uuid.uuid4())
        
        record = {
            'decision_id': decision_id,
            'timestamp': state.get('timestamp', datetime.utcnow().isoformat()),
            'unit_id': state.get('unit_id'),
            'risk_level': state.get('risk_level'),
            'failure_mode': state.get('failure_mode'),
            'rul_prediction': state.get('rul_prediction'),
            'anomaly_score': state.get('anomaly_score'),
            'failure_probability': state.get('failure_probability'),
            'action_priority': state.get('action_priority'),
            'maintenance_plan': state.get('maintenance_plan'),
            'requires_approval': state.get('requires_approval'),
            'approved': state.get('approved'),
            'confidence': state.get('confidence'),
            'execution_time_ms': self._get_execution_time(state),
            'errors': state.get('errors', [])
        }
        
        if self.use_dynamodb:
            try:
                self.tables['decisions'].put_item(Item=record)
            except Exception as e:
                print(f"Error logging to DynamoDB: {e}")
                self._log_to_local('decisions', record)
        else:
            self._log_to_local('decisions', record)
        
        return decision_id
    
    def log_llm_calls(self, state: Dict[str, Any], decision_id: str):
        """
        Log all LLM API calls.
        
        Args:
            state: Agent state
            decision_id: Associated decision ID
        """
        llm_calls = state.get('llm_calls', [])
        
        for i, call in enumerate(llm_calls):
            record = {
                'call_id': f"{decision_id}-{i}",
                'decision_id': decision_id,
                'timestamp': call['timestamp'],
                'model': call['model'],
                'prompt': call['prompt'],
                'response': call['response'],
                'tokens': call['tokens'],
                'latency_ms': call['latency_ms']
            }
            
            if self.use_dynamodb:
                try:
                    self.tables['llm_logs'].put_item(Item=record)
                except Exception as e:
                    print(f"Error logging LLM call to DynamoDB: {e}")
                    self._log_to_local('llm_logs', record)
            else:
                self._log_to_local('llm_logs', record)
    
    def log_retrieved_evidence(self, state: Dict[str, Any], decision_id: str):
        """
        Log retrieved RAG evidence.
        
        Args:
            state: Agent state
            decision_id: Associated decision ID
        """
        retrieved_docs = state.get('retrieved_docs', [])
        
        for i, doc in enumerate(retrieved_docs):
            record = {
                'evidence_id': f"{decision_id}-{i}",
                'decision_id': decision_id,
                'timestamp': datetime.utcnow().isoformat(),
                'text': doc.get('text', ''),
                'relevance_score': doc.get('score', 0.0),
                'source': doc.get('metadata', {}).get('file_name', 'unknown')
            }
            
            if self.use_dynamodb:
                try:
                    self.tables['evidence'].put_item(Item=record)
                except Exception as e:
                    print(f"Error logging evidence to DynamoDB: {e}")
                    self._log_to_local('evidence', record)
            else:
                self._log_to_local('evidence', record)
    
    def log_complete_workflow(self, state: Dict[str, Any]) -> str:
        """
        Log complete workflow execution.
        
        Args:
            state: Final agent state
            
        Returns:
            Decision ID
        """
        decision_id = self.log_decision(state)
        self.log_llm_calls(state, decision_id)
        self.log_retrieved_evidence(state, decision_id)
        
        print(f"✓ Logged workflow execution: {decision_id}")
        return decision_id
    
    def _log_to_local(self, table: str, record: Dict[str, Any]):
        """
        Log to local JSON file.
        
        Args:
            table: Table name
            record: Record to log
        """
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            data[table].append(record)
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error writing to local log: {e}")
    
    def _get_execution_time(self, state: Dict[str, Any]) -> float:
        """Calculate execution time in milliseconds."""
        try:
            start = datetime.fromisoformat(state['workflow_start_time'])
            end = datetime.fromisoformat(state.get('workflow_end_time', datetime.utcnow().isoformat()))
            return (end - start).total_seconds() * 1000
        except:
            return 0.0
    
    def get_decision_history(self, unit_id: Optional[int] = None, limit: int = 10) -> list:
        """
        Retrieve decision history.
        
        Args:
            unit_id: Filter by unit ID
            limit: Maximum number of records
            
        Returns:
            List of decisions
        """
        if self.use_dynamodb:
            # DynamoDB query implementation would go here
            pass
        else:
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                
                decisions = data.get('decisions', [])
                
                if unit_id is not None:
                    decisions = [d for d in decisions if d.get('unit_id') == unit_id]
                
                # Sort by timestamp, most recent first
                decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                return decisions[:limit]
            except Exception as e:
                print(f"Error retrieving history: {e}")
                return []
