"""
Agent state definition for LangGraph.
Defines the state structure passed between nodes.
"""
from typing import TypedDict, Dict, List, Optional, Any
from datetime import datetime


class AgentState(TypedDict, total=False):
    """
    State for the Predictive Maintenance Agent.
    
    This state is passed between nodes in the LangGraph workflow.
    """
    
    # Input data
    sensor_data: Dict[str, Any]  # Raw sensor readings
    unit_id: int  # Equipment unit identifier
    timestamp: str  # ISO format timestamp
    
    # ML predictions
    ml_predictions: Dict[str, Any]  # Output from InferenceEngine
    anomaly_score: float  # Combined anomaly score [0, 1]
    failure_probability: float  # Probability of failure [0, 1]
    rul_prediction: float  # Remaining Useful Life in cycles
    risk_level: str  # LOW, MEDIUM, HIGH
    feature_importance: Dict[str, float]  # Feature attribution
    
    # LLM reasoning
    root_cause_analysis: Optional[str]  # LLM-generated root cause
    failure_mode: Optional[str]  # Identified failure mode
    reasoning: Optional[str]  # LLM reasoning chain
    
    # RAG retrieval
    retrieved_docs: List[Dict[str, Any]]  # Documents from RAG
    rag_context: Optional[str]  # Formatted context string
    
    # Action planning
    maintenance_plan: Optional[str]  # Step-by-step maintenance plan
    action_priority: Optional[str]  # IMMEDIATE, SCHEDULED, MONITOR
    estimated_duration: Optional[str]  # Time estimate
    required_parts: Optional[List[str]]  # Parts list
    
    # Safety and approval
    safety_flag: bool  # Whether human approval is required
    safety_concerns: Optional[List[str]]  # List of safety issues
    requires_approval: bool  # Human-in-the-loop gate
    approved: Optional[bool]  # Approval status
    
    # Explanation
    explanation: Optional[str]  # User-facing explanation
    confidence: Optional[float]  # Overall confidence score
    
    # Metadata
    workflow_start_time: str  # Workflow start timestamp
    workflow_end_time: Optional[str]  # Workflow end timestamp
    node_execution_log: List[str]  # Log of executed nodes
    
    # LLM call tracking
    llm_calls: List[Dict[str, Any]]  # All LLM API calls made
    
    # Errors
    errors: List[str]  # Any errors encountered


class NodeExecutionMetadata(TypedDict):
    """Metadata for node execution tracking."""
    node_name: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str]


def create_initial_state(
    sensor_data: Dict[str, Any],
    unit_id: int
) -> AgentState:
    """
    Create initial agent state.
    
    Args:
        sensor_data: Sensor readings
        unit_id: Equipment unit ID
        
    Returns:
        Initial AgentState
    """
    return AgentState(
        sensor_data=sensor_data,
        unit_id=unit_id,
        timestamp=datetime.utcnow().isoformat(),
        workflow_start_time=datetime.utcnow().isoformat(),
        node_execution_log=[],
        llm_calls=[],
        errors=[],
        safety_flag=False,
        requires_approval=False
    )


def add_node_execution(state: AgentState, node_name: str) -> AgentState:
    """
    Add node execution to log.
    
    Args:
        state: Current state
        node_name: Name of executed node
        
    Returns:
        Updated state
    """
    state['node_execution_log'].append(f"{datetime.utcnow().isoformat()} - {node_name}")
    return state


def add_llm_call(
    state: AgentState,
    model: str,
    prompt: str,
    response: str,
    tokens: int,
    latency_ms: float
) -> AgentState:
    """
    Log an LLM API call.
    
    Args:
        state: Current state
        model: Model used
        prompt: Prompt sent
        response: Response received
        tokens: Token count
        latency_ms: Latency in milliseconds
        
    Returns:
        Updated state
    """
    call_record = {
        'timestamp': datetime.utcnow().isoformat(),
        'model': model,
        'prompt': prompt,
        'response': response,
        'tokens': tokens,
        'latency_ms': latency_ms
    }
    
    state['llm_calls'].append(call_record)
    return state
