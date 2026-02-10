"""
LangGraph workflow definition.
Orchestrates the agent nodes into a deterministic state machine.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from datetime import datetime

from src.agent.state import AgentState
from src.agent.nodes import AgentNodes
from src.ml import InferenceEngine
from src.rag import MaintenanceRetriever


class MaintenanceAgent:
    """LangGraph-based Predictive Maintenance Agent."""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        retriever: MaintenanceRetriever
    ):
        """
        Initialize the maintenance agent.
        
        Args:
            inference_engine: ML inference engine
            retriever: RAG retriever
        """
        self.nodes = AgentNodes(inference_engine, retriever)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("ml_inference", self.nodes.run_ml_inference)
        workflow.add_node("diagnose", self.nodes.diagnose_root_cause)
        workflow.add_node("retrieve_docs", self.nodes.retrieve_maintenance_docs)
        workflow.add_node("plan", self.nodes.plan_maintenance)
        workflow.add_node("safety_check", self.nodes.check_safety)
        workflow.add_node("explain", self.nodes.explain_decision)
        
        # Set entry point
        workflow.set_entry_point("ml_inference")
        
        # Add edges
        workflow.add_edge("ml_inference", "diagnose")
        workflow.add_edge("diagnose", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "plan")
        workflow.add_edge("plan", "safety_check")
        workflow.add_edge("safety_check", "explain")
        
        # Add conditional edge for human approval
        workflow.add_conditional_edges(
            "explain",
            self._should_wait_for_approval,
            {
                "wait_approval": "wait_for_approval",
                "complete": END
            }
        )
        
        # Add approval node
        workflow.add_node("wait_for_approval", self._wait_for_approval)
        workflow.add_edge("wait_for_approval", END)
        
        # Compile
        return workflow.compile()
    
    def _should_wait_for_approval(self, state: AgentState) -> Literal["wait_approval", "complete"]:
        """
        Determine if human approval is needed.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        if state.get('requires_approval', False) and state.get('approved') is None:
            return "wait_approval"
        return "complete"
    
    def _wait_for_approval(self, state: AgentState) -> AgentState:
        """
        Placeholder for human approval.
        In production, this would integrate with a UI or notification system.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        # In a real system, this would pause and wait for human input
        # For now, we'll just mark it as pending
        state['workflow_end_time'] = datetime.utcnow().isoformat()
        return state
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run the agent workflow.
        
        Args:
            state: Initial state
            
        Returns:
            Final state after workflow execution
        """
        # Execute the graph
        final_state = self.graph.invoke(state)
        
        # Add end timestamp if not set
        if not final_state.get('workflow_end_time'):
            final_state['workflow_end_time'] = datetime.utcnow().isoformat()
        
        return final_state
    
    def visualize(self, output_path: str = "workflow.png"):
        """
        Visualize the workflow graph.
        
        Args:
            output_path: Path to save visualization
        """
        try:
            from IPython.display import Image
            # This requires graphviz to be installed
            image_data = self.graph.get_graph().draw_png()
            with open(output_path, 'wb') as f:
                f.write(image_data)
            print(f"✓ Workflow visualization saved to {output_path}")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
            print("Install graphviz for workflow visualization")
    
    def get_workflow_summary(self, state: AgentState) -> dict:
        """
        Get a summary of the workflow execution.
        
        Args:
            state: Final state
            
        Returns:
            Summary dictionary
        """
        return {
            'unit_id': state.get('unit_id'),
            'risk_level': state.get('risk_level'),
            'failure_mode': state.get('failure_mode'),
            'rul_prediction': state.get('rul_prediction'),
            'anomaly_score': state.get('anomaly_score'),
            'action_priority': state.get('action_priority'),
            'requires_approval': state.get('requires_approval'),
            'confidence': state.get('confidence'),
            'nodes_executed': len(state.get('node_execution_log', [])),
            'llm_calls_made': len(state.get('llm_calls', [])),
            'errors': state.get('errors', []),
            'execution_time_ms': self._calculate_execution_time(state)
        }
    
    def _calculate_execution_time(self, state: AgentState) -> float:
        """Calculate total execution time in milliseconds."""
        try:
            from datetime import datetime
            start = datetime.fromisoformat(state['workflow_start_time'])
            end = datetime.fromisoformat(state['workflow_end_time'])
            return (end - start).total_seconds() * 1000
        except:
            return 0.0
