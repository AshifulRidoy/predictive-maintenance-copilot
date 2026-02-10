"""
Agent nodes for LangGraph workflow.
Implements ML inference, LLM reasoning, and decision logic.
"""
import time
from typing import Dict, Any
import numpy as np
from openai import OpenAI

from src.agent.state import AgentState, add_node_execution, add_llm_call
from src.ml import InferenceEngine
from src.rag import MaintenanceRetriever
from src.config import Config


class AgentNodes:
    """Collection of nodes for the maintenance agent."""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        retriever: MaintenanceRetriever
    ):
        """
        Initialize agent nodes.
        
        Args:
            inference_engine: ML inference engine
            retriever: RAG retriever
        """
        self.inference_engine = inference_engine
        self.retriever = retriever
        
        # Initialize LLM client for OpenRouter
        self.llm_client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_API_KEY
        )
    
    def run_ml_inference(self, state: AgentState) -> AgentState:
        """
        Run ML inference on sensor data.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with ML predictions
        """
        state = add_node_execution(state, "run_ml_inference")
        
        try:
            # Extract sensor sequence from state
            sensor_data = state['sensor_data']
            
            # Assuming sensor_data contains a preprocessed sequence
            if 'sequence' in sensor_data:
                sequence = np.array(sensor_data['sequence'])
            else:
                # Create dummy sequence for testing
                sequence = np.random.randn(1, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS))
            
            # Run inference
            predictions = self.inference_engine.run_full_inference(sequence)
            
            # Extract feature importance
            feature_importance = self.inference_engine.get_feature_importance(sequence)
            
            # Update state
            state['ml_predictions'] = predictions
            state['anomaly_score'] = predictions.get('combined_anomaly_score', 0.0)
            state['failure_probability'] = predictions.get('failure_probability', 0.0)
            state['rul_prediction'] = predictions.get('rul_prediction', 100.0)
            state['risk_level'] = predictions.get('risk_level', 'LOW')
            state['feature_importance'] = feature_importance
            
        except Exception as e:
            state['errors'].append(f"ML inference error: {str(e)}")
            # Set default values
            state['anomaly_score'] = 0.0
            state['failure_probability'] = 0.0
            state['rul_prediction'] = 100.0
            state['risk_level'] = 'LOW'
        
        return state
    
    def diagnose_root_cause(self, state: AgentState) -> AgentState:
        """
        Use LLM to diagnose root cause from ML predictions.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with root cause analysis
        """
        state = add_node_execution(state, "diagnose_root_cause")
        
        # Build prompt with ML evidence
        top_features = list(state.get('feature_importance', {}).items())[:5]
        features_str = "\n".join([f"- {name}: {score:.3f}" for name, score in top_features])
        
        prompt = f"""You are an expert turbofan engine diagnostician. Analyze the following sensor data and ML predictions to identify the most likely failure mode.

SENSOR EVIDENCE:
- Anomaly Score: {state['anomaly_score']:.2f} (0=normal, 1=critical)
- Failure Probability: {state['failure_probability']:.2f}
- Remaining Useful Life: {state['rul_prediction']:.0f} cycles
- Risk Level: {state['risk_level']}

TOP CONTRIBUTING SENSORS:
{features_str}

Based on this evidence, provide:
1. The most likely failure mode (one short label)
2. Brief technical reasoning (2-3 sentences)

Format your response as:
FAILURE_MODE: [label]
REASONING: [explanation]"""

        try:
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in turbofan engine diagnostics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Log the call
            state = add_llm_call(
                state,
                Config.LLM_MODEL,
                prompt,
                response_text,
                tokens_used,
                latency_ms
            )
            
            # Parse response
            lines = response_text.strip().split('\n')
            failure_mode = "Unknown"
            reasoning = ""
            
            for line in lines:
                if line.startswith("FAILURE_MODE:"):
                    failure_mode = line.replace("FAILURE_MODE:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
            
            state['failure_mode'] = failure_mode
            state['reasoning'] = reasoning
            state['root_cause_analysis'] = response_text
            
        except Exception as e:
            state['errors'].append(f"Root cause diagnosis error: {str(e)}")
            state['failure_mode'] = "Diagnosis unavailable"
            state['reasoning'] = f"Error: {str(e)}"
        
        return state
    
    def retrieve_maintenance_docs(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant maintenance documentation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved docs
        """
        state = add_node_execution(state, "retrieve_maintenance_docs")
        
        try:
            # Build query from failure mode and symptoms
            failure_mode = state.get('failure_mode', 'general maintenance')
            risk_level = state.get('risk_level', 'LOW')
            
            query = f"{failure_mode} troubleshooting maintenance procedures {risk_level} risk"
            
            # Retrieve documents
            docs = self.retriever.retrieve(query)
            
            # Format context
            context = self.retriever.format_context(docs)
            
            state['retrieved_docs'] = docs
            state['rag_context'] = context
            
        except Exception as e:
            state['errors'].append(f"Document retrieval error: {str(e)}")
            state['retrieved_docs'] = []
            state['rag_context'] = "No documentation available."
        
        return state
    
    def plan_maintenance(self, state: AgentState) -> AgentState:
        """
        Generate maintenance plan using LLM with RAG context.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with maintenance plan
        """
        state = add_node_execution(state, "plan_maintenance")
        
        prompt = f"""You are a maintenance planning expert. Create a detailed, step-by-step maintenance plan.

DIAGNOSIS:
Failure Mode: {state.get('failure_mode', 'Unknown')}
Risk Level: {state.get('risk_level', 'UNKNOWN')}
RUL: {state.get('rul_prediction', 0):.0f} cycles
Reasoning: {state.get('reasoning', 'N/A')}

RELEVANT MAINTENANCE PROCEDURES:
{state.get('rag_context', 'No procedures found.')}

Create a maintenance plan with:
1. PRIORITY: (IMMEDIATE/SCHEDULED/MONITOR)
2. ESTIMATED_DURATION: (e.g., "2-4 hours")
3. REQUIRED_PARTS: (comma-separated list or "None")
4. PROCEDURE: (numbered steps)

Format your response exactly as shown above with these four sections."""

        try:
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a maintenance planning expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Log the call
            state = add_llm_call(
                state,
                Config.LLM_MODEL,
                prompt,
                response_text,
                tokens_used,
                latency_ms
            )
            
            # Parse response
            state['maintenance_plan'] = response_text
            
            # Extract priority
            if "IMMEDIATE" in response_text.upper():
                state['action_priority'] = "IMMEDIATE"
            elif "SCHEDULED" in response_text.upper():
                state['action_priority'] = "SCHEDULED"
            else:
                state['action_priority'] = "MONITOR"
            
            # Simple parsing for duration and parts
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith("ESTIMATED_DURATION:"):
                    state['estimated_duration'] = line.replace("ESTIMATED_DURATION:", "").strip()
                elif line.startswith("REQUIRED_PARTS:"):
                    parts_str = line.replace("REQUIRED_PARTS:", "").strip()
                    if parts_str.lower() != "none":
                        state['required_parts'] = [p.strip() for p in parts_str.split(',')]
                    else:
                        state['required_parts'] = []
            
        except Exception as e:
            state['errors'].append(f"Maintenance planning error: {str(e)}")
            state['maintenance_plan'] = f"Error generating plan: {str(e)}"
            state['action_priority'] = "MONITOR"
        
        return state
    
    def check_safety(self, state: AgentState) -> AgentState:
        """
        Check safety conditions and set approval gates.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with safety checks
        """
        state = add_node_execution(state, "check_safety")
        
        safety_concerns = []
        
        # Check RUL threshold
        if state['rul_prediction'] < Config.HIGH_RISK_RUL_THRESHOLD:
            safety_concerns.append(f"RUL below safety threshold ({Config.HIGH_RISK_RUL_THRESHOLD} cycles)")
        
        # Check failure probability
        if state['failure_probability'] > Config.FAILURE_PROB_THRESHOLD:
            safety_concerns.append(f"Failure probability above threshold ({Config.FAILURE_PROB_THRESHOLD})")
        
        # Check anomaly score
        if state['anomaly_score'] > Config.ANOMALY_SCORE_THRESHOLD:
            safety_concerns.append(f"Anomaly score above threshold ({Config.ANOMALY_SCORE_THRESHOLD})")
        
        # Check for immediate priority
        if state.get('action_priority') == 'IMMEDIATE':
            safety_concerns.append("Action requires immediate attention")
        
        state['safety_concerns'] = safety_concerns
        state['safety_flag'] = len(safety_concerns) > 0
        state['requires_approval'] = state['safety_flag']  # Require approval if safety flag raised
        
        return state
    
    def explain_decision(self, state: AgentState) -> AgentState:
        """
        Generate user-facing explanation of the decision.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with explanation
        """
        state = add_node_execution(state, "explain_decision")
        
        prompt = f"""Explain this maintenance decision to an equipment operator in clear, non-technical language.

SITUATION:
- Equipment Status: {state['risk_level']} risk
- Predicted Issue: {state.get('failure_mode', 'None detected')}
- Time Until Potential Failure: {state['rul_prediction']:.0f} operating cycles

PLANNED ACTION:
{state.get('maintenance_plan', 'No action required')}

Provide a concise (2-3 sentences) explanation that:
1. States what the AI detected
2. Why it matters
3. What action is recommended

Be reassuring and clear."""

        try:
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant explaining technical decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=300
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Log the call
            state = add_llm_call(
                state,
                Config.LLM_MODEL,
                prompt,
                response_text,
                tokens_used,
                latency_ms
            )
            
            state['explanation'] = response_text
            
        except Exception as e:
            state['errors'].append(f"Explanation generation error: {str(e)}")
            state['explanation'] = "An automated maintenance recommendation has been generated."
        
        # Calculate overall confidence
        confidence_factors = []
        
        if state['anomaly_score'] < 0.3:
            confidence_factors.append(0.9)
        elif state['anomaly_score'] < 0.7:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        if len(state.get('retrieved_docs', [])) > 0:
            confidence_factors.append(0.8)
        
        state['confidence'] = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        return state
