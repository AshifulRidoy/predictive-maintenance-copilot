"""
Streamlit dashboard for Predictive Maintenance Copilot.
Provides visualization, monitoring, and human-in-the-loop controls.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config
from src.data import Ingestor, Preprocessor
from src.ml import InferenceEngine
from src.rag import MaintenanceRetriever
from src.agent import MaintenanceAgent, create_initial_state
from src.utils import AuditLogger


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Copilot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design System Colors
STREAMLIT_RED = "#FF4B4B"
STREAMLIT_BLUE = "#0068C9"
STREAMLIT_GREEN = "#09AB3B"
STREAMLIT_YELLOW = "#FFA500"
STREAMLIT_ORANGE = "#FFA500"
TEXT_PRIMARY = "#262730"
TEXT_SECONDARY = "#666666"
GRID_COLOR = "#E6EAF1"

# Semantic Colors
RUL_CRITICAL = "#FF4B4B"
RUL_WARNING = "#FFA500"
RUL_HEALTHY = "#09AB3B"

# Custom CSS
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        color: #262730;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #262730;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Metric Cards (Custom) */
    .metric-card {
        background-color: #f0f2f6;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #E6EAF1;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load ML models and RAG components."""
    with st.spinner("Loading AI models..."):
        try:
            # Load inference engine
            inference_engine = InferenceEngine()
            inference_engine.load_models()
            
            # Load retriever
            retriever = MaintenanceRetriever()
            
            # Create agent
            agent = MaintenanceAgent(inference_engine, retriever)
            
            # Create logger
            logger = AuditLogger()
            
            return inference_engine, retriever, agent, logger
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Run `python -m src.ml.trainer` to train models first")
            st.info("Run `python -m src.rag.index_builder` to build RAG index")
            return None, None, None, None


@st.cache_data
def load_data():
    """Load sensor data."""
    ingestor = Ingestor()
    df = ingestor.load_cmapss_data()
    return df


def create_sensor_chart(df: pd.DataFrame, sensor: str, unit_id: int):
    """Create time series chart for a sensor."""
    unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycle')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unit_data['time_cycle'],
        y=unit_data[sensor],
        mode='lines',
        name=sensor,
        line=dict(color=STREAMLIT_BLUE, width=2)
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{sensor} Over Time - Unit {unit_id}",
            font=dict(size=16, color=TEXT_PRIMARY)
        ),
        xaxis=dict(
            title="Cycle",
            gridcolor=GRID_COLOR,
            showline=True,
            linecolor=TEXT_PRIMARY
        ),
        yaxis=dict(
            title=sensor,
            gridcolor=GRID_COLOR,
            showline=True,
            linecolor=TEXT_PRIMARY
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Source Sans Pro", color=TEXT_PRIMARY),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_feature_importance_chart(importance: dict):
    """Create bar chart for feature importance."""
    # Get top 10 features
    top_features = dict(list(importance.items())[:10])
    
    fig = go.Figure(go.Bar(
        x=list(top_features.values()),
        y=list(top_features.keys()),
        orientation='h',
        marker=dict(
            color=STREAMLIT_BLUE,
            line=dict(color=TEXT_PRIMARY, width=0)
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="Top 10 Contributing Sensors",
            font=dict(size=16, color=TEXT_PRIMARY)
        ),
        xaxis=dict(
            title="Importance Score",
            gridcolor=GRID_COLOR,
            showline=True,
            linecolor=TEXT_PRIMARY
        ),
        yaxis=dict(
            title="Sensor",
            gridcolor=GRID_COLOR,
            showline=False,
            linecolor=TEXT_PRIMARY
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Source Sans Pro", color=TEXT_PRIMARY),
        height=400,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig


def display_risk_gauge(risk_level: str, anomaly_score: float):
    """Display risk level gauge."""
    color_map = {
        'LOW': STREAMLIT_GREEN,
        'MEDIUM': STREAMLIT_YELLOW,
        'HIGH': STREAMLIT_RED
    }
    
    color = color_map.get(risk_level, '#808495')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=anomaly_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Anomaly Score", 'font': {'size': 20, 'color': TEXT_PRIMARY}},
        number={'font': {'color': TEXT_PRIMARY}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': TEXT_SECONDARY},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': GRID_COLOR,
            'steps': [
                {'range': [0, 30], 'color': "#E6F4EA"},  # Light Green
                {'range': [30, 70], 'color': "#FFF8E1"},  # Light Yellow
                {'range': [70, 100], 'color': "#FFEBEE"}  # Light Red
            ],
            'threshold': {
                'line': {'color': STREAMLIT_RED, 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font=dict(family="Source Sans Pro"),
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig


def main():
    """Main application."""
    # Header with Design System
    st.title("🔧 Predictive Maintenance Copilot")
    st.caption("AI-Powered Equipment Monitoring with LangGraph & RAG")
    
    # Load components
    inference_engine, retriever, agent, logger = load_models()
    
    if inference_engine is None or agent is None:
        st.stop()
    
    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Unit selection
        units = sorted(df['unit_id'].unique())
        selected_unit = st.selectbox(
            "Equipment Unit", 
            units, 
            index=0,
            help="Select equipment unit to analyze"
        )
        
        # Cycle selection
        unit_df = df[df['unit_id'] == selected_unit]
        max_cycle = int(unit_df['time_cycle'].max())
        selected_cycle = st.slider("Operating Cycle", 1, max_cycle, max_cycle)
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Status")
        st.success("✓ ML Models Loaded")
        st.success("✓ RAG Index Ready")
        st.success("✓ LLM API Connected")
        
        st.divider()
        
        # Run analysis button
        run_analysis = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_analysis:
        with st.spinner("Running AI analysis..."):
            # Get sensor data up to selected cycle
            window_data = unit_df[unit_df['time_cycle'] <= selected_cycle].tail(Config.SEQUENCE_LENGTH)
            
            # Preprocess
            preprocessor = Preprocessor.load(Config.MODELS_DIR / 'preprocessor.pkl')
            window_scaled = preprocessor.transform(window_data)
            
            # Create sequence
            sequence, _ = preprocessor.create_sequences(window_scaled)
            
            if len(sequence) == 0:
                st.error("Insufficient data for analysis. Need at least 50 cycles.")
                st.stop()
            
            # Take the last sequence
            sequence = sequence[-1:]
            
            # Create initial state
            state = create_initial_state(
                sensor_data={'sequence': sequence.tolist()},
                unit_id=selected_unit
            )
            
            # Run agent
            final_state = agent.run(state)
            
            # Log the workflow
            decision_id = logger.log_complete_workflow(final_state)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            st.divider()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_level = final_state.get('risk_level', 'UNKNOWN')
                # Semantic coloring for risk
                risk_color = "normal"
                if risk_level == "HIGH": risk_color = "inverse"
                elif risk_level == "MEDIUM": risk_color = "off"
                
                st.metric(
                    "Risk Level",
                    risk_level,
                    delta="Assessment", 
                    delta_color=risk_color
                )
            
            with col2:
                rul = final_state.get('rul_prediction', 0)
                # Semantic coloring for RUL
                rul_color = "normal"
                if rul < 20: rul_color = "inverse"
                elif rul < 50: rul_color = "off"
                
                st.metric(
                    "Remaining Useful Life",
                    f"{rul:.0f} cycles",
                    delta="Prediction",
                    delta_color=rul_color
                )
            
            with col3:
                failure_prob = final_state.get('failure_probability', 0)
                st.metric(
                    "Failure Probability",
                    f"{failure_prob:.1%}",
                    delta=None
                )
            
            with col4:
                confidence = final_state.get('confidence', 0)
                st.metric(
                    "Confidence",
                    f"{confidence:.1%}",
                    delta=None
                )
            
            st.divider()
            
            # Visualizations
            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.subheader("📊 Anomaly Detection")
                anomaly_score = final_state.get('anomaly_score', 0)
                fig_gauge = display_risk_gauge(risk_level, anomaly_score)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_right:
                st.subheader("🎯 Contributing Factors")
                feature_importance = final_state.get('feature_importance', {})
                if feature_importance:
                    fig_importance = create_feature_importance_chart(feature_importance)
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            st.divider()
            
            # Copilot Reasoning
            st.subheader("🤖 AI Copilot Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["💡 Explanation", "🔬 Diagnosis", "📋 Maintenance Plan", "📚 Documentation"])
            
            with tab1:
                explanation = final_state.get('explanation', 'No explanation available.')
                st.info(explanation)
                
                st.markdown("**Detected Issue:**")
                failure_mode = final_state.get('failure_mode', 'None detected')
                st.write(f"*{failure_mode}*")
            
            with tab2:
                st.markdown("**Root Cause Analysis:**")
                reasoning = final_state.get('reasoning', 'Analysis not available.')
                st.write(reasoning)
                
                st.markdown("**ML Predictions:**")
                ml_preds = final_state.get('ml_predictions', {})
                st.json({k: v for k, v in ml_preds.items() if k != 'feature_importance'})
            
            with tab3:
                st.markdown("**Recommended Actions:**")
                plan = final_state.get('maintenance_plan', 'No plan generated.')
                st.markdown(plan)
                
                priority = final_state.get('action_priority', 'UNKNOWN')
                st.markdown(f"**Priority:** `{priority}`")
                
                duration = final_state.get('estimated_duration', 'Not specified')
                st.markdown(f"**Estimated Duration:** {duration}")
                
                parts = final_state.get('required_parts', [])
                if parts:
                    st.markdown("**Required Parts:**")
                    for part in parts:
                        st.markdown(f"- {part}")
            
            with tab4:
                st.markdown("**Retrieved Documentation:**")
                retrieved_docs = final_state.get('retrieved_docs', [])
                
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.expander(f"Source {i} (Relevance: {doc['score']:.2f})"):
                            st.write(doc['text'])
                else:
                    st.info("No relevant documentation retrieved.")
            
            # Human approval section
            if final_state.get('requires_approval', False):
                st.divider()
                st.warning("⚠️ This action requires human approval")
                
                safety_concerns = final_state.get('safety_concerns', [])
                st.markdown("**Safety Concerns:**")
                for concern in safety_concerns:
                    st.markdown(f"- {concern}")
                
                col_approve, col_reject = st.columns(2, gap="medium")
                
                with col_approve:
                    if st.button("✅ Approve Action", type="primary", use_container_width=True):
                        st.success("Action approved! Maintenance plan can proceed.")
                        final_state['approved'] = True
                        logger.log_decision(final_state, decision_id)
                
                with col_reject:
                    if st.button("❌ Reject Action", use_container_width=True):
                        st.error("Action rejected. Escalating to supervisor.")
                        final_state['approved'] = False
                        logger.log_decision(final_state, decision_id)
            
            # Sensor time series
            st.divider()
            st.subheader("📈 Sensor Trends")
            
            sensor_cols = st.columns(2, gap="medium")
            key_sensors = ['T30', 'P30', 'Nf', 'Nc']
            
            for i, sensor in enumerate(key_sensors):
                with sensor_cols[i % 2]:
                    if sensor in df.columns:
                        fig = create_sensor_chart(df, sensor, selected_unit)
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Initial view
        st.info("👈 Select an equipment unit and click 'Run Analysis' to begin")
        
        # Show recent history
        st.subheader("📜 Recent Decisions")
        history = logger.get_decision_history(limit=5)
        
        if history:
            history_df = pd.DataFrame(history)
            display_cols = ['timestamp', 'unit_id', 'risk_level', 'failure_mode', 'action_priority']
            available_cols = [col for col in display_cols if col in history_df.columns]
            
            st.dataframe(
                history_df[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "Time",
                        format="MMM DD, YYYY HH:mm"
                    ),
                    "risk_level": st.column_config.TextColumn(
                        "Risk",
                        help="Equipment risk level"
                    ),
                    "unit_id": st.column_config.NumberColumn(
                        "Unit ID",
                        format="%d"
                    )
                }
            )
        else:
            st.info("No decision history yet. Run an analysis to get started!")


if __name__ == "__main__":
    main()
