"""
Enhanced Streamlit App: OpenAI Text Generation with Logprob Highlighting
Optimized version with improved performance, UI/UX, and additional features.

This app demonstrates text completion with visual highlighting based on log probabilities.
Each word is colored based on the model's confidence in that token.
"""

import streamlit as st
import os
from openai import OpenAI
import numpy as np
from math import exp
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import time

import dotenv
from utils.cache_manager import CacheManager
from utils.color_schemes import ColorSchemeManager
from utils.export_manager import ExportManager
from utils.statistics import StatisticsCalculator

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="OpenAI Logprobs Text Generator", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize utility classes
cache_manager = CacheManager()
color_manager = ColorSchemeManager()
export_manager = ExportManager()
stats_calculator = StatisticsCalculator()

# Custom CSS for better mobile responsiveness and UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .token-highlight {
        display: inline-block;
        margin: 2px;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .token-highlight:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .analysis-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .token-highlight {
            margin: 1px;
            padding: 2px 4px;
            font-size: 0.9em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client with enhanced error handling
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from environment or user input."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            # Test the client with a simple request
            client.models.list()
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_completion_with_logprobs(api_key_hash, prompt, model="gpt-4o", max_tokens=100, temperature=0.7):
    """
    Get completion from OpenAI with logprobs enabled (cached version).
    
    Args:
        api_key_hash: Hash of API key for caching
        prompt: Input prompt text
        model: Model to use for completion
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Tuple of (OpenAI completion response with logprobs, error_message)
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5  # Get top 5 alternatives for each token
        )
        return response, None
    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        
        error_message = f"Error calling OpenAI API: {error_type} - {error_details}"
        
        # Return detailed error information for troubleshooting
        detailed_error = {
            "error_type": error_type,
            "error_details": error_details,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_length": len(prompt)
        }
        
        return None, detailed_error

def create_enhanced_highlighted_text(response, color_scheme="confidence"):
    """
    Create HTML with enhanced highlighted text based on logprobs.
    
    Args:
        response: OpenAI completion response with logprobs
        color_scheme: Color scheme to use for highlighting
    
    Returns:
        HTML string with highlighted text
    """
    if not response or not response.choices[0].logprobs:
        return "No logprobs available"
    
    tokens = response.choices[0].logprobs.content
    html_parts = []
    
    # Find min/max logprobs for better color scaling
    logprobs = [token.logprob for token in tokens]
    min_logprob = min(logprobs) if logprobs else -10
    max_logprob = max(logprobs) if logprobs else 0
    
    for token in tokens:
        # Decode token bytes to string
        token_str = bytes(token.bytes).decode("utf-8", errors="replace")
        
        # Calculate color based on logprob using selected scheme
        color = color_manager.get_color(token.logprob, min_logprob, max_logprob, color_scheme)
        
        # Create styled span with enhanced hover effects
        probability_percent = round(exp(token.logprob) * 100, 2)
        
        # Get alternative tokens if available
        alternatives = ""
        if hasattr(token, 'top_logprobs') and token.top_logprobs:
            alt_tokens = [bytes(alt.bytes).decode("utf-8", errors="replace") 
                         for alt in token.top_logprobs[:3] if alt.bytes != token.bytes]
            if alt_tokens:
                alternatives = f", Alternatives: {', '.join(repr(alt) for alt in alt_tokens)}"
        
        html_parts.append(
            f'<span class="token-highlight" style="background-color: {color}; color: black;" '
            f'title="Token: {repr(token_str)}, Logprob: {token.logprob:.3f}, '
            f'Probability: {probability_percent}%{alternatives}">{token_str}</span>'
        )
    
    return "".join(html_parts)

def create_enhanced_logprob_chart(response, chart_type="bar"):
    """
    Create enhanced charts showing logprob values for each token.
    
    Args:
        response: OpenAI completion response with logprobs
        chart_type: Type of chart to create ("bar", "line", "heatmap")
    
    Returns:
        Plotly figure
    """
    if not response or not response.choices[0].logprobs:
        return None
    
    tokens = response.choices[0].logprobs.content
    
    # Prepare data for chart
    data = []
    for i, token in enumerate(tokens):
        token_str = bytes(token.bytes).decode("utf-8", errors="replace")
        probability = exp(token.logprob) * 100
        
        data.append({
            "Position": i + 1,
            "Token": repr(token_str),
            "Display_Token": token_str,
            "Logprob": token.logprob,
            "Probability (%)": probability,
            "Confidence": "High" if probability > 50 else "Medium" if probability > 20 else "Low"
        })
    
    df = pd.DataFrame(data)
    
    if chart_type == "bar":
        fig = px.bar(
            df, 
            x="Position", 
            y="Probability (%)",
            hover_data=["Token", "Logprob"],
            title="Token Probabilities Distribution",
            color="Probability (%)",
            color_continuous_scale="RdYlGn",
            text="Display_Token"
        )
        fig.update_traces(textposition="outside")
        
    elif chart_type == "line":
        fig = px.line(
            df,
            x="Position",
            y="Probability (%)",
            title="Token Probability Trend",
            markers=True,
            hover_data=["Token", "Logprob"]
        )
        
    elif chart_type == "heatmap":
        # Create a heatmap view
        fig = px.imshow(
            [df["Probability (%)"].values],
            labels=dict(x="Token Position", y="", color="Probability (%)"),
            title="Token Probability Heatmap",
            color_continuous_scale="RdYlGn",
            text_auto=True
        )
    
    fig.update_layout(
        xaxis_title="Token Position",
        yaxis_title="Probability (%)",
        height=450,
        showlegend=False
    )
    
    return fig

# Enhanced model parameter presets
MODEL_PRESETS = {
    "Creative Writing": {"temperature": 0.9, "max_tokens": 150},
    "Technical Documentation": {"temperature": 0.3, "max_tokens": 100},
    "Casual Conversation": {"temperature": 0.7, "max_tokens": 80},
    "Poetry": {"temperature": 1.2, "max_tokens": 120},
    "Academic Writing": {"temperature": 0.4, "max_tokens": 200},
    "Custom": {"temperature": 0.7, "max_tokens": 100}
}

def main():
    """Main Streamlit application with enhanced features."""
    
    # Header with improved styling
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🤖 OpenAI Text Generation with Logprob Analysis")
    st.markdown("""
    Generate text with AI while visualizing the model's confidence through advanced logprob analysis.
    Each token is color-coded based on the model's certainty, with detailed statistics and export options.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key management with better UX
        client = get_openai_client()
        if not client:
            st.error("🔑 API Key Required")
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key. You can also set the OPENAI_API_KEY environment variable.",
                placeholder="sk-..."
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.rerun()
            else:
                st.warning("⚠️ Please provide your OpenAI API key to continue.")
                st.stop()
        else:
            st.success("✅ API Key Configured")
        
        st.divider()
        
        # Model selection with descriptions
        st.subheader("🎯 Model Selection")
        model_info = {
            "gpt-4o": "Latest GPT-4 Omni (Recommended)",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            "gpt-4o-mini": "Faster, cost-effective version",
            "gpt-4": "Previous generation GPT-4",
            "gpt-3.5-turbo": "Fast and efficient"
        }
        
        model = st.selectbox(
            "Model",
            list(model_info.keys()),
            format_func=lambda x: model_info[x],
            help="Select the OpenAI model for text generation."
        )
        
        st.divider()
        
        # Parameter presets
        st.subheader("🎛️ Generation Parameters")
        preset = st.selectbox(
            "Parameter Preset",
            list(MODEL_PRESETS.keys()),
            help="Choose a preset or select 'Custom' for manual configuration."
        )
        
        if preset != "Custom":
            temperature = MODEL_PRESETS[preset]["temperature"]
            max_tokens = MODEL_PRESETS[preset]["max_tokens"]
            st.info(f"Using {preset} preset: Temperature={temperature}, Max Tokens={max_tokens}")
        else:
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                help="Controls creativity. Lower = more focused, Higher = more creative"
            )
            
            max_tokens = st.slider(
                "Max Tokens", 
                min_value=10, 
                max_value=300, 
                value=100,
                help="Maximum number of tokens to generate"
            )
        
        st.divider()
        
        # Visualization options
        st.subheader("🎨 Visualization")
        color_scheme = st.selectbox(
            "Color Scheme",
            ["confidence", "rainbow", "heat", "ocean"],
            help="Choose the color scheme for confidence visualization"
        )
        
        chart_type = st.selectbox(
            "Chart Type",
            ["bar", "line", "heatmap"],
            help="Select the type of probability chart"
        )
        
        # Cache management
        st.divider()
        st.subheader("💾 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache", help="Clear all cached responses"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with col2:
            cache_info = cache_manager.get_cache_info()
            st.metric("Cached Items", cache_info.get("count", 0))
    
    # Main content area with improved layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Text Generation")
        
        # Enhanced prompt input with examples
        prompt_examples = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The key to solving climate change",
            "In the year 2050, technology will",
            "The most important lesson I learned"
        ]
        
        selected_example = st.selectbox(
            "Quick Examples (optional)",
            [""] + prompt_examples,
            help="Select an example or type your own prompt below"
        )
        
        prompt = st.text_area(
            "Enter your prompt:",
            value=selected_example if selected_example else "The future of artificial intelligence",
            height=120,
            help="Start typing and the AI will complete your text"
        )
        
        # Generation controls
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        with col_gen1:
            generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)
        with col_gen2:
            if st.button("🎲 Random", help="Generate with random prompt"):
                import random
                prompt = random.choice(prompt_examples)
                st.rerun()
        with col_gen3:
            if st.button("📋 Clear", help="Clear the prompt"):
                prompt = ""
                st.rerun()
    
    with col2:
        st.header("📊 Quick Stats")
        if hasattr(st.session_state, 'response') and st.session_state.response:
            response = st.session_state.response
            
            # Calculate and display key metrics
            perplexity = stats_calculator.calculate_perplexity(response)
            avg_confidence = stats_calculator.calculate_average_confidence(response)
            uncertainty_score = stats_calculator.calculate_uncertainty_score(response)
            
            st.metric("Perplexity", f"{perplexity:.2f}" if perplexity else "N/A")
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%" if avg_confidence else "N/A")
            st.metric("Uncertainty", f"{uncertainty_score:.2f}" if uncertainty_score else "N/A")
        else:
            st.info("Generate text to see statistics")
    
    # Generation and results
    if generate_button and prompt.strip():
        with st.spinner("🤖 Generating text..."):
            # Create a hash of the API key for caching
            api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            api_key_hash = hash(api_key) if api_key else "no_key"
            
            # Progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            response, error = get_completion_with_logprobs(
                api_key_hash,
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            progress_bar.empty()
            
            if response:
                # Store response in session state
                st.session_state.response = response
                st.session_state.original_prompt = prompt
                st.session_state.generation_params = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.success("✅ Text generated successfully!")
                
            elif error:
                st.error(f"❌ Generation failed: {error['error_details']}")
                with st.expander("🔍 Error Details"):
                    st.json(error)
                st.stop()
    
    # Display results with enhanced UI
    if hasattr(st.session_state, 'response') and st.session_state.response:
        response = st.session_state.response
        
        st.header("📝 Generated Text")
        
        # Display text in tabs for better organization
        tab1, tab2, tab3 = st.tabs(["🎨 Highlighted View", "📄 Plain Text", "🔗 Combined"])
        
        with tab1:
            st.markdown("**Confidence-Based Highlighting:**")
            highlighted_html = create_enhanced_highlighted_text(response, color_scheme)
            st.markdown(f'<div class="analysis-container">{highlighted_html}</div>', unsafe_allow_html=True)
        
        with tab2:
            completed_text = response.choices[0].message.content
            st.markdown("**Original Prompt:**")
            st.info(st.session_state.original_prompt)
            st.markdown("**AI Completion:**")
            st.success(completed_text)
        
        with tab3:
            st.markdown("**Full Text:**")
            full_text = f"{st.session_state.original_prompt} {response.choices[0].message.content}"
            st.markdown(f'<div class="analysis-container">{full_text}</div>', unsafe_allow_html=True)
        
        # Enhanced analysis section
        st.header("📊 Detailed Analysis")
        
        # Probability visualization
        fig = create_enhanced_logprob_chart(response, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics dashboard
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            perplexity = stats_calculator.calculate_perplexity(response)
            st.metric(
                "Perplexity", 
                f"{perplexity:.2f}" if perplexity else "N/A",
                help="Lower values indicate higher model confidence"
            )
        
        with col_stat2:
            avg_confidence = stats_calculator.calculate_average_confidence(response)
            st.metric(
                "Avg Confidence", 
                f"{avg_confidence:.1f}%" if avg_confidence else "N/A",
                help="Average probability across all tokens"
            )
        
        with col_stat3:
            uncertainty_score = stats_calculator.calculate_uncertainty_score(response)
            st.metric(
                "Uncertainty Score", 
                f"{uncertainty_score:.2f}" if uncertainty_score else "N/A",
                help="Measure of prediction uncertainty"
            )
        
        with col_stat4:
            token_count = len(response.choices[0].logprobs.content) if response.choices[0].logprobs else 0
            st.metric(
                "Token Count", 
                token_count,
                help="Number of generated tokens"
            )
        
        # Export functionality
        st.header("📤 Export Options")
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📄 Export Text", use_container_width=True):
                text_data = export_manager.export_text(response, st.session_state.original_prompt)
                st.download_button(
                    "Download Text",
                    text_data,
                    file_name=f"generated_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col_export2:
            if st.button("📊 Export Analysis", use_container_width=True):
                analysis_data = export_manager.export_analysis(response, st.session_state.generation_params)
                st.download_button(
                    "Download Analysis",
                    analysis_data,
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_export3:
            if st.button("📋 Export CSV", use_container_width=True):
                csv_data = export_manager.export_csv(response)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"token_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Detailed token analysis in expandable sections
        with st.expander("🔍 Token-by-Token Analysis"):
            tokens = response.choices[0].logprobs.content
            
            # Create detailed token dataframe
            token_data = []
            for i, token in enumerate(tokens):
                token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                probability = exp(token.logprob) * 100
                
                # Get alternative tokens
                alternatives = []
                if hasattr(token, 'top_logprobs') and token.top_logprobs:
                    for alt in token.top_logprobs[:3]:
                        alt_str = bytes(alt.bytes).decode("utf-8", errors="replace")
                        alt_prob = exp(alt.logprob) * 100
                        alternatives.append(f"{repr(alt_str)} ({alt_prob:.1f}%)")
                
                token_data.append({
                    "Position": i + 1,
                    "Token": repr(token_str),
                    "Raw Token": token_str,
                    "Logprob": round(token.logprob, 4),
                    "Probability (%)": round(probability, 2),
                    "Confidence": "High" if probability > 50 else "Medium" if probability > 20 else "Low",
                    "Alternatives": ", ".join(alternatives) if alternatives else "N/A",
                    "Bytes": str(token.bytes)
                })
            
            df = pd.DataFrame(token_data)
            st.dataframe(df, use_container_width=True, height=400)
        
        # Alternative tokens exploration
        with st.expander("🔀 Alternative Token Exploration"):
            st.markdown("Explore what other tokens the model considered at each position:")
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for i, token in enumerate(response.choices[0].logprobs.content):
                    if hasattr(token, 'top_logprobs') and token.top_logprobs:
                        token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                        st.markdown(f"**Position {i+1}: {repr(token_str)}**")
                        
                        alt_data = []
                        for j, alt in enumerate(token.top_logprobs):
                            alt_str = bytes(alt.bytes).decode("utf-8", errors="replace")
                            alt_prob = exp(alt.logprob) * 100
                            alt_data.append({
                                "Rank": j + 1,
                                "Token": repr(alt_str),
                                "Probability (%)": round(alt_prob, 2),
                                "Logprob": round(alt.logprob, 4)
                            })
                        
                        alt_df = pd.DataFrame(alt_data)
                        st.dataframe(alt_df, use_container_width=True, hide_index=True)
                        st.divider()

    # Footer with app information
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**🤖 Enhanced OpenAI Logprobs Explorer**")
    with col_footer2:
        st.markdown("Built with Streamlit & OpenAI API")
    with col_footer3:
        if hasattr(st.session_state, 'generation_params'):
            st.markdown(f"Last generated: {st.session_state.generation_params.get('timestamp', 'N/A')[:16]}")

if __name__ == "__main__":
    main()
