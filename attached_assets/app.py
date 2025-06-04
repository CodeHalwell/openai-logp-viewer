"""
Streamlit App: OpenAI Text Generation with Logprob Highlighting
Based on the guidance.md OpenAI logprobs cookbook examples.

This app demonstrates text completion with visual highlighting based on log probabilities.
Each word is colored based on the model's confidence in that token.
"""

import streamlit as st
import os
from openai import OpenAI
import numpy as np
from math import exp
import plotly.express as px
import pandas as pd

import dotenv
# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="OpenAI Logprobs Text Generator", 
    page_icon="🤖",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from environment or user input."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # If no environment variable, check Streamlit secrets
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def get_completion_with_logprobs(client, prompt, model="gpt-4o-mini", max_tokens=100, temperature=0.7):
    """
    Get completion from OpenAI with logprobs enabled.
    
    Args:
        client: OpenAI client instance
        prompt: Input prompt text
        model: Model to use for completion
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        OpenAI completion response with logprobs
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=3  # Get top 3 alternatives for each token
        )
        return response
    except Exception as e:
            error_type = type(e).__name__
            error_details = str(e)
            
            # Log detailed error information
            st.error(f"Error calling OpenAI API: {error_type} - {error_details}")
            
            # Show more context for troubleshooting
            st.error(f"""
            Request details:
            - Model: {model}
            - Max tokens: {max_tokens}
            - Temperature: {temperature}
            - Prompt length: {len(prompt)} characters
            
            Try checking:
            - API key validity and quota
            - Network connectivity
            - Model availability
            """)
            
            # Print to console for server-side logs
            print(f"OpenAI API Error: {error_type} - {error_details}")
            print(f"Request parameters: model={model}, max_tokens={max_tokens}, temperature={temperature}")
            
            return None

def calculate_color_intensity(logprob, min_logprob=-10, max_logprob=0):
    """
    Calculate color intensity based on logprob value.
    Higher logprob (more confident) = more intense green
    Lower logprob (less confident) = more intense red
    
    Args:
        logprob: Log probability value
        min_logprob: Minimum expected logprob for normalization
        max_logprob: Maximum expected logprob for normalization
    
    Returns:
        RGB color tuple
    """
    # Normalize logprob to 0-1 scale
    normalized = (logprob - min_logprob) / (max_logprob - min_logprob)
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Convert to color: red (low confidence) to green (high confidence)
    if normalized < 0.5:
        # Red to yellow
        red = 255
        green = int(255 * normalized * 2)
        blue = 0
    else:
        # Yellow to green
        red = int(255 * (1 - normalized) * 2)
        green = 255
        blue = 0
    
    return f"rgb({red}, {green}, {blue})"

def create_highlighted_text(response):
    """
    Create HTML with highlighted text based on logprobs.
    
    Args:
        response: OpenAI completion response with logprobs
    
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
        
        # Calculate color based on logprob
        color = calculate_color_intensity(token.logprob, min_logprob, max_logprob)
        
        # Create styled span
        probability_percent = round(exp(token.logprob) * 100, 2)
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
            f'border-radius: 3px; color: black; font-weight: bold;" '
            f'title="Token: {repr(token_str)}, Logprob: {token.logprob:.3f}, '
            f'Probability: {probability_percent}%">{token_str}</span>'
        )
    
    return "".join(html_parts)

def create_logprob_chart(response):
    """
    Create a chart showing logprob values for each token.
    
    Args:
        response: OpenAI completion response with logprobs
    
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
            "Logprob": token.logprob,
            "Probability (%)": probability
        })
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x="Position", 
        y="Probability (%)",
        hover_data=["Token", "Logprob"],
        title="Token Probabilities",
        color="Probability (%)",
        color_continuous_scale="RdYlGn"
    )
    
    fig.update_layout(
        xaxis_title="Token Position",
        yaxis_title="Probability (%)",
        height=400
    )
    
    return fig

def calculate_perplexity(response):
    """
    Calculate perplexity of the generated text.
    
    Args:
        response: OpenAI completion response with logprobs
    
    Returns:
        Perplexity value
    """
    if not response or not response.choices[0].logprobs:
        return None
    
    tokens = response.choices[0].logprobs.content
    logprobs = [token.logprob for token in tokens]
    
    if not logprobs:
        return None
    
    # Perplexity = exp(-mean(logprobs))
    perplexity = np.exp(-np.mean(logprobs))
    return perplexity

# Main Streamlit App
def main():
    st.title("🤖 OpenAI Text Generation with Logprob Highlighting")
    st.markdown("""
    This app demonstrates OpenAI's logprobs feature for text generation. Enter the beginning of a sentence,
    and the app will complete it while showing the model's confidence in each word through color coding:
    
    - **Green**: High confidence (high probability)
    - **Yellow**: Medium confidence 
    - **Red**: Low confidence (low probability)
    
    Hover over each word to see detailed probability information.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    client = get_openai_client()
    if not client:
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key. You can also set the OPENAI_API_KEY environment variable."
        )
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            st.warning("⚠️ Please provide your OpenAI API key to use this app.")
            st.stop()
    
    # Model selection
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        help="Select the OpenAI model to use for text generation."
    )
    
    # Generation parameters
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
        min_value=10, 
        max_value=200, 
        value=50,
        help="Maximum number of tokens to generate."
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="Controls randomness. Lower values make output more focused and deterministic."
    )
    
    # Main input area
    st.header("💭 Text Generation")
    
    prompt = st.text_area(
        "Enter the beginning of your sentence:",
        value="The future of artificial intelligence",
        height=100,
        help="Start typing a sentence and the AI will complete it."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_button = st.button("🚀 Generate", type="primary")
    
    if generate_button and prompt.strip():
        with st.spinner("Generating text..."):
            response = get_completion_with_logprobs(
                client, 
                f"Complete this sentence: {prompt}",
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response:
                # Store response in session state for analysis
                st.session_state.response = response
                st.session_state.original_prompt = prompt
                
                # Display results
                st.header("📝 Generated Text")
                
                # Get the completed text
                completed_text = response.choices[0].message.content
                
                # Show original prompt + completion
                st.markdown("**Your input:**")
                st.markdown(f"*{prompt}*")
                
                st.markdown("**AI Completion:**")
                st.markdown(f"*{completed_text}*")
                
                # Show highlighted version
                st.markdown("**Highlighted by Confidence:**")
                highlighted_html = create_highlighted_text(response)
                st.markdown(highlighted_html, unsafe_allow_html=True)
                
                # Calculate and display perplexity
                perplexity = calculate_perplexity(response)
                if perplexity:
                    st.metric(
                        "Perplexity", 
                        f"{perplexity:.2f}",
                        help="Lower perplexity indicates higher model confidence in the overall generation."
                    )
    
    # Analysis section
    if hasattr(st.session_state, 'response') and st.session_state.response:
        st.header("📊 Detailed Analysis")
        
        # Token probability chart
        fig = create_logprob_chart(st.session_state.response)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Token details table
        with st.expander("🔍 Token Details"):
            tokens = st.session_state.response.choices[0].logprobs.content
            
            token_data = []
            for i, token in enumerate(tokens):
                token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                probability = exp(token.logprob) * 100
                
                token_data.append({
                    "Position": i + 1,
                    "Token": repr(token_str),
                    "Raw Token": token_str,
                    "Logprob": round(token.logprob, 4),
                    "Probability (%)": round(probability, 2),
                    "Bytes": token.bytes
                })
            
            df = pd.DataFrame(token_data)
            st.dataframe(df, use_container_width=True)
        
        # Alternative tokens
        with st.expander("🔀 Alternative Token Choices"):
            st.markdown("See what other tokens the model considered at each position:")
            
            for i, token in enumerate(tokens):
                if hasattr(token, 'top_logprobs') and token.top_logprobs:
                    st.markdown(f"**Position {i+1}:**")
                    
                    alt_data = []
                    for alt_token in token.top_logprobs:
                        alt_token_str = bytes(alt_token.bytes).decode("utf-8", errors="replace")
                        alt_probability = exp(alt_token.logprob) * 100
                        
                        alt_data.append({
                            "Token": repr(alt_token_str),
                            "Logprob": round(alt_token.logprob, 4),
                            "Probability (%)": round(alt_probability, 2)
                        })
                    
                    alt_df = pd.DataFrame(alt_data)
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)
    
    # Information section
    with st.expander("ℹ️ About Logprobs"):
        st.markdown("""
        **Log Probabilities (logprobs)** provide insight into the model's confidence in each generated token:
        
        - **Logprob values** range from negative infinity to 0.0
        - **0.0** corresponds to 100% probability (maximum confidence)
        - **More negative values** indicate lower probability (less confidence)
        - **Linear probability** is calculated as `exp(logprob) × 100%`
        
        **Perplexity** measures the model's uncertainty:
        - **Lower perplexity** = higher confidence in the overall generation
        - **Higher perplexity** = more uncertainty in the generation
        - Calculated as `exp(-mean(logprobs))`
        
        This visualization helps you understand:
        - Which words the model was confident about
        - Where the model had uncertainty
        - Alternative words the model considered
        """)

if __name__ == "__main__":
    main()
