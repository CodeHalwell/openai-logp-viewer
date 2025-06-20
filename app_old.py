"""
Enhanced Streamlit App: OpenAI Text Generation with Logprob Highlighting
Optimized version with improved performance, UI/UX, and additional features.

This app demonstrates text completion with visual highlighting based on log probabilities.
Each word is colored based on the model's confidence in that token.
"""

import streamlit as st
import os
import html
import hashlib
import secrets
from openai import OpenAI
import numpy as np
from math import exp
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import time
import re

import dotenv
from utils.cache_manager import CacheManager
from utils.color_schemes import ColorSchemeManager
from utils.export_manager import ExportManager
from utils.statistics import StatisticsCalculator
from utils.rate_limiter import RateLimiter

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure Streamlit page with security headers
st.set_page_config(
    page_title="OpenAI Logprobs Text Generator", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add security headers and CSP via HTML
st.markdown("""
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.openai.com;">
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<meta http-equiv="X-Frame-Options" content="DENY">
<meta http-equiv="X-XSS-Protection" content="1; mode=block">
<style>
    /* Security: Prevent content injection */
    iframe { display: none !important; }
    script[src] { display: none !important; }
    object, embed, applet { display: none !important; }
    /* Allow only inline styles and scripts from streamlit */
</style>
""", unsafe_allow_html=True)

# Initialize utility classes
cache_manager = CacheManager()
color_manager = ColorSchemeManager()
export_manager = ExportManager()
stats_calculator = StatisticsCalculator()
rate_limiter = RateLimiter()

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
        display: inline;
        margin: 0;
        padding: 1px 2px;
        border-radius: 2px;
        font-weight: normal;
        white-space: normal;
        transition: all 0.1s ease;
        font-size: inherit;
        line-height: inherit;
        word-break: normal;
    }
    .token-highlight:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    .analysis-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        max-width: 100%;
        box-sizing: border-box;
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

# OpenAI client initialization (no caching since API key is user-provided)
def get_openai_client(api_key: str):
    """Initialize OpenAI client with user-provided API key."""
    try:
        client = OpenAI(api_key=api_key)
        # Test the client with a simple request
        client.models.list()
        return client
    except Exception as e:
        # Don't log the actual API key or expose detailed error info
        st.error("Failed to initialize OpenAI client. Please contact your administrator.")
        return None

def create_secure_api_key_hash(api_key: str) -> str:
    """Create a cryptographically secure hash of the API key for caching."""
    # Use a secure hash function with salt
    salt = secrets.token_bytes(32)
    hash_obj = hashlib.pbkdf2_hmac('sha256', api_key.encode('utf-8'), salt, 100000)
    # Return a hex representation of salt + hash for uniqueness
    return (salt + hash_obj).hex()

def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format and basic structure."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI API keys should start with sk- and be at least 40 characters
    if not api_key.startswith("sk-"):
        return False
    
    if len(api_key) < 40 or len(api_key) > 200:
        return False
    
    # Check for basic character set (alphanumeric + some special chars)
    if not re.match(r'^sk-[A-Za-z0-9\-_]+$', api_key):
        return False
    
    return True

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user prompt to prevent potential security issues."""
    if not prompt or not isinstance(prompt, str):
        return ""
    
    # Limit prompt length for security
    max_length = 10000
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    
    # Remove dangerous characters and control sequences
    import string
    # Remove null bytes and other control characters
    prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in ['\n', '\t'])
    
    # Remove potential script injection patterns
    dangerous_patterns = [
        '<script', '</script>', '<iframe', '</iframe>',
        'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror=',
        '${', '#{', '<%', '%>', '{{', '}}'
    ]
    
    prompt_lower = prompt.lower()
    for pattern in dangerous_patterns:
        if pattern in prompt_lower:
            # Replace with safe equivalent
            prompt = prompt.replace(pattern, '[FILTERED]')
            prompt = prompt.replace(pattern.upper(), '[FILTERED]')
            prompt = prompt.replace(pattern.capitalize(), '[FILTERED]')
    
    return prompt.strip()

def cleanup_session_security():
    """Clean up security-sensitive data from session state."""
    import secrets
    security_keys = ['api_key', 'api_key_hash', 'client_instance']
    for key in security_keys:
        if key in st.session_state:
            # Overwrite with random data multiple times for secure deletion
            for _ in range(3):
                st.session_state[key] = secrets.token_urlsafe(32)
            del st.session_state[key]

@st.cache_data(ttl=3600, show_spinner=False)
def get_completion_with_logprobs(api_key_hash, prompt, model="gpt-4o", max_tokens=100, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=None):
    """
    Get completion from OpenAI with logprobs enabled (cached version).
    
    Args:
        api_key_hash: Secure hash of API key for caching (not the actual key)
        prompt: Input prompt text (sanitized)
        model: Model to use for completion
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter
        seed: Seed for deterministic generation
    
    Returns:
        Tuple of (OpenAI completion response with logprobs, error_message)
    """
    try:
        # Get the actual API key from session state (never cache the actual key)
        api_key = st.session_state.get("api_key")
        if not api_key:
            return None, {"error_type": "APIKeyError", "error_details": "No API key available"}
        
        # Validate API key format again for security
        if not validate_api_key_format(api_key):
            return None, {"error_type": "APIKeyError", "error_details": "Invalid API key format"}
        
        # Sanitize prompt before sending
        prompt = sanitize_prompt(prompt)
        if not prompt:
            return None, {"error_type": "InputError", "error_details": "Invalid or empty prompt"}
        
        client = OpenAI(api_key=api_key)
        
        # Validate parameters
        max_tokens = max(1, min(max_tokens, 4000))  # Reasonable limits
        temperature = max(0.0, min(temperature, 2.0))
        top_p = max(0.0, min(top_p, 1.0))
        frequency_penalty = max(-2.0, min(frequency_penalty, 2.0))
        presence_penalty = max(-2.0, min(presence_penalty, 2.0))
        
        # Build API call parameters for text completion style
        # Use a system message to instruct the model to continue the text
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Continue the following text naturally. Do not respond conversationally, just continue writing where the text left off."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "logprobs": True,
            "top_logprobs": 5  # Get top 5 alternatives for each token
        }
        
        # Add seed if provided and valid
        if seed is not None:
            try:
                seed = max(0, min(int(seed), 2147483647))  # Valid 32-bit range
                api_params["seed"] = seed
            except (ValueError, TypeError):
                pass  # Ignore invalid seed values
        
        response = client.chat.completions.create(**api_params)
        return response, None
        
    except Exception as e:
        error_type = type(e).__name__
        
        # Sanitized error handling - never expose API keys or sensitive details
        if "authentication" in str(e).lower() or "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
            error_details = "Authentication failed. Please contact your administrator."
        elif "rate_limit" in str(e).lower() or "429" in str(e):
            error_details = "Rate limit exceeded. Please wait and try again."
        elif "quota" in str(e).lower() or "billing" in str(e).lower():
            error_details = "API quota exceeded. Please check your OpenAI billing."
        elif "timeout" in str(e).lower():
            error_details = "Request timed out. Please try again."
        elif "model" in str(e).lower() and "not found" in str(e).lower():
            error_details = "Selected model not available. Please choose a different model."
        else:
            error_details = "API request failed. Please check your connection and try again."
        
        # Return completely sanitized error information
        detailed_error = {
            "error_type": "APIError",
            "error_details": error_details
        }
        
        return None, detailed_error

def create_top_choice_analysis(response):
    """
    Create HTML showing whether each token was the highest probability choice.
    
    Args:
        response: OpenAI completion response with logprobs
    
    Returns:
        HTML string with top choice analysis
    """
    if not response or not response.choices[0].logprobs:
        return "No logprobs available"
    
    tokens = response.choices[0].logprobs.content
    html_parts = []
    
    for token in tokens:
        # Decode token bytes to string
        try:
            token_text = bytes(token.bytes).decode('utf-8', errors='replace')
        except (AttributeError, TypeError):
            token_text = str(token.token) if hasattr(token, 'token') else str(token)
        
        # Escape HTML characters
        token_text = html.escape(token_text)
        
        # Check if this was the top choice by examining top_logprobs
        was_top_choice = True
        if hasattr(token, 'top_logprobs') and token.top_logprobs:
            # The first entry in top_logprobs should be the selected token
            # If there are alternatives with higher probabilities, this wasn't the top choice
            current_logprob = token.logprob
            for alt_token in token.top_logprobs:
                if alt_token.logprob > current_logprob:
                    was_top_choice = False
                    break
        
        # Calculate probability percentage
        from math import exp
        probability = exp(token.logprob) * 100
        
        # Create styling based on whether it was top choice
        if was_top_choice:
            # Green background for top choice
            style = f"background-color: rgba(144, 238, 144, 0.3); padding: 1px 3px; border-radius: 3px; border: 1px solid rgba(0, 128, 0, 0.5);"
            title = f"TOP CHOICE: {probability:.1f}% probability"
        else:
            # Orange background for not top choice
            style = f"background-color: rgba(255, 165, 0, 0.3); padding: 1px 3px; border-radius: 3px; border: 1px solid rgba(255, 140, 0, 0.5);"
            title = f"NOT TOP CHOICE: {probability:.1f}% probability"
        
        html_parts.append(f'<span class="token-highlight" style="{style}" title="{title}">{token_text}</span>')
    
    return ''.join(html_parts)

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
                         for alt in token.top_logprobs[:3] if alt.bytes is not None and alt.bytes != token.bytes]
            if alt_tokens:
                alternatives = f", Alternatives: {', '.join(repr(alt) for alt in alt_tokens)}"
        
        # Properly escape HTML special characters in token string and title
        escaped_token = token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        escaped_repr = repr(token_str).replace('"', "&quot;")
        escaped_alternatives = alternatives.replace('"', "&quot;")
        
        html_parts.append(
            f'<span class="token-highlight" style="background-color: {color}; color: black;" '
            f'title="Token: {escaped_repr}, Logprob: {token.logprob:.3f}, '
            f'Probability: {probability_percent}%{escaped_alternatives}">{escaped_token}</span>'
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
    
    # Initialize fig variable to prevent unbound variable error
    fig = None
    
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
    else:
        # Default fallback to bar chart
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
    """CodeHelwell Homepage - Welcome to AI Text Generation Platform."""
    
    # Initialize session cleanup on first load for security
    if 'session_initialized' not in st.session_state:
        cleanup_session_security()
        st.session_state.session_initialized = True
    
    # Hero section with logo and branding
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    # Display CodeHelwell logo prominently
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        try:
            st.image("assets/logo.png", width=400)
        except:
            st.markdown("# **CodeHelwell**")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>Welcome to AI Text Generation Platform</h1>
        <h3>Experience the Future of AI-Powered Text Generation</h3>
        <p style='font-size: 1.2em; color: #666;'>
            Visualize AI model confidence in real-time with advanced logprob analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 **Token-Level Analysis**
        See exactly how confident the AI is in each generated word with color-coded visualization
        """)
    
    with col2:
        st.markdown("""
        ### 📊 **Statistical Insights**
        Detailed metrics including perplexity, entropy, and confidence distributions
        """)
    
    with col3:
        st.markdown("""
        ### 🔒 **Enterprise Security**
        Built with robust security measures, rate limiting, and input validation
        """)
    
    st.markdown("---")
    
    # Quick start section
    st.markdown("## Quick Start")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Ready to explore AI text generation?**
        
        1. Navigate to the **Text Generator** page using the sidebar
        2. Enter your prompt or choose from examples
        3. Adjust generation parameters to your preference
        4. Generate text and explore the confidence analysis
        5. Export your results in multiple formats
        
        The application provides real-time visualization of how confident the AI model is 
        in each token it generates, giving you unprecedented insight into the AI's 
        decision-making process.
        """)
    
    with col2:
        st.info("""
        **Navigation**
        
        📝 **Text Generator** - Main application
        
        ℹ️ **About** - Learn about CodeHelwell
        """)
    
    # Technology showcase
    st.markdown("---")
    st.markdown("## Technology Showcase")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("OpenAI Models", "6+", help="GPT-4o, GPT-4, GPT-3.5 and variants")
    
    with col2:
        st.metric("Security Layers", "5+", help="Rate limiting, input validation, CSP, error handling")
    
    with col3:
        st.metric("Visualization Types", "4+", help="Color schemes, charts, statistics, exports")
    
    with col4:
        st.metric("Export Formats", "3+", help="Text, JSON, CSV with detailed metadata")
    
    # Security and privacy notice
    st.markdown("---")
    st.success("🔒 **Secure & Private**: This application uses enterprise-grade security with rate limiting and secure API management.")
    
    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem 0;'>
        Built with precision by <strong>CodeHelwell</strong><br>
        <em>Transforming ideas into intelligent solutions</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API connection status
        st.info("🔑 OpenAI API Connected")
        
        # Connection notice
        st.success("🔒 **Secure Connection**: Using managed OpenAI API with built-in rate limiting and security protections.")
        
        with st.expander("🛡️ Security & Privacy Details"):
            st.markdown("""
            **Secure API Management:**
            - ✅ API keys stored securely in environment variables
            - ✅ No user credentials required or stored
            - ✅ All requests authenticated through secure server connection
            - ✅ Rate limiting prevents abuse and ensures fair access
            - ✅ Input validation and sanitization on all prompts
            - ✅ Session data automatically cleaned on reload
            
            **This app uses server-side API management** - all requests go through secure, rate-limited channels.
            
            **Additional Security Measures:**
            - Input validation and sanitization
            - Secure error handling (no sensitive data exposure)
            - Rate limiting protection
            - Automatic session cleanup
            """)
        
        # Add session cleanup on app start
        if st.button("🔄 Clear Session & Start Fresh", help="Clear all session data for security"):
            cleanup_session_security()
            st.cache_data.clear()
            st.rerun()
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("❌ OpenAI API key not found in environment variables.")
            st.info("The OPENAI_API_KEY environment variable needs to be set. Please check your secrets configuration.")
            st.stop()
        else:
            # Enhanced format validation for API key
            if not validate_api_key_format(api_key):
                st.error("❌ Invalid API key format. OpenAI keys should start with 'sk-' and be 40-200 characters.")
                st.stop()
            
            # Test the API key and store in session state with security measures
            try:
                test_client = OpenAI(api_key=api_key)
                # Quick test to validate the key
                test_client.models.list()
                st.success("✅ OpenAI API Connected Successfully")
                
                # Store API key securely in session state
                st.session_state.api_key = api_key
                
                # Create secure hash for caching (never cache the actual key)
                st.session_state.api_key_hash = create_secure_api_key_hash(api_key)
                
            except Exception as e:
                st.error("❌ Failed to connect to OpenAI API.")
                # Log error securely without exposing details to user
                import logging
                logging.error(f"OpenAI API connection failed: {type(e).__name__}")
                st.error("Connection error occurred. Please try again or contact administrator if the issue persists.")
                with st.expander("Common Solutions"):
                    st.markdown("""
                    **If you see 'Incorrect API key':**
                    - The API service may be temporarily unavailable
                    - Contact administrator if the issue persists
                    
                    **If you see 'quota exceeded':**
                    - The shared API quota has been reached
                    - Try again later or contact administrator
                    
                    **If you see 'network' or 'connection' errors:**
                    - Check your internet connection
                    - OpenAI service may be temporarily unavailable
                    """)
                st.stop()
        
        st.divider()
        
        # Model selection with actual model names
        st.subheader("🎯 Model Selection")
        models = [
            "gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106"
        ]
        
        model = st.selectbox(
            "Model",
            models,
            index=0,
            help="Select the OpenAI model for text generation."
        )
        
        st.divider()
        
        # Generation parameters with manual controls
        st.subheader("🎛️ Generation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                help="Controls creativity. Lower = more focused, Higher = more creative"
            )
            
            top_p = st.slider(
                "Top P", 
                min_value=0.0, 
                max_value=1.0, 
                value=1.0, 
                step=0.05,
                help="Nucleus sampling. Lower values focus on more probable tokens"
            )
            
            max_tokens = st.slider(
                "Max Tokens", 
                min_value=10, 
                max_value=500, 
                value=100,
                help="Maximum number of tokens to generate"
            )
        
        with col2:
            frequency_penalty = st.slider(
                "Frequency Penalty", 
                min_value=-2.0, 
                max_value=2.0, 
                value=0.0, 
                step=0.1,
                help="Reduces repetition. Positive values decrease likelihood of repeated tokens"
            )
            
            presence_penalty = st.slider(
                "Presence Penalty", 
                min_value=-2.0, 
                max_value=2.0, 
                value=0.0, 
                step=0.1,
                help="Encourages new topics. Positive values increase likelihood of new tokens"
            )
            
            seed = st.number_input(
                "Seed (optional)", 
                min_value=0, 
                max_value=2147483647,  # Max 32-bit integer for better security
                value=None,
                help="Set for deterministic outputs. Leave empty for random generation"
            )
        
        # Show parameter preset suggestions as info
        if temperature <= 0.4:
            st.info("💼 Academic/Technical writing mode")
        elif temperature <= 0.7:
            st.info("💬 Balanced conversation mode")
        elif temperature <= 1.0:
            st.info("🎨 Creative writing mode")
        else:
            st.info("🌈 Highly creative mode")
        
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
        
        # Rate limiting display
        st.divider()
        rate_limiter.display_rate_limit_info()
        
        # Security information
        with st.expander("🔒 Security & Fair Use Policy"):
            st.markdown("""
            **This app implements the following protections:**
            - **Rate Limiting**: 10 requests per minute maximum
            - **Token Limits**: 2,000 tokens per minute, 50,000 per day
            - **Request Size Limits**: Maximum 500 tokens per request
            - **Secure API Key Handling**: Keys stored securely in environment
            - **Input Validation**: All prompts are sanitized for security
            
            **Fair Use Guidelines:**
            - This app uses free OpenAI tokens provided for demonstration
            - Please use responsibly and avoid excessive requests
            - Rate limits help ensure fair access for all users
            - Large or commercial usage should use your own API key
            """)
        
        # Cache management with security considerations
        st.divider()
        st.subheader("💾 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache", help="Clear all cached responses (recommended for security)"):
                st.cache_data.clear()
                cache_manager.clear_cache()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("Clear Session", help="Clear all session data including API key"):
                cleanup_session_security()
                st.cache_data.clear()
                st.success("Session cleared!")
                st.rerun()
        
        # Show cache info without sensitive data
        cache_info = cache_manager.get_cache_info()
        st.metric("Cached Items", cache_info.get("count", 0), help="Number of cached API responses")
    
    # Main content area with improved layout
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
        help="Start typing and the AI will complete your text. Prompts are validated for security.",
        max_chars=10000  # Security limit
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
    
    # Generation and results
    if generate_button and prompt.strip():
        # Sanitize prompt before processing
        sanitized_prompt = sanitize_prompt(prompt)
        if not sanitized_prompt:
            st.error("❌ Invalid prompt. Please enter valid text.")
            st.stop()
        
        # Check rate limits before making API call
        rate_check = rate_limiter.check_and_record_request(sanitized_prompt, max_tokens)
        
        if not rate_check["allowed"]:
            if rate_check["reason"] == "request_rate":
                st.error(f"🚨 Too many requests. Please wait {rate_check['reset_in']:.0f} seconds before trying again.")
                st.info(f"Current: {rate_check['current']}/{rate_check['limit']} requests per minute")
            elif rate_check["reason"] == "token_rate_minute":
                st.error(f"🚨 Token limit exceeded. Please wait 60 seconds before trying again.")
                st.info(f"Current: {rate_check['current']}/{rate_check['limit']} tokens per minute")
            elif rate_check["reason"] == "token_rate_daily":
                st.error(f"🚨 Daily token limit reached. Please try again tomorrow.")
                st.info(f"Current: {rate_check['current']}/{rate_check['limit']} tokens per day")
            elif rate_check["reason"] == "token_per_request":
                st.error(f"🚨 Request too large. Maximum {rate_check['limit']} tokens per request.")
                st.info(f"Your request: {rate_check['requested']} tokens (reduce max_tokens or prompt length)")
            st.stop()
        
        with st.spinner("🤖 Generating text..."):
            # Get secure API key hash for caching (never cache the actual key)
            api_key_hash = st.session_state.get("api_key_hash")
            if not api_key_hash:
                st.error("❌ Session expired. Please refresh the page.")
                st.stop()
            
            # Progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            response, error = get_completion_with_logprobs(
                api_key_hash,
                sanitized_prompt,  # Use sanitized prompt
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed
            )
            
            progress_bar.empty()
            
            if response:
                # Record successful API usage for rate limiting
                actual_tokens_used = rate_check.get("estimated_tokens", max_tokens)
                rate_limiter.record_request(actual_tokens_used)
                
                # Store response in session state
                st.session_state.response = response
                st.session_state.original_prompt = sanitized_prompt  # Store sanitized version
                st.session_state.generation_params = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.success("✅ Text generated successfully!")
                
                # Show token usage information
                with st.expander("📊 Token Usage"):
                    st.info(f"Estimated tokens used: {actual_tokens_used}")
                    st.info(f"Input tokens: ~{rate_check.get('input_tokens', 0)}")
                    st.info(f"Output tokens: ~{actual_tokens_used - rate_check.get('input_tokens', 0)}")
                
            elif error:
                st.error(f"❌ Generation failed: {error['error_details']}")
                with st.expander("🔍 Troubleshooting"):
                    st.markdown("""
                    **Common solutions:**
                    - The API service may be temporarily unavailable
                    - Try reducing max_tokens or simplifying your prompt
                    - Wait a moment and try again if rate limited
                    - Check your internet connection
                    - Contact administrator if issues persist
                    """)
                st.stop()
    
    # Display results with enhanced UI
    if hasattr(st.session_state, 'response') and st.session_state.response:
        response = st.session_state.response
        
        st.header("📝 Generated Text")
        
        # Display text in tabs for better organization
        tab1, tab2, tab3 = st.tabs(["🎨 Highlighted View", "📄 Plain Text", "🔗 Combined"])
        
        with tab1:
            st.markdown("**Original Prompt:**")
            original_prompt = st.session_state.original_prompt
            st.markdown(f'<div style="background-color: #f0f0f0; padding: 8px; border-radius: 4px; margin-bottom: 10px; font-size: 14px; color: #666;">{html.escape(original_prompt)}</div>', unsafe_allow_html=True)
            
            st.markdown("**Confidence-Based Highlighting:**")
            highlighted_html = create_enhanced_highlighted_text(response, color_scheme)
            st.markdown(f'<div class="analysis-container">{highlighted_html}</div>', unsafe_allow_html=True)
            
            st.markdown("**Top Choice Analysis:**")
            st.caption("Shows whether each token was the highest probability option available")
            top_choice_html = create_top_choice_analysis(response)
            st.markdown(f'<div class="analysis-container">{top_choice_html}</div>', unsafe_allow_html=True)
        
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
                        try:
                            if hasattr(alt, 'bytes') and alt.bytes is not None:
                                alt_str = bytes(alt.bytes).decode("utf-8", errors="replace")
                            else:
                                alt_str = str(alt.token) if hasattr(alt, 'token') else str(alt)
                            alt_prob = exp(alt.logprob) * 100
                            alternatives.append(f"{repr(alt_str)} ({alt_prob:.1f}%)")
                        except (AttributeError, TypeError):
                            continue
                
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
                            if alt.bytes is not None:
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
    # Add cleanup on app termination
    try:
        main()
    finally:
        # Clean up on exit for security
        cleanup_session_security()
