"""
Main Text Generation Page
OpenAI Logprobs Text Generator - CodeHalwell
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

import sys

sys.path.append('.')

from utils.cache_manager import CacheManager
from utils.color_schemes import ColorSchemeManager
from utils.export_manager import ExportManager
from utils.statistics import StatisticsCalculator
from utils.rate_limiter import RateLimiter

# Import functions directly - create local implementations
import openai
from openai import OpenAI
import hashlib
import secrets

# Note: st.set_page_config() is only called in main app.py for multi-page apps


# Local function implementations
def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    return bool(api_key and api_key.startswith('sk-') and len(api_key) > 40)


def sanitize_prompt(prompt: str) -> str:
    """Sanitize user prompt."""
    if not prompt or not isinstance(prompt, str):
        return ""
    # Remove potential harmful content
    sanitized = prompt.strip()
    # Basic validation
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000]
    return sanitized


def cleanup_session_security():
    """Clean up security-sensitive data from session state."""
    sensitive_keys = ['api_key', 'api_key_hash', 'response']
    for key in sensitive_keys:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except:
                pass


@st.cache_data(ttl=3600)
def get_completion_with_logprobs(api_key_hash,
                                 prompt,
                                 model="gpt-4o",
                                 max_tokens=100,
                                 temperature=0.7,
                                 frequency_penalty=0.0,
                                 presence_penalty=0.0,
                                 seed=None):
    """Get completion from OpenAI with logprobs enabled."""
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, {"error_details": "API key not configured"}

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(model=model,
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }],
                                                  max_tokens=max_tokens,
                                                  temperature=temperature,
                                                  logprobs=True,
                                                  top_logprobs=5)

        return response, None

    except Exception as e:
        return None, {"error_details": f"Generation failed: {str(e)}"}


def create_enhanced_highlighted_text(response, color_scheme="confidence"):
    """Create HTML with enhanced highlighted text based on logprobs with hover tooltips."""
    try:
        if not response or not hasattr(response, 'choices'):
            return ""

        choice = response.choices[0]
        if not hasattr(choice, 'logprobs') or not choice.logprobs:
            return ""

        tokens = choice.logprobs.content
        if not tokens:
            return ""

        html_parts = []
        
        # Find min/max logprobs for better color scaling
        logprobs = [token.logprob for token in tokens]
        min_logprob = min(logprobs) if logprobs else -10
        max_logprob = max(logprobs) if logprobs else 0
        
        for token in tokens:
            # Get token string
            if hasattr(token, 'bytes') and token.bytes is not None:
                token_str = bytes(token.bytes).decode("utf-8", errors="replace")
            else:
                token_str = str(token.token) if hasattr(token, 'token') else str(token)
            
            # Calculate color based on logprob using color scheme manager
            color = color_manager.get_color(token.logprob, min_logprob, max_logprob, color_scheme)
            
            # Create styled span with enhanced hover effects
            probability_percent = round(exp(token.logprob) * 100, 2)
            
            # Get alternative tokens if available
            alternatives = ""
            if hasattr(token, 'top_logprobs') and token.top_logprobs:
                alt_list = []
                for alt in token.top_logprobs[:3]:
                    if hasattr(alt, 'bytes') and alt.bytes is not None:
                        alt_str = bytes(alt.bytes).decode("utf-8", errors="replace")
                    else:
                        alt_str = str(alt.token) if hasattr(alt, 'token') else str(alt)
                    
                    # Skip if it's the same as the selected token
                    if alt_str != token_str:
                        alt_prob = round(exp(alt.logprob) * 100, 2)
                        alt_list.append(f"{repr(alt_str)} ({alt_prob}%)")
                
                if alt_list:
                    alternatives = f", Alternatives: {', '.join(alt_list)}"
            
            # Properly escape HTML special characters in token string and title
            escaped_token = html.escape(token_str)
            escaped_repr = html.escape(repr(token_str))
            escaped_alternatives = html.escape(alternatives)
            
            html_parts.append(
                f'<span class="token-highlight" style="background-color: {color}; color: black; padding: 2px 4px; margin: 1px; border-radius: 3px; cursor: help;" '
                f'title="Token: {escaped_repr}, Logprob: {token.logprob:.4f}, '
                f'Probability: {probability_percent}%{escaped_alternatives}">{escaped_token}</span>'
            )
        
        return ''.join(html_parts)

    except Exception as e:
        return ""


def create_enhanced_logprob_chart(response, chart_type="bar"):
    """Create enhanced charts showing logprob values."""
    try:
        if not response or not hasattr(response, 'choices'):
            return None

        choice = response.choices[0]
        if not hasattr(choice, 'logprobs') or not choice.logprobs:
            return None

        logprobs = choice.logprobs.content
        if not logprobs:
            return None

        tokens = [
            token.token for token in logprobs if token.logprob is not None
        ]
        probabilities = [
            exp(token.logprob) * 100 for token in logprobs
            if token.logprob is not None
        ]

        if chart_type == "bar":
            fig = px.bar(
                x=tokens[:20],  # Limit to first 20 tokens
                y=probabilities[:20],
                title="Token Confidence Levels",
                labels={
                    "x": "Tokens",
                    "y": "Confidence (%)"
                })
        else:
            fig = px.line(x=list(range(len(tokens[:20]))),
                          y=probabilities[:20],
                          title="Token Confidence Progression",
                          labels={
                              "x": "Token Position",
                              "y": "Confidence (%)"
                          })

        return fig

    except Exception as e:
        return None


def create_top_choice_analysis(response):
    """Create HTML showing top choice analysis in paragraph style with color coding."""
    try:
        if not response or not hasattr(response, 'choices'):
            return ""

        choice = response.choices[0]
        if not hasattr(choice, 'logprobs') or not choice.logprobs:
            return ""

        logprobs = choice.logprobs.content
        if not logprobs:
            return ""

        html_parts = []
        for token in logprobs:
            if token.logprob is not None:
                # Get token string
                if hasattr(token, 'bytes') and token.bytes is not None:
                    token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                else:
                    token_str = str(token.token) if hasattr(token, 'token') else str(token)
                
                # Check if this was the top choice by examining top_logprobs
                was_top_choice = True
                if hasattr(token, 'top_logprobs') and token.top_logprobs:
                    current_logprob = token.logprob
                    for alt_token in token.top_logprobs:
                        if alt_token.logprob > current_logprob:
                            was_top_choice = False
                            break
                
                # Calculate probability percentage
                probability = exp(token.logprob) * 100
                
                # Create styling based on whether it was top choice
                if was_top_choice:
                    # Green background for top choice
                    style = "background-color: rgba(144, 238, 144, 0.6); padding: 2px 4px; margin: 1px; border-radius: 3px;"
                else:
                    # Orange background for not top choice
                    style = "background-color: rgba(255, 165, 0, 0.6); padding: 2px 4px; margin: 1px; border-radius: 3px;"
                
                # Escape HTML characters
                escaped_token = html.escape(token_str)
                
                html_parts.append(f'<span style="{style}">{escaped_token}</span>')

        return ''.join(html_parts)

    except Exception as e:
        return ""


# Initialize utility classes
cache_manager = CacheManager()
color_manager = ColorSchemeManager()
export_manager = ExportManager()
stats_calculator = StatisticsCalculator()
rate_limiter = RateLimiter()


def main():
    """Main text generation page."""

    # Initialize session cleanup on first load for security
    if 'session_initialized' not in st.session_state:
        cleanup_session_security()
        st.session_state.session_initialized = True

    st.title("🤖 AI Text Generation with Confidence Analysis")

    # Show loading indicator on first page load
    if 'page_loaded' not in st.session_state:
        # Create loading placeholder
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                try:
                    st.image("assets/logo.png", width=120)
                except:
                    st.markdown("**CodeHalwell**")
            st.markdown("""
                <div style='text-align: center; margin: 20px 0;'>
                    <div style='display: inline-block; animation: spin 2s linear infinite;'>
                        🔄
                    </div>
                    <p style='color: #666; margin-top: 10px;'>Confirming connection to OpenAI with API Key...</p>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
            """,
                        unsafe_allow_html=True)

        # Simulate loading time
        time.sleep(2)
        loading_placeholder.empty()
        st.session_state.page_loaded = True
        st.rerun()

    # Check API connection
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not validate_api_key_format(api_key):
        st.error("API configuration issue. Please contact administrator.")
        return

    # Store API key for session
    if "api_key" not in st.session_state:
        st.session_state.api_key = api_key
        st.session_state.api_key_hash = hashlib.sha256(
            api_key.encode()).hexdigest()[:16]

    # Sidebar configuration
    with st.sidebar:
        try:
            st.image("assets/logo.png", use_container_width=True)
        except:
            st.markdown("**CodeHalwell**")

        st.divider()

        st.header("⚙️ Configuration")
        st.success("🔗 Connected to OpenAI API")

        # Rate limiting display
        rate_limiter.display_rate_limit_info()

        st.divider()

        # Model selection
        st.subheader("🎯 Model Selection")
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
        model = st.selectbox("Model", models, index=0)

        # Generation parameters
        st.subheader("🎛️ Parameters")
        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 10, 500, 100)

        with col2:
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0,
                                          0.1)
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0,
                                         0.1)

        # Visualization options
        st.subheader("🎨 Visualization")
        color_scheme = st.selectbox("Color Scheme",
                                    ["confidence", "rainbow", "heat", "ocean"])
        chart_type = st.selectbox("Chart Type", ["bar", "line", "heatmap"])

    # Main content
    st.header("💭 Text Generation")

    # Prompt examples
    prompt_examples = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The key to solving climate change",
        "In the year 2050, technology will",
        "The most important lesson I learned"
    ]

    selected_example = st.selectbox("Quick Examples (optional)",
                                    [""] + prompt_examples)

    prompt = st.text_area("Enter your prompt:",
                          value=selected_example if selected_example else
                          "The future of artificial intelligence",
                          height=120,
                          max_chars=10000)

    # Generation controls
    col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
    with col_gen1:
        generate_button = st.button("🚀 Generate Text",
                                    type="primary",
                                    use_container_width=True)
    with col_gen2:
        if st.button("🎲 Random"):
            import random
            prompt = random.choice(prompt_examples)
            st.rerun()
    with col_gen3:
        if st.button("📋 Clear"):
            prompt = ""
            st.rerun()

    # Generation and results
    if generate_button and prompt.strip():
        sanitized_prompt = sanitize_prompt(prompt)
        if not sanitized_prompt:
            st.error("Invalid prompt. Please enter valid text.")
            return

        # Check rate limits
        rate_check = rate_limiter.check_and_record_request(
            sanitized_prompt, max_tokens)

        if not rate_check["allowed"]:
            if rate_check["reason"] == "request_rate":
                st.error(
                    f"Too many requests. Please wait {rate_check['reset_in']:.0f} seconds."
                )
            elif rate_check["reason"] == "token_rate_minute":
                st.error("Token limit exceeded. Please wait 60 seconds.")
            elif rate_check["reason"] == "token_rate_daily":
                st.error(
                    "Daily token limit reached. Please try again tomorrow.")
            elif rate_check["reason"] == "token_per_request":
                st.error(
                    f"Request too large. Maximum {rate_check['limit']} tokens per request."
                )
            return

        # Show loading indicator during generation
        generation_placeholder = st.empty()
        with generation_placeholder.container():
            st.markdown("""
                <div style='text-align: center; margin: 20px 0;'>
                    <div style='display: inline-block; animation: spin 2s linear infinite;'>
                        🔄
                    </div>
                    <p style='color: #666; margin-top: 10px;'>Generating text with OpenAI...</p>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
            """,
                        unsafe_allow_html=True)

        response, error = get_completion_with_logprobs(
            st.session_state.api_key_hash,
            sanitized_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty)

        generation_placeholder.empty()

        if response:
            # Record usage
            actual_tokens_used = rate_check.get("estimated_tokens", max_tokens)
            rate_limiter.record_request(actual_tokens_used)

            # Store response
            st.session_state.response = response
            st.session_state.original_prompt = sanitized_prompt

            st.success("Text generated successfully!")

            with st.expander("📊 Token Usage"):
                st.info(f"Estimated tokens used: {actual_tokens_used}")

        elif error:
            st.error(
                f"Generation failed: {error.get('error_details', 'Unknown error')}"
            )

    # Display results
    if hasattr(st.session_state, 'response') and st.session_state.response:
        response = st.session_state.response
        original_prompt = getattr(st.session_state, 'original_prompt', '')

        st.divider()
        st.header("📊 Results & Analysis")

        # Enhanced highlighted text with CSS styling
        highlighted_html = create_enhanced_highlighted_text(
            response, color_scheme)
        if highlighted_html:
            st.subheader("🎨 Highlighted Text")
            # Add enhanced CSS for hover effects
            st.markdown("""
            <style>
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
                    transform: scale(1.05);
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
            </style>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="analysis-container">{highlighted_html}</div>', unsafe_allow_html=True)

        # Top Choice Analysis
        st.subheader("🎯 Top Choice Analysis")
        top_choice_html = create_top_choice_analysis(response)
        if top_choice_html:
            st.markdown(f'<div class="analysis-container">{top_choice_html}</div>', unsafe_allow_html=True)

        # Probability Chart
        st.subheader("📈 Probability Chart")
        fig = create_enhanced_logprob_chart(response, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Token details table
        st.subheader("🔍 Token Details")
        st.write("Detailed information for each token:")
        
        # Create token details data
        token_data = []
        choice = response.choices[0]
        if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
            logprobs = choice.logprobs.content
            if logprobs:
                for i, token in enumerate(logprobs):
                    if token.logprob is not None:
                        confidence = min(100, max(0, exp(token.logprob) * 100))
                        
                        # Get top alternatives
                        alternatives = []
                        if hasattr(token, 'top_logprobs') and token.top_logprobs:
                            for alt in token.top_logprobs[:3]:
                                alt_confidence = exp(alt.logprob) * 100
                                alternatives.append(f"{alt.token} ({alt_confidence:.1f}%)")
                        
                        token_data.append({
                            "Position": i + 1,
                            "Token": f'"{token.token}"',
                            "Logprob": f"{token.logprob:.4f}",
                            "Confidence": f"{confidence:.2f}%",
                            "Top Alternatives": " | ".join(alternatives) if alternatives else "N/A"
                        })
        
        if token_data:
            # Display as interactive dataframe
            df = pd.DataFrame(token_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Position": st.column_config.NumberColumn("Pos", width="small"),
                    "Token": st.column_config.TextColumn("Token", width="small"),
                    "Logprob": st.column_config.TextColumn("Logprob", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Top Alternatives": st.column_config.TextColumn("Top Alternatives", width="large")
                }
            )

        # Statistics
        st.subheader("📊 Statistical Analysis")
        stats = stats_calculator.calculate_statistical_summary(response)
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Perplexity", f"{stats.get('perplexity', 0):.2f}")
            with col2:
                st.metric("Avg Confidence",
                          f"{stats.get('average_confidence', 0):.1f}%")
            with col3:
                st.metric("Uncertainty",
                          f"{stats.get('uncertainty_score', 0):.3f}")
            with col4:
                st.metric("Entropy", f"{stats.get('entropy', 0):.2f}")

        # Export options
        st.subheader("💾 Export Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export Text"):
                text_export = export_manager.export_text(
                    response, original_prompt)
                st.download_button("Download Text", text_export,
                                   "generated_text.txt")

        with col2:
            if st.button("📊 Export Analysis"):
                analysis_export = export_manager.export_analysis(response, {})
                st.download_button("Download Analysis", analysis_export,
                                   "analysis.json")

        with col3:
            if st.button("📈 Export CSV"):
                csv_export = export_manager.export_csv(response)
                st.download_button("Download CSV", csv_export,
                                   "token_analysis.csv")


if __name__ == "__main__":
    main()
