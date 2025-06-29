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
                token_str = bytes(token.bytes).decode("utf-8",
                                                      errors="replace")
            else:
                token_str = str(token.token) if hasattr(
                    token, 'token') else str(token)

            # Calculate color based on logprob using color scheme manager
            color = color_manager.get_color(token.logprob, min_logprob,
                                            max_logprob, color_scheme)

            # Create styled span with enhanced hover effects
            probability_percent = round(exp(token.logprob) * 100, 2)

            # Get alternative tokens if available
            alternatives = ""
            if hasattr(token, 'top_logprobs') and token.top_logprobs:
                alt_list = []
                for alt in token.top_logprobs[:3]:
                    if hasattr(alt, 'bytes') and alt.bytes is not None:
                        alt_str = bytes(alt.bytes).decode("utf-8",
                                                          errors="replace")
                    else:
                        alt_str = str(alt.token) if hasattr(
                            alt, 'token') else str(alt)

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


def create_enhanced_logprob_chart(response, chart_type="bar", max_tokens=20):
    """Create enhanced charts showing logprob values with multiple visualization options."""
    try:
        if not response or not hasattr(response, 'choices'):
            return None

        choice = response.choices[0]
        if not hasattr(choice, 'logprobs') or not choice.logprobs:
            return None

        logprobs = choice.logprobs.content
        if not logprobs:
            return None

        # Prepare data
        tokens = []
        probabilities = []
        positions = []

        for i, token in enumerate(logprobs):
            if token.logprob is not None:
                # Get token string
                if hasattr(token, 'bytes') and token.bytes is not None:
                    token_str = bytes(token.bytes).decode("utf-8",
                                                          errors="replace")
                else:
                    token_str = str(token.token) if hasattr(
                        token, 'token') else str(token)

                tokens.append(token_str)
                probabilities.append(exp(token.logprob) * 100)
                positions.append(i + 1)

        # Limit to user-specified number for visualization
        tokens = tokens[:max_tokens]
        probabilities = probabilities[:max_tokens]
        positions = positions[:max_tokens]

        if chart_type == "bar":
            # Create confidence-based colors for bars
            colors = []
            for prob in probabilities:
                if prob >= 80:
                    colors.append('#2E8B57')  # Sea Green - Very High
                elif prob >= 60:
                    colors.append('#32CD32')  # Lime Green - High
                elif prob >= 40:
                    colors.append('#FFD700')  # Gold - Medium
                elif prob >= 20:
                    colors.append('#FF8C00')  # Dark Orange - Low
                else:
                    colors.append('#DC143C')  # Crimson - Very Low

            fig = go.Figure(data=[
                go.Bar(x=positions,
                       y=probabilities,
                       marker_color=colors,
                       text=[
                           f"{token}<br>{prob:.1f}%"
                           for token, prob in zip(tokens, probabilities)
                       ],
                       textposition="outside",
                       textfont=dict(size=10),
                       hovertemplate="<b>Position:</b> %{x}<br>" +
                       "<b>Token:</b> %{text}<br>" +
                       "<b>Confidence:</b> %{y:.2f}%<extra></extra>",
                       customdata=tokens)
            ])
            fig.update_layout(
                title="Token Confidence Levels",
                xaxis_title="Token Position",
                yaxis_title="Confidence (%)",
                showlegend=False,
                yaxis=dict(range=[0,
                                  max(probabilities) * 1.2])  # Add 20% padding
            )

        elif chart_type == "line":
            fig = px.line(x=positions,
                          y=probabilities,
                          title="Token Confidence Progression",
                          labels={
                              "x": "Token Position",
                              "y": "Confidence (%)"
                          },
                          markers=True)
            fig.update_traces(hovertemplate="<b>Position:</b> %{x}<br>" +
                              "<b>Token:</b> %{text}<br>" +
                              "<b>Confidence:</b> %{y:.2f}%<extra></extra>",
                              line=dict(width=3),
                              marker=dict(size=8),
                              text=tokens)
            # Add text annotations for each point
            for i, (pos, prob,
                    token) in enumerate(zip(positions, probabilities, tokens)):
                fig.add_annotation(x=pos,
                                   y=prob,
                                   text=token,
                                   showarrow=False,
                                   yshift=15,
                                   font=dict(size=10))
            fig.update_layout(
                yaxis=dict(range=[0, max(probabilities) *
                                  1.3])  # Add 30% padding for annotations
            )

        elif chart_type == "scatter":
            # Create confidence categories for color coding
            confidence_categories = []
            for prob in probabilities:
                if prob >= 80:
                    confidence_categories.append("Very High (80%+)")
                elif prob >= 60:
                    confidence_categories.append("High (60-80%)")
                elif prob >= 40:
                    confidence_categories.append("Medium (40-60%)")
                elif prob >= 20:
                    confidence_categories.append("Low (20-40%)")
                else:
                    confidence_categories.append("Very Low (<20%)")

            fig = px.scatter(
                x=positions,
                y=probabilities,
                color=confidence_categories,
                size=[max(5, (100 - prob) / 3) for prob in probabilities
                      ],  # Inverted size: lower confidence = larger
                title="Token Confidence Distribution",
                labels={
                    "x": "Token Position",
                    "y": "Confidence (%)",
                    "color": "Confidence Level"
                },
                color_discrete_map={
                    "Very High (80%+)": "#2E8B57",
                    "High (60-80%)": "#32CD32",
                    "Medium (40-60%)": "#FFD700",
                    "Low (20-40%)": "#FF8C00",
                    "Very Low (<20%)": "#DC143C"
                })
            fig.update_traces(hovertemplate="<b>Position:</b> %{x}<br>" +
                              "<b>Token:</b> %{text}<br>" +
                              "<b>Confidence:</b> %{y:.2f}%<br>" +
                              "<b>Level:</b> %{color}<extra></extra>",
                              text=tokens)
            # Add text annotations for each point
            for i, (pos, prob,
                    token) in enumerate(zip(positions, probabilities, tokens)):
                fig.add_annotation(x=pos,
                                   y=prob,
                                   text=token,
                                   showarrow=False,
                                   yshift=15,
                                   font=dict(size=10))
            fig.update_layout(
                yaxis=dict(range=[0, max(probabilities) *
                                  1.3])  # Add 30% padding for annotations
            )

        elif chart_type == "box":
            # Create box plot showing confidence distribution
            fig = go.Figure()
            fig.add_trace(
                go.Box(y=probabilities,
                       name="Token Confidence",
                       boxpoints='all',
                       jitter=0.3,
                       pointpos=-1.8,
                       text=tokens,
                       hovertemplate="<b>Token:</b> %{text}<br>" +
                       "<b>Confidence:</b> %{y:.2f}%<extra></extra>"))
            fig.update_layout(title="Token Confidence Distribution (Box Plot)",
                              yaxis_title="Confidence (%)",
                              showlegend=False)

        else:  # Default to bar chart
            fig = px.bar(x=positions,
                         y=probabilities,
                         title="Token Confidence Levels",
                         labels={
                             "x": "Token Position",
                             "y": "Confidence (%)"
                         })

        # Common styling
        fig.update_layout(height=400,
                          font=dict(size=12),
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')

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
                    token_str = bytes(token.bytes).decode("utf-8",
                                                          errors="replace")
                else:
                    token_str = str(token.token) if hasattr(
                        token, 'token') else str(token)

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

                html_parts.append(
                    f'<span style="{style}">{escaped_token}</span>')

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
        chart_type = st.selectbox("Chart Type",
                                  ["bar", "line", "scatter", "box"])
        max_tokens_display = st.slider(
            "Tokens to Display",
            5,
            500,
            50,
            1,
            help="Number of tokens to show in charts")

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

    with st.expander(
            "ℹ️ Important: How this app works & how your data is used",
            expanded=False,
            icon="🚨"):
        st.warning("""
                This educational tool is provided to you **free of charge**. This is made possible by allowing OpenAI to process the (non-personal) text you enter to improve their services.
        
                Here’s a clear breakdown of what this means:
        
                * **Your input is sent to OpenAI:** To generate text completions and their probabilities, the text you enter is processed by OpenAI's servers.
        
                * **⚠️ Do Not Share Sensitive Information:** Because your input is shared, you **must not** enter any personally identifiable information (PII) such as names, addresses, phone numbers, or any other private data.
        
                * **Stateless Interaction:** This app has no memory. Your prompts are not saved or associated with you by this application.
                """)
    with st.expander("💡 User Guide: Understanding the Generation Settings",
                     expanded=False):
        st.markdown("""

            ## 💡 User Guide: Understanding the Generation Settings
        
            This guide explains what each setting in the sidebar does and how you can use them to shape the AI's response to your prompt. Experimenting with these parameters is the best way to understand their effect.

            ## 🤖 Model Selection
            This dropdown selects the "brain" of the AI. Each model has different strengths, speeds, and costs.

            - gpt-4o-mini: The current model is a fantastic all-rounder. It's designed to be very fast and cost-effective while still being highly intelligent. It's great for a wide range of tasks, from creative writing to factual Q&A and code generation.

            ## 🎛️ Parameters
            These sliders allow you to fine-tune the behaviour of the selected model.

            ### Temperature
            **What it does:** 
            
            - Controls the randomness of the output. Think of it as a "creativity" or "risk-taking" knob.

            **How it works:**

            - Lower values (e.g., 0.2): Make the model more focused and deterministic. It will pick the most common and logical words, making the output safer and more predictable. This is ideal for factual summaries, translations, or code.

            - Higher values (e.g., 1.2): Make the model more creative and surprising. It will be more willing to pick less common words, which can lead to interesting ideas but also increases the chance of errors or nonsensical text. This is good for brainstorming or creative writing.

            - Default (0.70): A balanced setting suitable for most tasks.

            ### Max Tokens
            **What it does:** 
            
            - Sets a hard limit on the maximum length of the AI's response.

            **How it works:** 
            
            - A "token" is a piece of a word; on average, 100 tokens is about 75 words. The model will stop generating text once it reaches this limit, even if it's in the middle of a sentence.

            **When to use it:**

            - Use lower values if you need a short, concise answer (like a headline or a quick summary).

            - Use higher values if you need a long, detailed response (like an article or a story).

            ### Frequency Penalty
            **What it does:** 
            
            - Discourages the model from repeating the same word or phrase over and over again.

            **How it works:** 
            
            - Positive values apply a small penalty to words each time they are used, making them slightly less likely to be chosen again. This helps increase the linguistic variety of the output.

            **When to use it:**

            - Set it to a positive value (e.g., 0.5 - 1.5) if you find the AI output is too repetitive and you want more varied language.

            - Leave it at 0.0 if you don't want to influence this behaviour.

            ### Presence Penalty
            **What it does:** 
            
            - Encourages the model to introduce new topics or concepts in its response.

            **How it works:**
            
            - While frequency penalty penalises words for being used repeatedly, presence penalty penalises them just for being used at all. A positive value will push the model to talk about different things and avoid getting stuck on a single idea.

            **When to use it:**

            - Set it to a positive value (e.g., 0.5 - 1.5) when you are brainstorming or want the AI to explore a topic from multiple different angles.

            - Leave it at 0.0 for more focused, on-topic responses.
        """)

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
            """,
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="analysis-container">{highlighted_html}</div>',
                unsafe_allow_html=True)

        # Top Choice Analysis
        st.subheader("🎯 Top Choice Analysis")
        top_choice_html = create_top_choice_analysis(response)
        if top_choice_html:
            st.markdown(
                f'<div class="analysis-container">{top_choice_html}</div>',
                unsafe_allow_html=True)

        # Probability Chart
        st.subheader("📈 Probability Chart")
        fig = create_enhanced_logprob_chart(response, chart_type,
                                            max_tokens_display)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Token details table
        st.subheader("🔍 Token Details")
        st.write("Interactive table with detailed information for each token:")

        # Create token details data
        token_data = []
        choice = response.choices[0]
        if hasattr(choice,
                   'logprobs') and choice.logprobs and choice.logprobs.content:
            logprobs = choice.logprobs.content
            if logprobs:
                for i, token in enumerate(logprobs):
                    if token.logprob is not None:
                        confidence = min(100, max(0, exp(token.logprob) * 100))

                        # Get token string
                        if hasattr(token, 'bytes') and token.bytes is not None:
                            token_str = bytes(token.bytes).decode(
                                "utf-8", errors="replace")
                        else:
                            token_str = str(token.token) if hasattr(
                                token, 'token') else str(token)

                        # Get top alternatives with better formatting
                        alt_1, alt_2, alt_3 = "—", "—", "—"
                        alt_1_conf, alt_2_conf, alt_3_conf = "", "", ""

                        if hasattr(token,
                                   'top_logprobs') and token.top_logprobs:
                            for idx, alt in enumerate(token.top_logprobs[:3]):
                                if hasattr(alt,
                                           'bytes') and alt.bytes is not None:
                                    alt_str = bytes(alt.bytes).decode(
                                        "utf-8", errors="replace")
                                else:
                                    alt_str = str(alt.token) if hasattr(
                                        alt, 'token') else str(alt)

                                alt_confidence = exp(alt.logprob) * 100

                                if idx == 0:
                                    alt_1 = f'"{alt_str}"'
                                    alt_1_conf = f"{alt_confidence:.1f}%"
                                elif idx == 1:
                                    alt_2 = f'"{alt_str}"'
                                    alt_2_conf = f"{alt_confidence:.1f}%"
                                elif idx == 2:
                                    alt_3 = f'"{alt_str}"'
                                    alt_3_conf = f"{alt_confidence:.1f}%"

                        # Determine confidence level
                        if confidence >= 80:
                            conf_level = "🟢 Very High"
                        elif confidence >= 60:
                            conf_level = "🟡 High"
                        elif confidence >= 40:
                            conf_level = "🟠 Medium"
                        elif confidence >= 20:
                            conf_level = "🔴 Low"
                        else:
                            conf_level = "⚫ Very Low"

                        token_data.append({
                            "Position": i + 1,
                            "Token": f'"{token_str}"',
                            "Logprob": round(token.logprob, 4),
                            "Confidence %": round(confidence, 2),
                            "Level": conf_level,
                            "Alt 1": alt_1,
                            "Alt 1 %": alt_1_conf,
                            "Alt 2": alt_2,
                            "Alt 2 %": alt_2_conf,
                            "Alt 3": alt_3,
                            "Alt 3 %": alt_3_conf
                        })

        if token_data:
            # Search and filter functionality
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input(
                    "🔍 Search tokens:",
                    placeholder="Enter token text to search...")
            with col2:
                confidence_filter = st.selectbox("Filter by confidence:", [
                    "All", "Very High (80%+)", "High (60-80%)",
                    "Medium (40-60%)", "Low (20-40%)", "Very Low (<20%)"
                ])

            # Filter data based on search and confidence
            df = pd.DataFrame(token_data)

            # Remove quotation marks from Token and Alternative columns for cleaner display
            df['Token'] = df['Token'].str.replace('"', '')
            df['Alt 1'] = df['Alt 1'].str.replace('"', '')
            df['Alt 2'] = df['Alt 2'].str.replace('"', '')
            df['Alt 3'] = df['Alt 3'].str.replace('"', '')

            if search_term:
                mask = df['Token'].str.contains(search_term,
                                                case=False,
                                                na=False)
                df = df[mask]

            if confidence_filter != "All":
                if confidence_filter == "Very High (80%+)":
                    df = df[df['Confidence %'] >= 80]
                elif confidence_filter == "High (60-80%)":
                    df = df[(df['Confidence %'] >= 60)
                            & (df['Confidence %'] < 80)]
                elif confidence_filter == "Medium (40-60%)":
                    df = df[(df['Confidence %'] >= 40)
                            & (df['Confidence %'] < 60)]
                elif confidence_filter == "Low (20-40%)":
                    df = df[(df['Confidence %'] >= 20)
                            & (df['Confidence %'] < 40)]
                elif confidence_filter == "Very Low (<20%)":
                    df = df[df['Confidence %'] < 20]

            # Display results count
            if search_term or confidence_filter != "All":
                st.info(f"Showing {len(df)} of {len(token_data)} tokens")

            # Display enhanced interactive dataframe
            st.dataframe(df,
                         use_container_width=True,
                         hide_index=True,
                         column_config={
                             "Position":
                             st.column_config.NumberColumn("Pos",
                                                           width="small"),
                             "Token":
                             st.column_config.TextColumn("Token",
                                                         width="medium"),
                             "Logprob":
                             st.column_config.NumberColumn("Logprob",
                                                           width="small",
                                                           format="%.4f"),
                             "Confidence %":
                             st.column_config.NumberColumn("Conf %",
                                                           width="small",
                                                           format="%.2f"),
                             "Level":
                             st.column_config.TextColumn("Level",
                                                         width="medium"),
                             "Alt 1":
                             st.column_config.TextColumn("Alternative 1",
                                                         width="medium"),
                             "Alt 1 %":
                             st.column_config.TextColumn("Conf %",
                                                         width="small"),
                             "Alt 2":
                             st.column_config.TextColumn("Alternative 2",
                                                         width="medium"),
                             "Alt 2 %":
                             st.column_config.TextColumn("Conf %",
                                                         width="small"),
                             "Alt 3":
                             st.column_config.TextColumn("Alternative 3",
                                                         width="medium"),
                             "Alt 3 %":
                             st.column_config.TextColumn("Conf %",
                                                         width="small")
                         },
                         height=400)

            # Export functionality
            if st.button("📥 Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=
                    f"token_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv")

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
