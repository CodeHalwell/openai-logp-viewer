import streamlit as st

# Configure page - must be the very first Streamlit command
st.set_page_config(page_title="CodeHalwell - AI Text Generation",
                   page_icon="🤖",
                   layout="wide",
                   initial_sidebar_state="expanded")

import os
from utils.rate_limiter import RateLimiter

# Initialize rate limiter
rate_limiter = RateLimiter()


def main():
    """CodeHalwell Homepage - Welcome to AI Text Generation Platform."""

    # Hero section with title first, then logo
    st.markdown('<div class="main-header">', unsafe_allow_html=True)

    # Welcome message
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>Welcome to AI Text Generation Platform</h1>
        <h3>Experience the Future of AI-Powered Text Generation</h3>
        <p style='font-size: 1.2em; color: #666;'>
            Visualize AI model confidence in real-time with advanced logprob analysis
        </p>
    </div>
    """,
                unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(
            "ℹ️ Important: How this app works & how your data is used",
            expanded=True,
            icon="🚨"):
        st.warning("""
            This educational tool is provided to you **free of charge**. This is made possible by allowing OpenAI to process the (non-personal) text you enter to improve their services.

            Here’s a clear breakdown of what this means:

            * **Your input is sent to OpenAI:** To generate text completions and their probabilities, the text you enter is processed by OpenAI's servers.

            * **⚠️ Do Not Share Sensitive Information:** Because your input is shared, you **must not** enter any personally identifiable information (PII) such as names, addresses, phone numbers, or any other private data.

            * **Stateless Interaction:** This app has no memory. Your prompts are not saved or associated with you by this application.
            """)

    # Sidebar with logo
    with st.sidebar:
        try:
            st.image("assets/logo.png", use_container_width=True)
        except:
            st.markdown("**CodeHalwell**")

        st.divider()

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
        
        ℹ️ **About** - Learn about CodeHalwell
        """)

    # Technology showcase
    st.markdown("---")
    st.markdown("## Technology Showcase")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("OpenAI Models",
                  "3",
                  help="GPT-4o, GPT-4o-mini, GPT-4.1-mini")

    with col2:
        st.metric("Security Layers",
                  "5+",
                  help="Rate limiting, input validation, CSP, error handling")

    with col3:
        st.metric("Visualization Types",
                  "4+",
                  help="Color schemes, charts, statistics, exports")

    with col4:
        st.metric("Export Formats",
                  "3",
                  help="Text, JSON, CSV with detailed metadata")

    # Current usage status
    st.markdown("---")
    st.markdown("## Current System Status")

    # Display current rate limiting status
    rate_limiter.display_rate_limit_info()

    # Security and privacy notice
    st.success(
        "🔒 **Secure & Private**: This application uses enterprise-grade security with rate limiting and secure API management."
    )

    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem 0;'>
        Built with precision by <strong>CodeHalwell</strong><br>
        <em>Transforming ideas into intelligent solutions</em><br>
        <a href="https://codehalwell.io" target="_blank" style="color: #0066cc; text-decoration: none;">🌐 codehalwell.io</a>
    </div>
    """,
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
