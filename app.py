"""
CodeHelwell - AI Text Generation Platform
Homepage and main entry point for the multi-page application
"""

import streamlit as st
import os
from utils.rate_limiter import RateLimiter

# Configure page
st.set_page_config(
    page_title="CodeHelwell - AI Text Generation",
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
</style>
""", unsafe_allow_html=True)

# Initialize rate limiter
rate_limiter = RateLimiter()

def main():
    """CodeHelwell Homepage - Welcome to AI Text Generation Platform."""
    
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
    
    # Current usage status
    st.markdown("---")
    st.markdown("## Current System Status")
    
    # Display current rate limiting status
    rate_limiter.display_rate_limit_info()
    
    # Security and privacy notice
    st.success("🔒 **Secure & Private**: This application uses enterprise-grade security with rate limiting and secure API management.")
    
    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem 0;'>
        Built with precision by <strong>CodeHelwell</strong><br>
        <em>Transforming ideas into intelligent solutions</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()