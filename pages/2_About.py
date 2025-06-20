"""
About CodeHelwell Page
OpenAI Logprobs Text Generator - CodeHelwell
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="About CodeHelwell",
    page_icon="🏢",
    layout="wide"
)

def main():
    """Display the About CodeHelwell section."""
    
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("assets/logo.png", width=300)
        except:
            st.markdown("**CodeHelwell**")
    
    st.title("About CodeHelwell")
    
    st.markdown("""
    ## Welcome to CodeHelwell
    
    **Empowering Innovation Through AI and Technology Solutions**
    
    CodeHelwell specializes in creating cutting-edge applications that demonstrate the power of artificial intelligence 
    and modern software development. Our mission is to make advanced AI technologies accessible and understandable 
    through practical, real-world applications.
    
    ### What We Do
    
    - **AI Application Development**: Creating innovative applications that showcase AI capabilities
    - **Technology Consulting**: Helping businesses integrate AI solutions into their workflows
    - **Educational Resources**: Building tools that help others understand AI and machine learning
    - **Open Source Contributions**: Sharing knowledge and tools with the developer community
    
    ### About This Application
    
    The OpenAI Logprobs Text Generator is a demonstration of how we can visualize AI model confidence in real-time. 
    This application shows:
    
    - **Token-level Analysis**: See exactly how confident the AI is in each word it generates
    - **Visual Feedback**: Color-coded highlighting based on probability scores
    - **Statistical Insights**: Detailed metrics about text generation patterns
    - **Security Best Practices**: Enterprise-grade security and rate limiting
    
    ### Our Approach
    
    At CodeHelwell, we believe in:
    
    - **Transparency**: Making AI decisions visible and understandable
    - **Security**: Implementing robust security measures in all applications
    - **User Experience**: Creating intuitive interfaces that anyone can use
    - **Performance**: Building efficient, scalable solutions
    - **Innovation**: Pushing the boundaries of what's possible with AI
    
    ### Technology Stack
    
    This application demonstrates our expertise with:
    
    - **Frontend**: Streamlit for rapid prototyping and deployment
    - **AI Integration**: OpenAI API with advanced logprobs analysis
    - **Security**: Multi-layered security with rate limiting and input validation
    - **Data Visualization**: Interactive charts and real-time feedback
    - **Performance**: Efficient caching and resource management
    
    ### Get In Touch
    
    Interested in working with CodeHelwell or learning more about our projects?
    
    - **Explore**: Try out the Text Generator to see our work in action
    - **Connect**: Reach out to discuss potential collaborations
    - **Learn**: Use our applications to understand AI capabilities
    
    ### Future Developments
    
    We're constantly working on new features and applications:
    
    - Enhanced visualization techniques
    - Support for more AI models and providers
    - Advanced analytics and reporting
    - Integration with business workflows
    - Educational content and tutorials
    
    ---
    
    **Built with precision by CodeHelwell**
    
    *Transforming ideas into intelligent solutions*
    """)
    
    # Footer metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projects Delivered", "25+")
    with col2:
        st.metric("AI Models Integrated", "10+")
    with col3:
        st.metric("Years of Experience", "5+")

if __name__ == "__main__":
    main()