"""
About CodeHalwell Page
OpenAI Logprobs Text Generator - CodeHalwell
"""

import streamlit as st

# Note: st.set_page_config() is only called in main app.py for multi-page apps

def main():
    """Display the About CodeHalwell section."""
    
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("assets/logo.png", width=300)
        except:
            st.markdown("**CodeHalwell**")
    
    st.title("About Daniel & CodeHalwell")
    
    st.markdown("""
    ## Hello! I'm Daniel
    
    **Senior Scientist & Data Scientist | AI Innovation Enthusiast**
    
    I'm deeply passionate about using advanced analytics, machine learning (ML), and artificial intelligence (AI) to tackle complex challenges and spark innovation. My journey into the world of data science is built upon a robust foundation of analytical problem-solving, a skill I initially developed and refined within the pharmaceutical industry.
    
    ### My Expertise
    
    **Core Technologies & Skills:**
    - **Python Ecosystem**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, PyTorch
    - **Machine Learning**: Supervised & Unsupervised Learning, Deep Learning (CNN, RNN, Transformers)
    - **AI Specializations**: Natural Language Processing, Computer Vision, Generative AI
    - **Data Science**: Statistical Analysis, Feature Engineering, Bayesian Optimization, Time Series Analysis
    - **Development**: REST API Development (FastAPI, Flask), Web Development fundamentals
    - **Cloud & MLOps**: AWS, GCP, Azure (personal use), Docker, CI/CD understanding
    
    ### Professional Journey
    
    **Current Role: Senior Scientist at AstraZeneca (2021 - Present)**
    - Leverage Python for task automation and developing predictive machine learning models
    - Created a Generative AI troubleshooting chatbot using OpenAI with Streamlit interface
    - Designed and deployed REST APIs using FastAPI to improve data accessibility
    - Active member of AstraZeneca's internal Coding Network, mentoring and coaching others
    
    **Previous Experience:**
    - **Analytical Specialist** at Sanofi/Recipharm (2016-2021)
    - **Senior Quality Analyst** at Sanofi (2012-2016)
    
    ### Achievements & Recognition
    
    - **4th Place** in Kaggle Playground Series competition
    - **Multiple Professional Certifications**: IBM AI Developer, DeepLearning.AI TensorFlow Developer, Google Data Analytics
    - **Continuous Learning**: Actively engaged across platforms like Coursera, DataCamp, Codecademy
    
    ### About This Application
    
    This OpenAI Logprobs Text Generator represents the intersection of my professional expertise and passion for making AI accessible. As someone who regularly develops AI-powered applications at AstraZeneca, I wanted to create a tool that demonstrates:
    
    - **Token-level Analysis**: Visualize exactly how confident AI models are in each generated word
    - **Real-time Insights**: Interactive exploration of model decision-making processes
    - **Educational Value**: Help others understand how modern language models work
    - **Production Quality**: Enterprise-grade security, rate limiting, and error handling
    
    ### My Philosophy
    
    I believe in the power of sharing knowledge and insights, which is why I regularly contribute articles on coding and data science to my Medium platform, "CodeHalwell". Whether it's automating intricate tasks, building sophisticated predictive models, or diving headfirst into emerging technologies, I am consistently driven by the intellectual challenge and the profound opportunity to make a tangible, positive impact.
    
    ### Technical Implementation
    
    This application showcases several key technical concepts:
    
    - **Streamlit Framework**: Rapid prototyping and deployment of data science applications
    - **OpenAI API Integration**: Advanced logprobs analysis for model transparency
    - **Statistical Analysis**: Perplexity, entropy, and confidence distribution calculations
    - **Data Visualization**: Interactive charts using Plotly for real-time feedback
    - **Security Architecture**: Multi-layered protection with input validation and rate limiting
    
    ### CodeHalwell Platform
    
    This application is part of my broader CodeHalwell initiative, where I share knowledge through:
    
    - **Medium Articles**: Technical tutorials and insights on data science and AI
    - **GitHub Projects**: Open-source implementations and code examples
    - **Interactive Demos**: Applications like this that make complex concepts accessible
    - **Community Engagement**: Mentoring and knowledge sharing within professional networks
    
    ### Professional Background
    
    **Education**: MChem in Chemistry from Loughborough University (2007-2012)
    
    **Certifications**:
    - IBM AI Developer Professional Certificate (2024)
    - DeepLearning.AI TensorFlow Developer (2024)
    - Google Data Analytics Professional Certificate (2024)
    - Docker Foundations Professional Certificate (2025)
    
    ### Beyond Work
    
    I'm driven by continuous learning and diverse interests including:
    - **Technology**: Constantly expanding knowledge in coding, AI, and cybersecurity
    - **Sports**: Avid follower of rugby, football, and Formula 1
    - **Gaming & Entertainment**: Unwinding with games and diverse TV shows
    - **Social Awareness**: Staying informed about current events with strong values around inclusion and diversity
    
    ---
    
    **Built with precision by Daniel | CodeHalwell**
    
    *Transforming data into insights, ideas into intelligent solutions*
    """)
    
    # Professional metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Years Experience", "12+", help="Professional experience since 2012")
    with col2:
        st.metric("Kaggle Ranking", "4th Place", help="Kaggle Playground Series competition")
    with col3:
        st.metric("Certifications", "4+", help="IBM AI, TensorFlow, Google Analytics, Docker")
    with col4:
        st.metric("Current Role", "Senior Scientist", help="AstraZeneca, 2021-Present")

if __name__ == "__main__":
    main()