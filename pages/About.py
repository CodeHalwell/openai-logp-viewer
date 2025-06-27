"""
About CodeHalwell Page
OpenAI Logprobs Text Generator - CodeHalwell
"""

import streamlit as st

# Note: st.set_page_config() is only called in main app.py for multi-page apps


def main():
    """Display the About CodeHalwell section."""

    st.title("About This Application and CodeHalwell")

    # Sidebar with logo
    with st.sidebar:
        try:
            st.image("assets/logo.png", use_container_width=True)
        except:
            st.markdown("**CodeHalwell**")

        st.divider()

    st.markdown("""
    ## About This Application

    This OpenAI Logprobs Text Generator is a tool designed to help people new to Generative AI to peek under the hood and see some of the underlying mechanics.

    ### Core Features

    - **Token-level Analysis**: Visualize exactly how confident AI models are in each generated word
    - **Real-time Insights**: Interactive exploration of model decision-making processes
    - **Educational Value**: Help others understand how modern language models work, how context is critical to the output, and how the model can be uncertain.
    - **Production Quality**: Enterprise-grade security, rate limiting, and error handling

    ### Technical Implementation

    This application showcases several key technical concepts:

    - **Streamlit Framework**: Rapid prototyping and deployment of data science applications
    - **OpenAI API Integration**: Advanced logprobs analysis for model transparency
    - **Statistical Analysis**: Perplexity, entropy, and confidence distribution calculations (detailed below)
    - **Data Visualization**: Interactive charts using Plotly for real-time feedback
    - **Security Architecture**: Multi-layered protection with input validation and rate limiting

    ## Mathematical Foundations

    ### Statistical Calculations Used in This Application

    #### Perplexity
    
    Perplexity relates to the LLM output by measuring how predictable the entire generated text is to the model. A low score signifies that the sequence of words is conventional and aligns closely with what the model would expect based on its training, making the text feel coherent. A high score means the text is surprising and contains unusual word combinations, which can be a sign of either creativity or a lack of logical flow.

    - **Low Temperature**: The model generates more predictable, focused text, which results in a lower perplexity score.
    
    - **High Temperature**: The model generates more random and surprising text, which results in a higher perplexity score.
    
    **Formula:**
    ```
    Perplexity = exp(-1/N * Σ(log P(token_i)))
    ```
    
    Where:
    - N = total number of tokens
    - P(token_i) = probability of token i
    - log P(token_i) = log probability (logprob) of token i
    
    **Implementation:**
    ```python
    def calculate_perplexity(response):
        logprobs = [choice.logprob for choice in response.choices[0].logprobs.content]
        avg_logprob = sum(logprobs) / len(logprobs)
        return exp(-avg_logprob)
    ```

    #### Entropy
    
    Entropy relates to the LLM output by measuring the model's uncertainty at each specific point in the text as it's being generated. It essentially quantifies how "undecided" the model is about the next word. Low entropy at a certain point means the next word was an obvious choice, whereas high entropy reveals a moment where the model saw many different words as viable options, indicating a creative juncture or a point of ambiguity.

    - **Low Temperature**: The model's uncertainty is reduced, forcing it to favour the most likely word, resulting in lower entropy.
    
    - **High Temperature**: The model's uncertainty is increased by giving more weight to unlikely words, resulting in higher entropy.
    
    **Formula:**
    ```
    Entropy = -Σ(P(token_i) * log P(token_i))
    ```
    
    Where:
    - P(token_i) = probability of token i (converted from logprob: exp(logprob))
    
    **Implementation:**
    ```python
    def calculate_entropy(response):
        logprobs = [choice.logprob for choice in response.choices[0].logprobs.content]
        probs = [exp(logprob) for logprob in logprobs]
        entropy = -sum(p * log(p) for p in probs if p > 0)
        return entropy
    ```

    #### Confidence Distribution
    
    Confidence directly measures how strongly the model believed in the specific words it ultimately chose for the output. A high average confidence indicates that the model consistently selected the tokens it deemed most probable, leading to a reliable and stable response. A low average confidence, and a distribution with more low-confidence tokens, reveals that the model frequently ignored the "safe" choice in favour of more novel or risky ones.

    - **Low Temperature**: The model is forced to pick the highest-probability words, leading to a higher average confidence.
    
    - **High Temperature**: The model is encouraged to select lower-probability words, which naturally results in a lower average confidence.
    
    **Confidence Levels:**
    - Very High: P ≥ 80% (logprob ≥ -0.223)
    - High: 60% ≤ P < 80% (-0.511 ≤ logprob < -0.223)
    - Medium: 40% ≤ P < 60% (-0.916 ≤ logprob < -0.511)
    - Low: 20% ≤ P < 40% (-1.609 ≤ logprob < -0.916)
    - Very Low: P < 20% (logprob < -1.609)
    
    **Implementation:**
    ```python
    def get_confidence_distribution(response):
        logprobs = [choice.logprob for choice in response.choices[0].logprobs.content]
        distribution = {"Very High": 0, "High": 0, "Medium": 0, "Low": 0, "Very Low": 0}
        
        for logprob in logprobs:
            if logprob >= -0.223:
                distribution["Very High"] += 1
            elif logprob >= -0.511:
                distribution["High"] += 1
            elif logprob >= -0.916:
                distribution["Medium"] += 1
            elif logprob >= -1.609:
                distribution["Low"] += 1
            else:
                distribution["Very Low"] += 1
        
        return distribution
    ```

    #### Average Confidence
    
    The mean probability across all generated tokens, converted from logprobs.
    
    **Formula:**
    ```
    Average Confidence = (1/N) * Σ(exp(logprob_i)) * 100%
    ```
    
    **Implementation:**
    ```python
    def calculate_average_confidence(response):
        logprobs = [choice.logprob for choice in response.choices[0].logprobs.content]
        probs = [exp(logprob) for logprob in logprobs]
        return sum(probs) / len(probs) * 100
    ```

    #### Color Mapping for Visualization
    
    The application uses logprob values to assign colors to tokens for visual analysis:
    
    **Color Scheme Logic:**
    ```python
    def get_color_from_logprob(logprob):
        # Convert logprob to probability
        probability = exp(logprob)
        
        # Map to RGB color (green for high confidence, red for low)
        if probability >= 0.8:
            return "rgba(34, 139, 34, 0.8)"  # Dark green
        elif probability >= 0.6:
            return "rgba(154, 205, 50, 0.8)"  # Yellow green
        elif probability >= 0.4:
            return "rgba(255, 215, 0, 0.8)"   # Gold
        elif probability >= 0.2:
            return "rgba(255, 140, 0, 0.8)"   # Dark orange
        else:
            return "rgba(220, 20, 60, 0.8)"   # Crimson
    ```

    ## About Me 
    
    ### Hello! I'm Daniel
    
    **Senior Scientist & "Data Scientist" | AI Innovation Enthusiast**
    
    I'm deeply passionate about using advanced analytics, machine learning (ML), and artificial intelligence (AI) to tackle complex challenges and spark innovation. My journey into the world of data science is built upon a robust foundation of analytical problem-solving, a skill I initially developed and refined within the pharmaceutical industry.
    
    ### My Expertise
    
    **Core Technologies & Skills:**
    - **Python Ecosystem**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, PyTorch
    - **Machine Learning**: Supervised (e.g., XGBoost, Random Forest, LGBM), Unsupervised (e.g., k-Means, PCA, t-SNE), Deep Learning (MLP, CNN, RNN, GNN etc.)
    - **AI Specializations**: Generative AI, Natural Language Processing, Computer Vision
    - **Data Science**: Statistical Analysis, Feature Engineering, Bayesian Optimization, Chemical Informatics
    - **Development**: REST API Development (FastAPI, Flask), Web Development fundamentals
    - **Cloud & MLOps**: AWS, GCP, Azure (all for personal use), Docker, CI/CD understanding
    
    ### Achievements & Recognition
    - **WINNER** of Modal Labs Choice Award 2025 for my Model Context Protocol entry to the Gradio MCP + Agents Hackathon - ShallowCodeResearch.
    - **4th Place** in Kaggle Playground Series competition
    - **Multiple Professional Certifications**: IBM AI Developer, DeepLearning.AI TensorFlow Developer, Google Data Analytics
    - **Continuous Learning**: Actively engaged across platforms like Coursera, DataCamp, Codecademy
    
    ### My Philosophy
    
    I believe in the power of sharing knowledge and insights, which is why I have recently released my website [codehalwell.io](https://codehalwell.io), aiming to discuss projects and my opinions on the ML and AI landscape. Whether it's automating intricate tasks, building sophisticated predictive models, or diving headfirst into emerging technologies, I am consistently driven by the intellectual challenge and the profound opportunity to make a tangible, positive impact.
    
    ### CodeHalwell Platform
    
    This application is part of my broader CodeHalwell initiative, where I share knowledge through:
    
    - **Website**: [codehalwell.io](https://codehalwell.io) - Central hub for all projects and content
    - **Medium Articles**: [Medium Code Halwell](https://medium.com/@danielhalwell) Technical tutorials and insights on data science and AI
    - **GitHub Projects**: [GitHub CodeHalwell](https://github.com/CodeHalwell) Open-source implementations and code examples
    - **Interactive Demos**: Applications like this that make complex concepts accessible
    - **Community Engagement**: Mentoring and knowledge sharing within professional networks
    
    ### Professional Background
    
    **Education**: MChem in Chemistry from Loughborough University (2007-2012)
    
    **Certifications**:
    - Microsoft Python Development Professional Certificate (2025)
    - Docker Foundations Professional Certificate (2025)
    - IBM AI Developer Professional Certificate (2024)
    - DeepLearning.AI TensorFlow Developer (2024)
    - Google Data Analytics Professional Certificate (2024)
    - Data Scientist Associate Certificate (2023)

    **Ongoing Certifications**:
    - IBM Generative AI Engineering
    - IBM RAG and Agentic AI
    - Agile with Atlassian Jira
    - Google Advanced Data Analytics
    - AWS Generative AI Applications

    (I like to keep busy 🤣)
    
    ### Beyond Work
    
    I'm driven by continuous learning and diverse interests including:
    - **Technology**: Constantly expanding knowledge in coding, AI, and cybersecurity
    - **Sports**: Avid follower of rugby, football, and Formula 1
    - **Gaming & Entertainment**: Unwinding with games and diverse TV shows
    - **Social Awareness**: Staying informed about current events with strong values around inclusion and diversity
    
    ---
    
    **Built with precision by Daniel | CodeHalwell**
    
    *Transforming data into insights, ideas into intelligent solutions*
    
    🌐 **Visit**: [codehalwell.io](https://codehalwell.io)
    """)

    # Professional metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Years Experience",
                  "12+",
                  help="Professional experience since 2012")
    with col2:
        st.metric("Kaggle Ranking",
                  "4th Place",
                  help="Kaggle Playground Series competition")
    with col3:
        st.metric("Certifications",
                  "6+",
                  help="IBM AI, TensorFlow, Google Analytics, Docker")
    with col4:
        st.metric("Learning Platforms",
                  "3+",
                  help="Coursera, DataCamp, Codecademy, and more")


if __name__ == "__main__":
    main()
