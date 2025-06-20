# OpenAI Logprobs Text Generator

## Overview

This is a Streamlit-based web application that demonstrates text completion using OpenAI's API with visual highlighting based on log probabilities. The application generates text continuations from user prompts and visualizes the model's confidence in each token through color-coded highlighting. Each word is colored based on the model's confidence level, providing insights into the generation process.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based UI framework for Python applications
- **Interactive Components**: Text input, sliders, selectboxes, and real-time visualization
- **Responsive Layout**: Wide layout configuration with expandable sidebar
- **Plotly Integration**: Interactive charts and visualizations for statistical analysis

### Backend Architecture
- **OpenAI API Integration**: Direct integration with OpenAI's completion API with logprobs enabled
- **Modular Utility System**: Separated concerns into utility modules for better maintainability
- **Caching Layer**: Smart caching system for API responses to reduce costs and improve performance
- **Statistical Processing**: Real-time calculation of perplexity, confidence metrics, and token analysis

### Security Architecture
- **Secure API Key Handling**: Cryptographic hashing and secure session management
- **Input Sanitization**: Proper validation and sanitization of user inputs
- **Error Handling**: Sanitized error messages without information disclosure
- **Content Security**: Prevention of script injection and iframe attacks

## Key Components

### Core Application (`app.py`)
- Main Streamlit application with security hardening
- OpenAI API client initialization and management
- Text generation with logprob analysis
- Real-time visualization of token confidence
- Export functionality for generated content

### Utility Modules
1. **Cache Manager** (`utils/cache_manager.py`)
   - SHA256-based cache key generation with salt
   - Session-based caching for API responses
   - Cache statistics tracking and management

2. **Color Scheme Manager** (`utils/color_schemes.py`)
   - Multiple color schemes for token confidence visualization
   - Absolute probability mapping for consistent coloring
   - Schemes: confidence, rainbow, heat, ocean, monochrome, pastel

3. **Export Manager** (`utils/export_manager.py`)
   - Multiple export formats (text, JSON, CSV)
   - Comprehensive data export including metadata
   - Timestamp and attribution tracking

4. **Statistics Calculator** (`utils/statistics.py`)
   - Perplexity calculation for generated text
   - Average confidence metrics
   - Statistical analysis using scipy

### Configuration
- **Streamlit Config** (`.streamlit/config.toml`): Server settings and theme configuration
- **Project Dependencies** (`pyproject.toml`): Python package management with uv
- **Environment Variables**: Secure API key management through .env files

## Data Flow

1. **User Input**: User enters a prompt and selects generation parameters
2. **Cache Check**: System checks if response is cached using SHA256 key
3. **API Request**: If not cached, makes secure request to OpenAI API with logprobs enabled
4. **Response Processing**: Extracts tokens, logprobs, and calculates statistics
5. **Visualization**: Applies color schemes based on token confidence levels
6. **Display**: Renders highlighted text with interactive statistics
7. **Export Options**: Provides multiple export formats for generated content

## External Dependencies

### Core Dependencies
- **OpenAI**: Official OpenAI Python client for API integration
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation for export functionality
- **Plotly**: Interactive visualizations and charts
- **SciPy**: Statistical calculations and analysis
- **Python-dotenv**: Environment variable management

### Development Tools
- **UV**: Modern Python package manager for dependency resolution
- **Nix**: Reproducible development environment
- **Git**: Version control with comprehensive .gitignore

## Deployment Strategy

### Replit Deployment
- **Autoscale Deployment**: Configured for automatic scaling based on demand
- **Port Configuration**: Streamlit runs on port 5000, exposed as port 80
- **Workflow Management**: Parallel execution workflows for development and production

### Environment Configuration
- **Python 3.11**: Specified Python version for consistency
- **Nix Packages**: System-level dependencies for mathematical computations
- **Security Headers**: Content security policies implemented via Streamlit

### Performance Optimizations
- **Caching Strategy**: Reduces API calls and improves response times
- **Resource Management**: Efficient memory usage and cleanup
- **Lazy Loading**: Components loaded on-demand to reduce initial load time

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 20, 2025. Initial setup