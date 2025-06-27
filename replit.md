# CodeHalwell - OpenAI Logprobs Text Generator

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

## Recent Changes

- June 27, 2025: Enhanced Token Details table with advanced interactivity
  - Added search functionality to filter tokens by text content
  - Implemented confidence level filtering (Very High, High, Medium, Low, Very Low)
  - Separated token alternatives into individual columns for better readability
  - Added color-coded confidence indicators with emoji symbols
  - Included CSV export functionality with timestamped filenames
  - Removed quotation marks from token display for cleaner presentation
  - Set table height to 400px for optimal scrolling experience
- June 27, 2025: Added comprehensive mathematical foundations to About page
  - Detailed Perplexity calculation formulas and implementation
  - Entropy calculation with mathematical notation
  - Confidence distribution categorization with logprob thresholds
  - Average confidence calculation methodology
  - Color mapping logic for token visualization
  - Educational code examples for all statistical measures
- June 27, 2025: Updated model selection to include only gpt-4o-mini (default), gpt-4o, and gpt-4.1-mini
  - Added loading indicators with spinning animation for page load and text generation
  - Made logo smaller (120px) specifically on Text Generator page
  - Added codehalwell.io website links to footer and About page sections
- June 20, 2025: Updated About page to reflect Daniel's professional background and expertise
  - Personalized content around Senior Scientist and Data Scientist experience
  - Highlighted Kaggle achievements and professional certifications
  - Integrated CodeHalwell Medium platform information
  - Removed company-specific references for privacy
  - Made logo smaller and more centered across all pages
- June 20, 2025: Fixed navigation structure and branding
  - Renamed main app from "app" to "Home" 
  - Removed duplicate about pages
  - Changed all "CodeHelwell" references to "CodeHalwell"
  - Streamlined navigation: Home, Text Generator, About
- June 20, 2025: Created multi-page app structure with professional CodeHalwell branding
  - Added homepage showcasing platform features
  - Integrated logo across all pages
  - Built dedicated About page highlighting Daniel's expertise
- June 20, 2025: Completed comprehensive cybersecurity audit and security hardening
  - Fixed critical MD5 vulnerability (replaced with SHA-256)
  - Enhanced input sanitization to prevent injection attacks
  - Implemented secure memory cleanup for sensitive data
  - Added Content Security Policy and security headers
  - Improved token estimation accuracy
  - Fixed error information disclosure vulnerabilities
  - Enhanced cache security mechanisms
- June 20, 2025: Implemented comprehensive rate limiting system to protect OpenAI tokens from abuse
  - 10 requests per minute limit
  - 2,000 tokens per minute limit  
  - 50,000 tokens per day limit
  - 500 tokens per request maximum
- June 20, 2025: Removed API key display from UI for security (critical vulnerability fix)
- June 20, 2025: Updated API key handling to use environment variables instead of manual input for improved security
- June 20, 2025: Migrated from Replit Agent to standard Replit environment
- June 20, 2025: Initial setup completed

## Changelog

Changelog:
- June 20, 2025. Initial setup