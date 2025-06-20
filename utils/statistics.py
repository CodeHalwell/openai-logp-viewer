"""
Statistics Calculator for OpenAI Logprobs Application
Handles statistical calculations and analysis for text generation data.
"""

import numpy as np
from math import exp, log
from typing import Dict, Any, Optional, List
import streamlit as st


class StatisticsCalculator:
    """Handles statistical calculations for OpenAI API responses with logprobs."""
    
    def __init__(self):
        """Initialize statistics calculator."""
        pass
    
    def calculate_perplexity(self, response: Any) -> float:
        """
        Calculate perplexity of the generated text.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Perplexity value
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return 0.0
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return 0.0
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return 0.0
            
            # Calculate average log probability
            total_logprob = sum(token.logprob for token in logprobs if token.logprob is not None)
            num_tokens = len([token for token in logprobs if token.logprob is not None])
            
            if num_tokens == 0:
                return 0.0
            
            avg_logprob = total_logprob / num_tokens
            perplexity = exp(-avg_logprob)
            
            return perplexity
        
        except Exception as e:
            # Log error securely without exposing details
            return 0.0
    
    def calculate_entropy(self, response: Any) -> float:
        """
        Calculate entropy of the generated text.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Entropy value
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return 0.0
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return 0.0
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return 0.0
            
            # Calculate entropy as negative sum of p * log(p)
            entropy = 0.0
            for token in logprobs:
                if token.logprob is not None:
                    prob = exp(token.logprob)
                    if prob > 0:
                        entropy -= prob * token.logprob
            
            return entropy
        
        except Exception as e:
            return 0.0
    
    def calculate_average_confidence(self, response: Any) -> float:
        """
        Calculate average confidence percentage.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Average confidence as percentage
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return 0.0
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return 0.0
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return 0.0
            
            # Convert log probabilities to percentages
            confidences = []
            for token in logprobs:
                if token.logprob is not None:
                    confidence = exp(token.logprob) * 100
                    confidences.append(confidence)
            
            if not confidences:
                return 0.0
            
            return sum(confidences) / len(confidences)
        
        except Exception as e:
            return 0.0
    
    def calculate_uncertainty_score(self, response: Any) -> float:
        """
        Calculate uncertainty score based on logprob variance.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Uncertainty score (0-1, higher = more uncertain)
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return 0.0
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return 0.0
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return 0.0
            
            # Calculate variance of log probabilities
            logprob_values = [token.logprob for token in logprobs if token.logprob is not None]
            
            if len(logprob_values) < 2:
                return 0.0
            
            mean_logprob = sum(logprob_values) / len(logprob_values)
            variance = sum((x - mean_logprob) ** 2 for x in logprob_values) / len(logprob_values)
            
            # Normalize to 0-1 scale (higher variance = higher uncertainty)
            uncertainty = min(variance / 10.0, 1.0)  # Cap at 1.0
            
            return uncertainty
        
        except Exception as e:
            return 0.0
    
    def get_confidence_distribution(self, response: Any) -> Dict[str, int]:
        """
        Get distribution of confidence levels.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Dictionary with confidence distribution
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return {"High": 0, "Medium": 0, "Low": 0}
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return {"High": 0, "Medium": 0, "Low": 0}
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return {"High": 0, "Medium": 0, "Low": 0}
            
            distribution = {"High": 0, "Medium": 0, "Low": 0}
            
            for token in logprobs:
                if token.logprob is not None:
                    confidence = exp(token.logprob) * 100
                    
                    if confidence >= 70:
                        distribution["High"] += 1
                    elif confidence >= 30:
                        distribution["Medium"] += 1
                    else:
                        distribution["Low"] += 1
            
            return distribution
        
        except Exception as e:
            return {"High": 0, "Medium": 0, "Low": 0}
    
    def calculate_statistical_summary(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive statistical summary.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            Dictionary with all statistical measures
        """
        try:
            if not response:
                return None
            
            summary = {
                "perplexity": self.calculate_perplexity(response),
                "entropy": self.calculate_entropy(response),
                "average_confidence": self.calculate_average_confidence(response),
                "uncertainty_score": self.calculate_uncertainty_score(response),
                "confidence_distribution": self.get_confidence_distribution(response)
            }
            
            return summary
        
        except Exception as e:
            return None
    
    def get_token_statistics(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Get detailed statistics for each token.
        
        Args:
            response: OpenAI API response with logprobs
        
        Returns:
            List of token statistics
        """
        try:
            if not response or not hasattr(response, 'choices'):
                return None
            
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return None
            
            logprobs = choice.logprobs.content
            if not logprobs:
                return None
            
            token_stats = []
            
            for i, token in enumerate(logprobs):
                if token.logprob is not None:
                    confidence = exp(token.logprob) * 100
                    
                    token_stat = {
                        "position": i,
                        "token": token.token,
                        "logprob": token.logprob,
                        "probability": exp(token.logprob),
                        "confidence_percent": confidence,
                        "confidence_level": "High" if confidence >= 70 else "Medium" if confidence >= 30 else "Low"
                    }
                    
                    token_stats.append(token_stat)
            
            return token_stats
        
        except Exception as e:
            return None