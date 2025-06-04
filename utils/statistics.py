"""
Statistics Calculator for OpenAI Logprobs Application
Provides statistical analysis and metrics for token generation data.
"""

import numpy as np
from math import exp, log
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats as scipy_stats

class StatisticsCalculator:
    """Calculates various statistics and metrics from OpenAI logprobs data."""
    
    def __init__(self):
        """Initialize statistics calculator."""
        pass
    
    def calculate_perplexity(self, response: Any) -> Optional[float]:
        """
        Calculate perplexity of the generated text.
        Perplexity = exp(-mean(logprobs))
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Perplexity value or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in tokens]
        
        if not logprobs:
            return None
        
        return float(np.exp(-np.mean(logprobs)))
    
    def calculate_average_confidence(self, response: Any) -> Optional[float]:
        """
        Calculate average confidence (probability) across all tokens.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Average confidence percentage or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        probabilities = [exp(token.logprob) * 100 for token in tokens]
        
        if not probabilities:
            return None
        
        return float(np.mean(probabilities))
    
    def calculate_uncertainty_score(self, response: Any) -> Optional[float]:
        """
        Calculate uncertainty score based on logprob variance.
        Higher variance indicates more uncertainty in predictions.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Uncertainty score or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in tokens]
        
        if len(logprobs) < 2:
            return None
        
        return float(np.std(logprobs))
    
    def calculate_entropy(self, response: Any) -> Optional[float]:
        """
        Calculate average entropy across all tokens.
        Entropy measures the unpredictability of the next token.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Average entropy or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        entropies = []
        
        for token in tokens:
            if hasattr(token, 'top_logprobs') and token.top_logprobs:
                # Calculate entropy from top logprobs
                probs = [exp(alt.logprob) for alt in token.top_logprobs]
                # Normalize probabilities to sum to 1
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                    entropy = -sum(p * log(p) for p in probs if p > 0)
                    entropies.append(entropy)
        
        if not entropies:
            return None
        
        return float(np.mean(entropies))
    
    def calculate_confidence_distribution(self, response: Any) -> Optional[Dict[str, int]]:
        """
        Calculate distribution of confidence levels (High/Medium/Low).
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Dictionary with confidence level counts or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        distribution = {"High": 0, "Medium": 0, "Low": 0}
        
        for token in tokens:
            probability = exp(token.logprob) * 100
            if probability > 50:
                distribution["High"] += 1
            elif probability > 20:
                distribution["Medium"] += 1
            else:
                distribution["Low"] += 1
        
        return distribution
    
    def calculate_token_surprisal(self, response: Any) -> Optional[List[float]]:
        """
        Calculate surprisal (negative logprob) for each token.
        Higher surprisal indicates more surprising/unexpected tokens.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            List of surprisal values or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        surprisals = [-token.logprob for token in tokens]
        
        return surprisals
    
    def calculate_rolling_perplexity(self, response: Any, window_size: int = 5) -> Optional[List[float]]:
        """
        Calculate rolling perplexity over a sliding window.
        
        Args:
            response: OpenAI completion response with logprobs
            window_size: Size of the sliding window
        
        Returns:
            List of rolling perplexity values or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in tokens]
        
        if len(logprobs) < window_size:
            return None
        
        rolling_perplexities = []
        for i in range(len(logprobs) - window_size + 1):
            window_logprobs = logprobs[i:i + window_size]
            window_perplexity = np.exp(-np.mean(window_logprobs))
            rolling_perplexities.append(float(window_perplexity))
        
        return rolling_perplexities
    
    def calculate_statistical_summary(self, response: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive statistical summary of the generation.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Dictionary with various statistical measures
        """
        if not response or not response.choices[0].logprobs:
            return {"error": "No logprobs available for analysis"}
        
        tokens = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in tokens]
        probabilities = [exp(lp) * 100 for lp in logprobs]
        
        summary = {
            "token_count": len(tokens),
            "perplexity": self.calculate_perplexity(response),
            "average_confidence": self.calculate_average_confidence(response),
            "uncertainty_score": self.calculate_uncertainty_score(response),
            "entropy": self.calculate_entropy(response),
            "confidence_distribution": self.calculate_confidence_distribution(response),
            "logprob_statistics": {
                "min": float(np.min(logprobs)),
                "max": float(np.max(logprobs)),
                "mean": float(np.mean(logprobs)),
                "median": float(np.median(logprobs)),
                "std": float(np.std(logprobs)),
                "variance": float(np.var(logprobs))
            },
            "probability_statistics": {
                "min_percent": float(np.min(probabilities)),
                "max_percent": float(np.max(probabilities)),
                "mean_percent": float(np.mean(probabilities)),
                "median_percent": float(np.median(probabilities)),
                "std_percent": float(np.std(probabilities))
            }
        }
        
        # Add quartile information
        try:
            q25, q75 = np.percentile(logprobs, [25, 75])
            summary["logprob_statistics"]["q25"] = float(q25)
            summary["logprob_statistics"]["q75"] = float(q75)
            summary["logprob_statistics"]["iqr"] = float(q75 - q25)
        except Exception:
            pass
        
        # Add skewness and kurtosis if scipy is available
        try:
            summary["logprob_statistics"]["skewness"] = float(scipy_stats.skew(logprobs))
            summary["logprob_statistics"]["kurtosis"] = float(scipy_stats.kurtosis(logprobs))
        except (ImportError, Exception):
            pass
        
        return summary
    
    def identify_outliers(self, response: Any, method: str = "iqr") -> Optional[List[Dict[str, Any]]]:
        """
        Identify outlier tokens based on statistical methods.
        
        Args:
            response: OpenAI completion response with logprobs
            method: Method to use for outlier detection ("iqr", "zscore")
        
        Returns:
            List of outlier token information or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in tokens]
        outliers = []
        
        if method == "iqr":
            # Interquartile Range method
            q25, q75 = np.percentile(logprobs, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
                if logprob < lower_bound or logprob > upper_bound:
                    token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                    outliers.append({
                        "position": i + 1,
                        "token": token_str,
                        "logprob": logprob,
                        "probability_percent": exp(logprob) * 100,
                        "outlier_type": "low" if logprob < lower_bound else "high"
                    })
        
        elif method == "zscore":
            # Z-score method (|z| > 2)
            mean_logprob = np.mean(logprobs)
            std_logprob = np.std(logprobs)
            
            if std_logprob > 0:
                for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
                    z_score = abs((logprob - mean_logprob) / std_logprob)
                    if z_score > 2:
                        token_str = bytes(token.bytes).decode("utf-8", errors="replace")
                        outliers.append({
                            "position": i + 1,
                            "token": token_str,
                            "logprob": logprob,
                            "probability_percent": exp(logprob) * 100,
                            "z_score": float(z_score),
                            "outlier_type": "statistical"
                        })
        
        return outliers
    
    def calculate_consistency_score(self, response: Any) -> Optional[float]:
        """
        Calculate consistency score based on how stable the model's confidence is.
        Lower standard deviation indicates more consistent predictions.
        
        Args:
            response: OpenAI completion response with logprobs
        
        Returns:
            Consistency score (0-1, higher is more consistent) or None if no logprobs available
        """
        if not response or not response.choices[0].logprobs:
            return None
        
        tokens = response.choices[0].logprobs.content
        probabilities = [exp(token.logprob) for token in tokens]
        
        if len(probabilities) < 2:
            return None
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        if mean_prob == 0:
            return None
        
        cv = std_prob / mean_prob
        # Convert to 0-1 score where 1 is most consistent
        consistency_score = 1 / (1 + cv)
        
        return float(consistency_score)
