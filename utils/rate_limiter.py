"""
Rate Limiter for OpenAI API Protection
Implements comprehensive rate limiting to prevent abuse of OpenAI tokens.
"""

import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib
import json

class RateLimiter:
    """Comprehensive rate limiting system for OpenAI API protection."""
    
    def __init__(self):
        """Initialize rate limiter with configurable limits."""
        # Rate limits (configurable)
        self.REQUESTS_PER_MINUTE = 10
        self.TOKENS_PER_MINUTE = 2000
        self.TOKENS_PER_DAY = 50000
        self.MAX_TOKENS_PER_REQUEST = 500
        
        # Initialize session state for tracking
        if "rate_limiter" not in st.session_state:
            st.session_state.rate_limiter = {
                "requests": [],
                "tokens_used": [],
                "daily_tokens": 0,
                "daily_reset_date": datetime.now().date().isoformat()
            }
    
    def get_client_id(self) -> str:
        """Generate a unique client identifier based on session."""
        # Use session ID and some browser info for client identification
        session_info = f"{st.session_state.get('session_id', 'unknown')}"
        return hashlib.sha256(session_info.encode()).hexdigest()[:16]
    
    def cleanup_old_records(self) -> None:
        """Remove old tracking records to prevent memory buildup."""
        now = time.time()
        one_minute_ago = now - 60
        one_day_ago = now - (24 * 60 * 60)
        
        # Clean up request timestamps older than 1 minute
        st.session_state.rate_limiter["requests"] = [
            req_time for req_time in st.session_state.rate_limiter["requests"]
            if req_time > one_minute_ago
        ]
        
        # Clean up token usage older than 1 minute
        st.session_state.rate_limiter["tokens_used"] = [
            token_record for token_record in st.session_state.rate_limiter["tokens_used"]
            if token_record["timestamp"] > one_minute_ago
        ]
        
        # Reset daily counter if it's a new day
        current_date = datetime.now().date().isoformat()
        if st.session_state.rate_limiter["daily_reset_date"] != current_date:
            st.session_state.rate_limiter["daily_tokens"] = 0
            st.session_state.rate_limiter["daily_reset_date"] = current_date
    
    def check_request_limit(self) -> Dict[str, Any]:
        """Check if request rate limit is exceeded."""
        self.cleanup_old_records()
        
        current_requests = len(st.session_state.rate_limiter["requests"])
        
        if current_requests >= self.REQUESTS_PER_MINUTE:
            return {
                "allowed": False,
                "reason": "request_rate",
                "current": current_requests,
                "limit": self.REQUESTS_PER_MINUTE,
                "reset_in": 60 - (time.time() - min(st.session_state.rate_limiter["requests"]))
            }
        
        return {"allowed": True}
    
    def check_token_limits(self, requested_tokens: int) -> Dict[str, Any]:
        """Check token rate limits (per minute and per day)."""
        self.cleanup_old_records()
        
        # Check tokens per minute
        current_minute_tokens = sum(
            record["tokens"] for record in st.session_state.rate_limiter["tokens_used"]
        )
        
        if current_minute_tokens + requested_tokens > self.TOKENS_PER_MINUTE:
            return {
                "allowed": False,
                "reason": "token_rate_minute",
                "current": current_minute_tokens,
                "requested": requested_tokens,
                "limit": self.TOKENS_PER_MINUTE,
                "reset_in": 60
            }
        
        # Check tokens per day
        daily_tokens = st.session_state.rate_limiter["daily_tokens"]
        if daily_tokens + requested_tokens > self.TOKENS_PER_DAY:
            return {
                "allowed": False,
                "reason": "token_rate_daily",
                "current": daily_tokens,
                "requested": requested_tokens,
                "limit": self.TOKENS_PER_DAY,
                "reset_in": self._get_seconds_until_midnight()
            }
        
        # Check single request token limit
        if requested_tokens > self.MAX_TOKENS_PER_REQUEST:
            return {
                "allowed": False,
                "reason": "token_per_request",
                "requested": requested_tokens,
                "limit": self.MAX_TOKENS_PER_REQUEST
            }
        
        return {"allowed": True}
    
    def record_request(self, tokens_used: int) -> None:
        """Record a successful API request."""
        now = time.time()
        
        # Record request timestamp
        st.session_state.rate_limiter["requests"].append(now)
        
        # Record token usage
        st.session_state.rate_limiter["tokens_used"].append({
            "timestamp": now,
            "tokens": tokens_used
        })
        
        # Update daily token count
        st.session_state.rate_limiter["daily_tokens"] += tokens_used
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for display."""
        self.cleanup_old_records()
        
        current_requests = len(st.session_state.rate_limiter["requests"])
        current_minute_tokens = sum(
            record["tokens"] for record in st.session_state.rate_limiter["tokens_used"]
        )
        daily_tokens = st.session_state.rate_limiter["daily_tokens"]
        
        return {
            "requests_per_minute": {
                "current": current_requests,
                "limit": self.REQUESTS_PER_MINUTE,
                "percentage": (current_requests / self.REQUESTS_PER_MINUTE) * 100
            },
            "tokens_per_minute": {
                "current": current_minute_tokens,
                "limit": self.TOKENS_PER_MINUTE,
                "percentage": (current_minute_tokens / self.TOKENS_PER_MINUTE) * 100
            },
            "tokens_per_day": {
                "current": daily_tokens,
                "limit": self.TOKENS_PER_DAY,
                "percentage": (daily_tokens / self.TOKENS_PER_DAY) * 100
            }
        }
    
    def _get_seconds_until_midnight(self) -> int:
        """Calculate seconds until midnight for daily reset."""
        now = datetime.now()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int((midnight - now).total_seconds())
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text with improved accuracy."""
        if not text:
            return 0
        
        # More accurate token estimation based on OpenAI's patterns
        # Split by common token boundaries
        import re
        
        # Count words, punctuation, and special characters separately
        words = re.findall(r'\b\w+\b', text)
        punctuation = re.findall(r'[^\w\s]', text)
        whitespace_chunks = re.findall(r'\s+', text)
        
        # Estimate: words ≈ 0.75 tokens, punctuation ≈ 1 token each, whitespace chunks ≈ 0.25 tokens
        estimated_tokens = int(len(words) * 0.75 + len(punctuation) + len(whitespace_chunks) * 0.25)
        
        # Add 20% buffer for safety and account for subword tokenization
        return max(1, int(estimated_tokens * 1.2))
    
    def check_and_record_request(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """
        Comprehensive check before making API request.
        
        Args:
            prompt: User prompt text
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with check results
        """
        # Estimate input tokens
        input_tokens = self.estimate_tokens(prompt)
        total_estimated_tokens = input_tokens + max_tokens
        
        # Check request rate limit
        request_check = self.check_request_limit()
        if not request_check["allowed"]:
            return request_check
        
        # Check token limits
        token_check = self.check_token_limits(total_estimated_tokens)
        if not token_check["allowed"]:
            return token_check
        
        return {
            "allowed": True,
            "estimated_tokens": total_estimated_tokens,
            "input_tokens": input_tokens,
            "max_output_tokens": max_tokens
        }
    
    def display_rate_limit_info(self) -> None:
        """Display current rate limit status in Streamlit UI."""
        status = self.get_rate_limit_status()
        
        st.subheader("🛡️ Usage Limits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            req_pct = status["requests_per_minute"]["percentage"]
            color = "red" if req_pct > 80 else "orange" if req_pct > 60 else "green"
            st.metric(
                "Requests/min",
                f"{status['requests_per_minute']['current']}/{status['requests_per_minute']['limit']}",
                delta=f"{req_pct:.0f}%"
            )
            st.progress(min(req_pct / 100, 1.0))
        
        with col2:
            token_min_pct = status["tokens_per_minute"]["percentage"]
            color = "red" if token_min_pct > 80 else "orange" if token_min_pct > 60 else "green"
            st.metric(
                "Tokens/min",
                f"{status['tokens_per_minute']['current']}/{status['tokens_per_minute']['limit']}",
                delta=f"{token_min_pct:.0f}%"
            )
            st.progress(min(token_min_pct / 100, 1.0))
        
        with col3:
            daily_pct = status["tokens_per_day"]["percentage"]
            color = "red" if daily_pct > 80 else "orange" if daily_pct > 60 else "green"
            st.metric(
                "Tokens/day",
                f"{status['tokens_per_day']['current']}/{status['tokens_per_day']['limit']}",
                delta=f"{daily_pct:.0f}%"
            )
            st.progress(min(daily_pct / 100, 1.0))
        
        # Show warnings if approaching limits
        if req_pct > 80 or token_min_pct > 80:
            st.warning("⚠️ Approaching rate limits. Please wait a moment before making more requests.")
        
        if daily_pct > 90:
            st.error("🚨 Daily token limit almost reached. Consider reducing request size or waiting until tomorrow.")