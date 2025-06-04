"""
Cache Manager for OpenAI Logprobs Application
Handles caching strategies and cache management functionality.
"""

import streamlit as st
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class CacheManager:
    """Manages caching for OpenAI API responses and application state."""
    
    def __init__(self):
        """Initialize cache manager."""
        if 'cache_stats' not in st.session_state:
            st.session_state.cache_stats = {
                'hits': 0,
                'misses': 0,
                'created': datetime.now().isoformat()
            }
    
    def create_cache_key(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """
        Create a unique cache key for API requests.
        
        Args:
            prompt: Input prompt text
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
        
        Returns:
            SHA256 hash as cache key
        """
        cache_data = {
            'prompt': prompt.strip(),
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict[Any, Any]]:
        """
        Retrieve cached response if available and not expired.
        
        Args:
            cache_key: Unique cache key
        
        Returns:
            Cached response or None if not found/expired
        """
        cache_store = getattr(st.session_state, 'response_cache', {})
        
        if cache_key in cache_store:
            cached_item = cache_store[cache_key]
            
            # Check if cache is still valid (1 hour TTL)
            cache_time = datetime.fromisoformat(cached_item['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=1):
                st.session_state.cache_stats['hits'] += 1
                return cached_item['response']
            else:
                # Remove expired cache
                del cache_store[cache_key]
        
        st.session_state.cache_stats['misses'] += 1
        return None
    
    def cache_response(self, cache_key: str, response: Any) -> None:
        """
        Cache an API response.
        
        Args:
            cache_key: Unique cache key
            response: OpenAI API response to cache
        """
        if not hasattr(st.session_state, 'response_cache'):
            st.session_state.response_cache = {}
        
        st.session_state.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Limit cache size to prevent memory issues
        if len(st.session_state.response_cache) > 50:
            # Remove oldest entries
            oldest_key = min(
                st.session_state.response_cache.keys(),
                key=lambda k: st.session_state.response_cache[k]['timestamp']
            )
            del st.session_state.response_cache[oldest_key]
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        if hasattr(st.session_state, 'response_cache'):
            st.session_state.response_cache.clear()
        
        # Reset cache stats
        st.session_state.cache_stats = {
            'hits': 0,
            'misses': 0,
            'created': datetime.now().isoformat()
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_store = getattr(st.session_state, 'response_cache', {})
        stats = st.session_state.cache_stats
        
        total_requests = stats['hits'] + stats['misses']
        hit_rate = (stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'count': len(cache_store),
            'hits': stats['hits'],
            'misses': stats['misses'],
            'hit_rate': round(hit_rate, 1),
            'created': stats['created']
        }
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        cache_store = getattr(st.session_state, 'response_cache', {})
        expired_keys = []
        
        for key, item in cache_store.items():
            cache_time = datetime.fromisoformat(item['timestamp'])
            if datetime.now() - cache_time >= timedelta(hours=1):
                expired_keys.append(key)
        
        for key in expired_keys:
            del cache_store[key]
        
        return len(expired_keys)
    
    def get_cache_size_mb(self) -> float:
        """
        Estimate cache size in megabytes.
        
        Returns:
            Estimated cache size in MB
        """
        cache_store = getattr(st.session_state, 'response_cache', {})
        
        try:
            # Rough estimation based on JSON serialization
            cache_json = json.dumps(cache_store, default=str)
            size_bytes = len(cache_json.encode('utf-8'))
            return round(size_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0
