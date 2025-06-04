"""
Color Schemes for Token Confidence Visualization
Provides various color schemes for highlighting tokens based on logprob values.
"""

import numpy as np
from typing import Tuple

class ColorSchemeManager:
    """Manages different color schemes for token confidence visualization."""
    
    def __init__(self):
        """Initialize color scheme manager with available schemes."""
        self.schemes = {
            'confidence': self._confidence_scheme,
            'rainbow': self._rainbow_scheme,
            'heat': self._heat_scheme,
            'ocean': self._ocean_scheme,
            'monochrome': self._monochrome_scheme,
            'pastel': self._pastel_scheme
        }
    
    def get_color(self, logprob: float, min_logprob: float, max_logprob: float, scheme: str = 'confidence') -> str:
        """
        Get color for a token based on its logprob value.
        
        Args:
            logprob: Log probability value
            min_logprob: Minimum logprob for normalization
            max_logprob: Maximum logprob for normalization
            scheme: Color scheme to use
        
        Returns:
            RGB color string
        """
        # Normalize logprob to 0-1 scale
        if max_logprob == min_logprob:
            normalized = 0.5  # Default to middle value if no variation
        else:
            normalized = (logprob - min_logprob) / (max_logprob - min_logprob)
        
        normalized = max(0, min(1, normalized))  # Clamp to 0-1
        
        # Get color from selected scheme
        color_func = self.schemes.get(scheme, self._confidence_scheme)
        return color_func(normalized)
    
    def _confidence_scheme(self, normalized_value: float) -> str:
        """
        Confidence-based color scheme: Red (low) to Green (high) through Yellow.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        if normalized_value < 0.5:
            # Red to yellow
            red = 255
            green = int(255 * normalized_value * 2)
            blue = 0
        else:
            # Yellow to green
            red = int(255 * (1 - normalized_value) * 2)
            green = 255
            blue = 0
        
        return f"rgb({red}, {green}, {blue})"
    
    def _rainbow_scheme(self, normalized_value: float) -> str:
        """
        Rainbow color scheme across the spectrum.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        # Convert to HSV and then to RGB
        hue = normalized_value * 300  # 0 to 300 degrees (red to purple)
        saturation = 0.8
        value = 0.9
        
        # HSV to RGB conversion
        c = value * saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = value - c
        
        if 0 <= hue < 60:
            rgb = (c, x, 0)
        elif 60 <= hue < 120:
            rgb = (x, c, 0)
        elif 120 <= hue < 180:
            rgb = (0, c, x)
        elif 180 <= hue < 240:
            rgb = (0, x, c)
        elif 240 <= hue < 300:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        
        red = int((rgb[0] + m) * 255)
        green = int((rgb[1] + m) * 255)
        blue = int((rgb[2] + m) * 255)
        
        return f"rgb({red}, {green}, {blue})"
    
    def _heat_scheme(self, normalized_value: float) -> str:
        """
        Heat map color scheme: Black to Red to Yellow to White.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        if normalized_value < 0.33:
            # Black to red
            t = normalized_value / 0.33
            red = int(255 * t)
            green = 0
            blue = 0
        elif normalized_value < 0.66:
            # Red to yellow
            t = (normalized_value - 0.33) / 0.33
            red = 255
            green = int(255 * t)
            blue = 0
        else:
            # Yellow to white
            t = (normalized_value - 0.66) / 0.34
            red = 255
            green = 255
            blue = int(255 * t)
        
        return f"rgb({red}, {green}, {blue})"
    
    def _ocean_scheme(self, normalized_value: float) -> str:
        """
        Ocean color scheme: Deep blue to light cyan.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        # Deep blue to light cyan
        red = int(0 + normalized_value * 100)
        green = int(100 + normalized_value * 155)
        blue = int(139 + normalized_value * 116)
        
        # Ensure values are within RGB range
        red = min(255, max(0, red))
        green = min(255, max(0, green))
        blue = min(255, max(0, blue))
        
        return f"rgb({red}, {green}, {blue})"
    
    def _monochrome_scheme(self, normalized_value: float) -> str:
        """
        Monochrome color scheme: Black to white.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        intensity = int(255 * normalized_value)
        return f"rgb({intensity}, {intensity}, {intensity})"
    
    def _pastel_scheme(self, normalized_value: float) -> str:
        """
        Pastel color scheme: Soft, muted colors.
        
        Args:
            normalized_value: Value between 0 and 1
        
        Returns:
            RGB color string
        """
        # Use rainbow scheme but with reduced saturation for pastel effect
        hue = normalized_value * 300
        saturation = 0.3  # Low saturation for pastel effect
        value = 0.9
        
        # HSV to RGB conversion (same as rainbow but different saturation)
        c = value * saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = value - c
        
        if 0 <= hue < 60:
            rgb = (c, x, 0)
        elif 60 <= hue < 120:
            rgb = (x, c, 0)
        elif 120 <= hue < 180:
            rgb = (0, c, x)
        elif 180 <= hue < 240:
            rgb = (0, x, c)
        elif 240 <= hue < 300:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        
        red = int((rgb[0] + m) * 255)
        green = int((rgb[1] + m) * 255)
        blue = int((rgb[2] + m) * 255)
        
        return f"rgb({red}, {green}, {blue})"
    
    def get_available_schemes(self) -> list:
        """
        Get list of available color schemes.
        
        Returns:
            List of scheme names
        """
        return list(self.schemes.keys())
    
    def get_scheme_description(self, scheme: str) -> str:
        """
        Get description of a color scheme.
        
        Args:
            scheme: Scheme name
        
        Returns:
            Description string
        """
        descriptions = {
            'confidence': 'Red (low confidence) → Yellow → Green (high confidence)',
            'rainbow': 'Full spectrum rainbow colors',
            'heat': 'Heat map: Black → Red → Yellow → White',
            'ocean': 'Ocean blues: Deep blue → Light cyan',
            'monochrome': 'Grayscale: Black → White',
            'pastel': 'Soft pastel colors'
        }
        return descriptions.get(scheme, 'Custom color scheme')
