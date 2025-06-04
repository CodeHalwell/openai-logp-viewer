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
            min_logprob: Minimum logprob for normalization (ignored for absolute mapping)
            max_logprob: Maximum logprob for normalization (ignored for absolute mapping)
            scheme: Color scheme to use
        
        Returns:
            RGB color string
        """
        # Convert logprob to probability percentage for absolute mapping
        from math import exp
        probability = exp(logprob) * 100
        
        # Map probability to 0-1 scale with absolute thresholds and better high-end differentiation
        if probability >= 95:
            normalized = 1.0   # Extremely high confidence
        elif probability >= 85:
            normalized = 0.9   # Very high confidence
        elif probability >= 70:
            normalized = 0.8   # High confidence
        elif probability >= 55:
            normalized = 0.7   # Medium-high confidence
        elif probability >= 40:
            normalized = 0.6   # Medium confidence
        elif probability >= 25:
            normalized = 0.45  # Medium-low confidence
        elif probability >= 15:
            normalized = 0.3   # Low-medium confidence
        elif probability >= 8:
            normalized = 0.2   # Low confidence
        elif probability >= 3:
            normalized = 0.1   # Very low confidence
        else:
            normalized = 0.0   # Extremely low confidence
        
        # Get color from selected scheme
        color_func = self.schemes.get(scheme, self._confidence_scheme)
        return color_func(normalized)
    
    def _confidence_scheme(self, normalized_value: float) -> str:
        """
        Enhanced confidence-based color scheme with fine gradients for 10% probability distinctions.
        Maps logprob values to colors with much finer granularity for better visual distinction.
        
        Args:
            normalized_value: Value between 0 and 1 (normalized logprob)
        
        Returns:
            RGB color string with fine gradient
        """
        # Use a much more granular approach with smoother transitions
        # This creates distinct colors for every ~10% probability difference
        
        if normalized_value <= 0.1:
            # Very low confidence: Deep red
            red = 220
            green = 50
            blue = 50
        elif normalized_value <= 0.2:
            # Low confidence: Red to orange-red
            t = (normalized_value - 0.1) / 0.1
            red = 220 + int(35 * t)
            green = 50 + int(70 * t)
            blue = 50
        elif normalized_value <= 0.3:
            # Low-medium confidence: Orange-red to orange
            t = (normalized_value - 0.2) / 0.1
            red = 255
            green = 120 + int(60 * t)
            blue = 50
        elif normalized_value <= 0.4:
            # Medium-low confidence: Orange to yellow-orange
            t = (normalized_value - 0.3) / 0.1
            red = 255
            green = 180 + int(50 * t)
            blue = 50 + int(70 * t)
        elif normalized_value <= 0.5:
            # Medium confidence: Yellow-orange to yellow
            t = (normalized_value - 0.4) / 0.1
            red = 255
            green = 230 + int(25 * t)
            blue = 120 + int(50 * t)
        elif normalized_value <= 0.6:
            # Medium confidence (40-55%): Yellow to yellow-green
            t = (normalized_value - 0.5) / 0.1
            red = 255 - int(40 * t)
            green = 255
            blue = 170 - int(30 * t)
        elif normalized_value <= 0.7:
            # Medium-high confidence (55-70%): Yellow-green to lime
            t = (normalized_value - 0.6) / 0.1
            red = 215 - int(50 * t)
            green = 255
            blue = 140 - int(30 * t)
        elif normalized_value <= 0.8:
            # High confidence (70-85%): Lime to light green
            t = (normalized_value - 0.7) / 0.1
            red = 165 - int(50 * t)
            green = 255
            blue = 110 - int(30 * t)
        elif normalized_value <= 0.9:
            # Very high confidence (85-95%): Light green to green
            t = (normalized_value - 0.8) / 0.1
            red = 115 - int(40 * t)
            green = 255 - int(30 * t)
            blue = 80 - int(40 * t)
        else:
            # Extremely high confidence (95%+): Deep green
            red = 75
            green = 225
            blue = 40
        
        # Ensure RGB values are within valid range
        red = max(0, min(255, red))
        green = max(0, min(255, green))
        blue = max(0, min(255, blue))
        
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
