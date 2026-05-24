"""
Darktag: Particle tagging for dark matter simulations.

A Python package for assigning stellar mass to dark matter particles 
in dark matter-only simulations using advanced particle tagging methods.
"""

__version__ = "1.0.0"

import warnings

try:
    from .analysis import *
except ImportError:
    warnings.warn("Analysis module could not be loaded (missing astrophysics dependencies)")

try:
    from .tagging import *
except ImportError:
    warnings.warn("Tagging module could not be loaded (missing astrophysics dependencies)")