"""
Darktag: Particle tagging for dark matter simulations.

A Python package for assigning stellar mass to dark matter particles 
in dark matter-only simulations using advanced particle tagging methods.
"""

__version__ = "1.0.0"

# Try to import optional dependencies
try:
    from .analysis import *
except ImportError as e:
    if "pynbody" in str(e) or "tangos" in str(e):
        print("Warning: Analysis module requires astrophysics dependencies.")
        print("Install with: pip install darktag[astrophysics]")
    else:
        raise

try:
    from .tagging import *
except ImportError as e:
    if "pynbody" in str(e) or "tangos" in str(e) or "darklight" in str(e):
        print("Warning: Tagging module requires astrophysics dependencies.")
        print("Install with: pip install darktag[astrophysics]")
    else:
        raise