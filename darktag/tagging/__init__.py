import warnings

try:
    from .tagging_wrapper_func import *
except ImportError as e:
    warnings.warn(f"tagging_wrapper_func could not be loaded: {e}")

try:
    from .spatial_tagging import *
except ImportError as e:
    warnings.warn(f"spatial_tagging could not be loaded: {e}")

try:
    from .angular_momentum_tagging import *
except ImportError as e:
    warnings.warn(f"angular_momentum_tagging could not be loaded: {e}")

try:
    from .binding_energy_tagging import *
except ImportError as e:
    warnings.warn(f"binding_energy_tagging could not be loaded: {e}")

try:
    from .utils import *
except ImportError as e:
    warnings.warn(f"utils could not be loaded: {e}")

try:
    from .clustering import *
except ImportError as e:
    warnings.warn(f"clustering could not be loaded: {e}")
