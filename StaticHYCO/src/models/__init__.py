# src/models/__init__.py

from .synthetic_model import SyntheticModel
from .physical_model import PhysicalModel
from .other_models import PINN


__all__ = ["SyntheticModel", "PhysicalModel"]