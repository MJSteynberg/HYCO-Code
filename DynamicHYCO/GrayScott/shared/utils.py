"""
Backward-compatible utilities for Gray-Scott models.
This module provides wrapper functions for legacy code compatibility.
All new code should import directly from the specific modules:
- data_loader: Data loading functions
- model_loader: Model loading functions  
- visualization: Plotting functions
"""

# Import from new organized modules
from shared.data_loader import (
    load_tuple_data,
    load_true_data_at_times,
    evaluate_hybrid_models
)

from shared.model_loader import (
    SimpleScaler,
    load_pinn_model_and_scalers,
    evaluate_pinn_model,
    load_hybrid_models,
    load_experiment_models
)


# Legacy function names for backward compatibility
# These are kept for any external scripts that may still use old import names

def load_model_and_scalers(model_dir):
    """
    DEPRECATED: Use load_pinn_model_and_scalers from shared.model_loader instead.
    Load a saved PINN model and its associated scalers.
    """
    return load_pinn_model_and_scalers(model_dir)


def load_trained_models(model_path, device, create_physical_model_func=None, neural_model_class=None):
    """
    DEPRECATED: Use load_hybrid_models from shared.model_loader instead.
    Load previously trained physical and neural models.
    """
    return load_hybrid_models(model_path, device)


# Export all public functions for backward compatibility
__all__ = [
    # Data loading
    'load_tuple_data',
    'load_true_data_at_times',
    'evaluate_hybrid_models',
    # Model loading
    'SimpleScaler',
    'load_pinn_model_and_scalers',
    'load_model_and_scalers',  # Legacy
    'evaluate_pinn_model',
    'load_hybrid_models',
    'load_trained_models',  # Legacy
    'load_experiment_models',
]
