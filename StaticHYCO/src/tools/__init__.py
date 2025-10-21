# src/models/__init__.py

from .finite_element_method import *
# Re-export training utilities
from .training import (
    format_params,
    compute_param_error,
    vmapped_model,
    train_step_data_only,
    train_step_hybrid,
    ModelTrainer,
    HybridTrainer,
    FEMTrainer,
    PINNTrainer,
)
from .experiment_utils import (
    replace_zeros_linear,
    replace_zeros_nearest,
    save_results_generic,
)