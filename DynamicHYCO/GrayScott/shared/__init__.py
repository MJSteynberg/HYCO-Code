"""Shared Gray-Scott utilities exposed as a package."""

from .pinn import GrayScottPINN
from .model import GrayScottModel, SimpleNeuralModel
from .utils import load_model_and_scalers, load_tuple_data
from .error_tracking import ErrorTracker
from .data_generator import generate_data, check_data_exists, convert_to_tuples
from .training import hybrid_train, train_neural_minibatch, train_physical_with_neural_feedback, train_neural_lbfgs
from .model_utils import (
    create_training_model, 
    interpolate_to_coordinates, 
    convert_physical_solution_to_tuples,
    compute_mse_vs_true_solution,
    save_trained_models
)

__all__ = [
    "GrayScottPINN",
    "GrayScottModel",
    "SimpleNeuralModel",
    "load_model_and_scalers",
    "load_tuple_data",
    "ErrorTracker",
    "generate_data",
    "check_data_exists",
    "convert_to_tuples",
    "hybrid_train",
    "train_neural_minibatch",
    "train_physical_with_neural_feedback",
    "train_neural_lbfgs",
    "create_training_model",
    "interpolate_to_coordinates",
    "convert_physical_solution_to_tuples",
    "compute_mse_vs_true_solution",
    "save_trained_models",
]
