"""
Model utility functions for Gray-Scott HYCO training.

This module provides helper functions for model operations including:
- Model creation and configuration
- Spatial interpolation
- Solution conversion and evaluation
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from shared.model import GrayScottModel


def create_training_model(width, height, domain_width, domain_height,
                          target_f, target_k, loaded_params, time_points, device, 
                          initial_Du=0.2, initial_Dv=0.08):
    """
    Create and configure a GrayScottModel for training.
    
    Args:
        width: Grid width for training model
        height: Grid height for training model
        domain_width: Physical domain width
        domain_height: Physical domain height
        target_f: Feed rate parameter
        target_k: Kill rate parameter
        loaded_params: Parameters from loaded data
        time_points: Array of time points
        device: PyTorch device ('cpu' or 'cuda')
        initial_Du: Initial diffusion coefficient for U
        initial_Dv: Initial diffusion coefficient for V
        
    Returns:
        GrayScottModel: Configured model ready for training
    """
    model = GrayScottModel(
        width=width, height=height, device=device,
        domain_width=domain_width, domain_height=domain_height,
        Du=initial_Du, Dv=initial_Dv, f=target_f, k=target_k
    )

    # Set dt to match data generation
    total_time = time_points[-1]
    data_total_steps = loaded_params['steps']
    data_width = loaded_params['width']

    model.dt = (data_width / width)**2 * total_time / data_total_steps

    return model


def interpolate_to_coordinates(field, x_coords, y_coords, x_grid, y_grid):
    """
    Interpolate 2D field to given coordinates using bilinear interpolation.
    
    Args:
        field: 2D PyTorch tensor [height, width] to interpolate
        x_coords: Target x coordinates
        y_coords: Target y coordinates
        x_grid: 1D tensor of x grid coordinates
        y_grid: 1D tensor of y grid coordinates
        
    Returns:
        torch.Tensor: Interpolated values at target coordinates
    """
    device = field.device
    height, width = field.shape

    # Normalize coordinates to [0, width-1] and [0, height-1]
    x_indices = (x_coords / x_grid[-1]) * (width - 1)
    y_indices = (y_coords / y_grid[-1]) * (height - 1)

    # Clamp to valid range
    x_indices = torch.clamp(x_indices, 0, width - 1)
    y_indices = torch.clamp(y_indices, 0, height - 1)

    # Get integer and fractional parts
    x_floor = torch.floor(x_indices).long()
    y_floor = torch.floor(y_indices).long()
    x_ceil = torch.clamp(x_floor + 1, 0, width - 1)
    y_ceil = torch.clamp(y_floor + 1, 0, height - 1)

    x_frac = x_indices - x_floor.float()
    y_frac = y_indices - y_floor.float()

    # Bilinear interpolation
    val_ll = field[y_floor, x_floor]
    val_lr = field[y_floor, x_ceil]
    val_ul = field[y_ceil, x_floor]
    val_ur = field[y_ceil, x_ceil]

    val_lower = val_ll + x_frac * (val_lr - val_ll)
    val_upper = val_ul + x_frac * (val_ur - val_ul)
    interpolated = val_lower + y_frac * (val_upper - val_lower)

    return interpolated


def convert_physical_solution_to_tuples(model, time_points, domain_width, domain_height, precomputed_grid=None):
    """
    Convert physical model solution to (x,y,t,u,v) tuple format.
    
    Args:
        model: GrayScottModel instance
        time_points: Array or tensor of time points to evaluate
        domain_width: Physical domain width
        domain_height: Physical domain height
        precomputed_grid: Optional tuple of (X, Y, x_flat, y_flat) to avoid recomputation
        
    Returns:
        torch.Tensor: Tensor of shape [N, 5] containing (x, y, t, u, v) tuples
    """
    device = model.U.device
    width, height = model.width, model.height

    if precomputed_grid is not None:
        X, Y, x_flat, y_flat = precomputed_grid
    else:
        x_coords = torch.linspace(0, domain_width, width, device=device)
        y_coords = torch.linspace(0, domain_height, height, device=device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        x_flat = X.flatten()
        y_flat = Y.flatten()

    # Run simulation and collect states
    model.reset_state()
    current_step = 0
    all_tuples = []

    for target_time in time_points:
        target_step = int(target_time / model.dt)
        steps_to_simulate = target_step - current_step

        if steps_to_simulate > 0:
            model.forward(steps=steps_to_simulate)
            current_step = target_step

        U_field = model.U.squeeze()
        V_field = model.V.squeeze()

        u_flat = U_field.flatten()
        v_flat = V_field.flatten()
        t_flat = torch.full_like(x_flat, target_time)

        tuples = torch.stack([x_flat, y_flat, t_flat, u_flat, v_flat], dim=1)
        all_tuples.append(tuples)

    return torch.cat(all_tuples, dim=0)


def compute_mse_vs_true_solution(physical_model, neural_model, time_points_tensor,
                                 model_params, true_solution_tuples, precomputed_grid):
    """
    Compute MSE between models and true solution on full time-space grid.
    
    Args:
        physical_model: Trained GrayScottModel
        neural_model: Trained neural network model
        time_points_tensor: Tensor of time points
        model_params: Dictionary of model parameters
        true_solution_tuples: Ground truth solution tuples
        precomputed_grid: Precomputed spatial grid
        
    Returns:
        tuple: (physical_mse, neural_mse) - MSE values for both models
    """
    physical_tuples = convert_physical_solution_to_tuples(
        physical_model, time_points_tensor, model_params['domain_width'], model_params['domain_height'],
        precomputed_grid=precomputed_grid
    )

    true_u_values = true_solution_tuples[:, 3]
    true_v_values = true_solution_tuples[:, 4]

    physical_u_pred = physical_tuples[:, 3]
    physical_v_pred = physical_tuples[:, 4]

    physical_mse_u = F.mse_loss(physical_u_pred, true_u_values)
    physical_mse_v = F.mse_loss(physical_v_pred, true_v_values)
    physical_mse_total = physical_mse_u + physical_mse_v

    true_x_coords = true_solution_tuples[:, 0]
    true_y_coords = true_solution_tuples[:, 1]
    true_t_coords = true_solution_tuples[:, 2]

    with torch.no_grad():
        neural_u_pred, neural_v_pred = neural_model(true_x_coords, true_y_coords, true_t_coords)

    neural_mse_u = F.mse_loss(neural_u_pred, true_u_values)
    neural_mse_v = F.mse_loss(neural_v_pred, true_v_values)
    neural_mse_total = neural_mse_u + neural_mse_v

    return physical_mse_total.item(), neural_mse_total.item()


def save_trained_models(physical_model, neural_model, training_params, save_dir="saved_models"):
    """
    Save trained physical and neural models with their parameters.
    
    Args:
        physical_model: Trained GrayScottModel
        neural_model: Trained neural network
        training_params: Dictionary of training parameters and history
        save_dir: Directory to save models
        
    Returns:
        str: Path to saved models directory
    """
    model_save_path = save_dir
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Saving models to: {model_save_path}")

    # Save physical model
    physical_params = physical_model.get_trainable_parameters()
    physical_state = {
        'trainable_params': physical_params,
        'log_Du': physical_model.log_Du.detach().cpu(),
        'log_Dv': physical_model.log_Dv.detach().cpu(),
        'width': physical_model.width,
        'height': physical_model.height,
        'domain_width': physical_model.domain_width,
        'domain_height': physical_model.domain_height,
        'dt': physical_model.dt,
        'f': physical_model.f.item(),
        'k': physical_model.k.item()
    }
    torch.save(physical_state, os.path.join(model_save_path, "physical_model.pt"))

    # Save neural model
    neural_params = training_params.get('neural_params', {'hidden_size': 64, 'num_layers': 3})
    neural_state = {
        'state_dict': neural_model.state_dict(),
        'hidden_size': neural_params['hidden_size'],
        'num_layers': neural_params['num_layers'],
        'normalization': {
            't_mean': neural_model.t_mean if hasattr(neural_model, 't_mean') else None,
            't_std': neural_model.t_std if hasattr(neural_model, 't_std') else None,
            'x_mean': neural_model.x_mean if hasattr(neural_model, 'x_mean') else None,
            'x_std': neural_model.x_std if hasattr(neural_model, 'x_std') else None,
            'y_mean': neural_model.y_mean if hasattr(neural_model, 'y_mean') else None,
            'y_std': neural_model.y_std if hasattr(neural_model, 'y_std') else None,
        }
    }
    torch.save(neural_state, os.path.join(model_save_path, "neural_model.pt"))

    # Save training parameters
    with open(os.path.join(model_save_path, "training_params.json"), 'w') as f:
        json.dump(training_params, f, indent=2)

    # Create summary
    summary = {
        'final_physical_params': physical_params,
        'training_config': training_params,
        'files': ['physical_model.pt', 'neural_model.pt', 'training_params.json']
    }
    with open(os.path.join(model_save_path, "model_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Models saved successfully!")
    return model_save_path


__all__ = [
    'create_training_model',
    'interpolate_to_coordinates',
    'convert_physical_solution_to_tuples',
    'compute_mse_vs_true_solution',
    'save_trained_models'
]
