"""
Data loading utilities for Gray-Scott models.
Handles loading training data, timeseries data, and ground truth.
"""

import os
import numpy as np
from typing import Dict, List
import torch


def load_tuple_data(data_dir: str) -> Dict:
    """
    Load tuple-based simulation data from a directory.
    
    Args:
        data_dir: Directory containing data_tuples.npy, parameters.npy, and time_points.npy
        
    Returns:
        Dictionary with keys:
            - 'data_tuples': numpy array [N, 5] with columns [u, v, x, y, t]
            - 'parameters': dict of simulation parameters (if available)
            - 'time_points': numpy array of time points (if available)
    """
    print(f"Loading tuple data from: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load data tuples (required)
    data_tuples_path = os.path.join(data_dir, 'data_tuples.npy')
    if not os.path.exists(data_tuples_path):
        raise FileNotFoundError(f"data_tuples.npy not found in {data_dir}")
    
    data_tuples = np.load(data_tuples_path)
    
    # Validate data format
    if data_tuples.shape[1] != 5:
        raise ValueError(f"data_tuples must have 5 columns [u,v,x,y,t], got shape {data_tuples.shape}")
    
    print(f"Loaded data tuples with shape: {data_tuples.shape}")
    
    # Load parameters (optional)
    params_path = os.path.join(data_dir, 'parameters.npy')
    parameters = np.load(params_path, allow_pickle=True).item() if os.path.exists(params_path) else {}
    
    # Load time points (optional)
    time_points_path = os.path.join(data_dir, 'time_points.npy')
    time_points = np.load(time_points_path) if os.path.exists(time_points_path) else None
    
    return {
        'data_tuples': data_tuples,
        'parameters': parameters,
        'time_points': time_points
    }


def load_true_data_at_times(data_dir: str, target_times: List[float]) -> Dict:
    """
    Load ground truth data at specific time points.
    
    Args:
        data_dir: Directory containing U_timeseries.npy, V_timeseries.npy, etc.
        target_times: List of time points to load
        
    Returns:
        Dictionary mapping time -> {'u', 'v', 'x_coords', 'y_coords', 'actual_time'}
    """
    print("Loading ground truth data...")

    U_timeseries = np.load(os.path.join(data_dir, 'U_timeseries.npy'))
    V_timeseries = np.load(os.path.join(data_dir, 'V_timeseries.npy'))
    time_points = np.load(os.path.join(data_dir, 'time_points.npy'))
    params = np.load(os.path.join(data_dir, 'parameters.npy'), allow_pickle=True).item()

    x_coords = np.linspace(0, params['domain_width'], params['width'])
    y_coords = np.linspace(0, params['domain_height'], params['height'])

    true_data = {}
    for target_time in target_times:
        time_idx = np.argmin(np.abs(time_points - target_time))
        actual_time = time_points[time_idx]

        true_data[target_time] = {
            'u': U_timeseries[time_idx],
            'v': V_timeseries[time_idx],
            'x_coords': x_coords,
            'y_coords': y_coords,
            'actual_time': actual_time
        }
        print(f"Loaded data for t={target_time} (actual: {actual_time:.1f})")

    return true_data


def evaluate_hybrid_models(
    physical_model, 
    neural_model, 
    data_dir: str, 
    target_times: List[float]
) -> Dict:
    """
    Evaluate both hybrid-trained models at specific time points.
    
    Args:
        physical_model: GrayScottModel instance
        neural_model: SimpleNeuralModel instance
        data_dir: Directory containing parameter data
        target_times: List of time points to evaluate
        
    Returns:
        Dictionary with 'physical' and 'neural' predictions
    """
    print("Evaluating hybrid trained models...")

    device = next(neural_model.parameters()).device

    # Load parameters for spatial grid
    params = np.load(os.path.join(data_dir, 'parameters.npy'), allow_pickle=True).item()

    # Create spatial coordinate grids (on CPU for numpy operations later)
    x_coords_np = np.linspace(0, params['domain_width'], params['width'])
    y_coords_np = np.linspace(0, params['domain_height'], params['height'])
    X_np, Y_np = np.meshgrid(x_coords_np, y_coords_np)

    # Generate predictions from both models
    predictions = {'physical': {}, 'neural': {}}

    # Physical model predictions
    print("Generating physical model predictions...")
    physical_model.to(device)
    physical_model.reset_state()
    current_step = 0

    with torch.no_grad():
        for target_time in target_times:
            target_step = int(target_time / physical_model.dt)
            steps_to_simulate = target_step - current_step

            if steps_to_simulate > 0:
                physical_model.forward(steps=steps_to_simulate)
                current_step = target_step

            U_field_gpu = physical_model.U.squeeze()
            V_field_gpu = physical_model.V.squeeze()

            U_field = U_field_gpu.cpu().numpy().copy()
            V_field = V_field_gpu.cpu().numpy().copy()

            predictions['physical'][target_time] = {
                'u': U_field, 'v': V_field,
                'x_coords': x_coords_np, 'y_coords': y_coords_np
            }

    # Neural model predictions
    print("Generating neural model predictions...")
    neural_model.to(device)
    neural_model.eval()
    
    for target_time in target_times:
        x_flat_gpu = torch.tensor(X_np.flatten(), dtype=torch.float32).to(device)
        y_flat_gpu = torch.tensor(Y_np.flatten(), dtype=torch.float32).to(device)
        t_flat_gpu = torch.tensor(
            np.full(x_flat_gpu.shape, target_time), 
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred_u_flat_gpu, pred_v_flat_gpu = neural_model(
                x_flat_gpu, y_flat_gpu, t_flat_gpu
            )

        pred_u = pred_u_flat_gpu.cpu().numpy().reshape(params['height'], params['width'])
        pred_v = pred_v_flat_gpu.cpu().numpy().reshape(params['height'], params['width'])

        predictions['neural'][target_time] = {
            'u': pred_u, 'v': pred_v,
            'x_coords': x_coords_np, 'y_coords': y_coords_np
        }

    return predictions
