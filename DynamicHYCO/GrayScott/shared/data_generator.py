"""
Data generation utilities for Gray-Scott reaction-diffusion simulations.

This module provides functions for generating and managing Gray-Scott simulation data,
including stability checking, data conversion to tuple format, and data validation.
"""

import os
import numpy as np
import torch
from shared.model import GrayScottModel


def check_data_exists(output_dir, target_params):
    """
    Check if data already exists with matching parameters.
    
    Args:
        output_dir: Directory to check for existing data
        target_params: Dictionary of target parameters to validate against
        
    Returns:
        bool: True if compatible data exists, False otherwise
    """
    required_files = [
        'parameters.npy', 'U_timeseries.npy', 'V_timeseries.npy',
        'time_points.npy', 'data_tuples.npy'
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            print(f"Missing file: {file}")
            return False

    try:
        existing_params = np.load(os.path.join(output_dir, 'parameters.npy'), allow_pickle=True).item()

        essential_keys = [
            'width', 'height', 'domain_width', 'domain_height',
            'steps', 'save_interval', 'Du', 'Dv', 'f', 'k'
        ]

        for key in essential_keys:
            if key not in existing_params or key not in target_params:
                print(f"Missing parameter key: {key}")
                return False

            if existing_params[key] != target_params[key]:
                print(f"Parameter mismatch for {key}: existing={existing_params[key]}, target={target_params[key]}")
                return False

        print("✓ Compatible data found with matching parameters!")
        return True

    except Exception as e:
        print(f"Error checking existing data: {e}")
        return False


def convert_to_tuples(U_timeseries, V_timeseries, time_points, params):
    """
    Convert simulation data to (u,v,x,y,t) tuple format.
    
    Args:
        U_timeseries: Array of U field snapshots [num_snapshots, height, width]
        V_timeseries: Array of V field snapshots [num_snapshots, height, width]
        time_points: Array of time values for each snapshot
        params: Dictionary containing 'domain_width', 'domain_height', 'width', 'height'
        
    Returns:
        numpy.ndarray: Array of shape [N, 5] containing (u, v, x, y, t) tuples
    """
    num_snapshots = len(time_points)

    x_coords = np.linspace(0, params['domain_width'], params['width'])
    y_coords = np.linspace(0, params['domain_height'], params['height'])
    X, Y = np.meshgrid(x_coords, y_coords)

    u_data, v_data, x_data, y_data, t_data = [], [], [], [], []

    for t_idx in range(num_snapshots):
        current_time = time_points[t_idx]
        U_field = U_timeseries[t_idx]
        V_field = V_timeseries[t_idx]

        u_flat = U_field.flatten()
        v_flat = V_field.flatten()
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full(u_flat.shape, current_time)

        u_data.extend(u_flat)
        v_data.extend(v_flat)
        x_data.extend(x_flat)
        y_data.extend(y_flat)
        t_data.extend(t_flat)

    u_array = np.array(u_data, dtype=np.float32)
    v_array = np.array(v_data, dtype=np.float32)
    x_array = np.array(x_data, dtype=np.float32)
    y_array = np.array(y_data, dtype=np.float32)
    t_array = np.array(t_data, dtype=np.float32)

    data_tuples = np.column_stack((u_array, v_array, x_array, y_array, t_array))
    return data_tuples


def generate_data(params, output_dir='data', force_regenerate=False):
    """
    Generate Gray-Scott simulation data.
    
    Args:
        params: Dictionary containing simulation parameters:
            - width, height: Grid dimensions
            - domain_width, domain_height: Physical domain size
            - steps: Number of simulation steps
            - save_interval: Interval for saving snapshots
            - Du, Dv, f, k: Gray-Scott parameters
            - final_time: Total simulation time
            - use_gpu: Whether to use GPU (optional, default True)
        output_dir: Directory to save generated data
        force_regenerate: If True, regenerate even if data exists
        
    Returns:
        str: Path to the output directory containing generated data
    """
    if not force_regenerate and check_data_exists(output_dir, params):
        print("Using existing data...")
        return output_dir

    print("Generating new data...")

    device = 'cuda' if params.get('use_gpu', True) and torch.cuda.is_available() else 'cpu'
    print(f"Generating data on {device}...")

    os.makedirs(output_dir, exist_ok=True)

    final_time = params.get('final_time', 2000.0)
    dt = final_time / params['steps']

    model = GrayScottModel(
        width=params['width'],
        height=params['height'],
        domain_width=params['domain_width'],
        domain_height=params['domain_height'],
        Du=params['Du'],
        Dv=params['Dv'],
        f=params['f'],
        k=params['k'],
        dt=dt,
        device=device
    )

    stability_info = model.get_stability_info()
    print(f"Using dt = {dt:.6f}")
    print(f"Stability limit = {stability_info['stability_limit']:.6f}")
    if not stability_info['is_stable']:
        print(f"WARNING: dt is above stability limit!")

    num_snapshots = params['steps'] // params['save_interval'] + 1
    print(f"Will generate {num_snapshots} snapshots over {params['steps']:,} steps")

    U_timeseries = np.zeros((num_snapshots, params['height'], params['width']), dtype=np.float32)
    V_timeseries = np.zeros((num_snapshots, params['height'], params['width']), dtype=np.float32)
    time_points = np.zeros(num_snapshots, dtype=np.float32)

    U_np, V_np = model.get_numpy_arrays()
    U_timeseries[0] = U_np
    V_timeseries[0] = V_np
    time_points[0] = 0.0

    print("Running simulation...")
    snapshot_idx = 1

    with torch.no_grad():
        for step in range(1, params['steps'] + 1):
            model.forward(steps=1)

            if step % params['save_interval'] == 0:
                U_np, V_np = model.get_numpy_arrays()
                U_timeseries[snapshot_idx] = U_np
                V_timeseries[snapshot_idx] = V_np
                time_points[snapshot_idx] = step * dt
                snapshot_idx += 1

    print("Saving data...")
    np.save(os.path.join(output_dir, 'parameters.npy'), params)
    np.save(os.path.join(output_dir, 'U_timeseries.npy'), U_timeseries)
    np.save(os.path.join(output_dir, 'V_timeseries.npy'), V_timeseries)
    np.save(os.path.join(output_dir, 'time_points.npy'), time_points)

    print("Converting to (u,v,x,y,t) tuple format...")
    data_tuples = convert_to_tuples(U_timeseries, V_timeseries, time_points, params)
    np.save(os.path.join(output_dir, 'data_tuples.npy'), data_tuples)

    print(f"Data saved to: {output_dir}")
    return output_dir


__all__ = [
    'check_data_exists',
    'convert_to_tuples',
    'generate_data'
]
