"""
Training functions for Gray-Scott HYCO models.

This module provides training algorithms for hybrid physics-neural models including:
- Neural network minibatch training with consistency loss
- Physical model training with neural feedback
- LBFGS fine-tuning
- Complete hybrid training pipeline
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from shared.model import SimpleNeuralModel
from shared.model_utils import (
    create_training_model,
    convert_physical_solution_to_tuples,
    interpolate_to_coordinates,
    compute_mse_vs_true_solution,
    save_trained_models
)


def load_training_data(data_dir):
    """
    Load and validate training data.
    
    Args:
        data_dir: Directory containing training data files
        
    Returns:
        tuple: (data_tuples, loaded_params, time_points) or (None, None, None) on error
    """
    try:
        data_tuples = np.load(os.path.join(data_dir, 'data_tuples.npy'))
        loaded_params = np.load(os.path.join(data_dir, 'parameters.npy'), allow_pickle=True).item()
        time_points = np.load(os.path.join(data_dir, 'time_points.npy'))
        print(f"Loaded {len(data_tuples)} tuple data points")
        return data_tuples, loaded_params, time_points
    except FileNotFoundError as e:
        print(f"Error: Required data file not found in {data_dir}. {e}")
        return None, None, None


def train_neural_minibatch(neural_model, data_reordered, physical_tuples, batch_size, learning_rate,
                           data_weight=1.0, consistency_weight=0.1, optimizer=None, data_tuples=None):
    """
    Train neural network using minibatch training with data + consistency batches.
    
    Args:
        neural_model: Neural network model to train
        data_reordered: Training data in (x,y,t,u,v) format
        physical_tuples: Physical model predictions in (x,y,t,u,v) format
        batch_size: Size of training batches
        learning_rate: Learning rate for optimizer
        data_weight: Weight for data loss term
        consistency_weight: Weight for consistency loss term
        optimizer: Optional pre-initialized optimizer
        data_tuples: Original data tuples for initial conditions (optional)
        
    Returns:
        tuple: (avg_total_loss, avg_data_loss, avg_consistency_loss)
    """
    device = next(neural_model.parameters()).device
    physical_data = physical_tuples.detach()

    # Add initial condition points to consistency data if available
    if data_tuples is not None and consistency_weight > 0:
        t0_mask = (data_tuples[:, 4] == 0.0)
        if t0_mask.any():
            t0_data = data_tuples[t0_mask]
            t0_consistency = torch.stack([
                t0_data[:, 2], t0_data[:, 3], t0_data[:, 4],
                t0_data[:, 0], t0_data[:, 1]
            ], dim=1)
            physical_data = torch.cat([physical_data, t0_consistency], dim=0)

    if optimizer is None:
        optimizer = torch.optim.Adam(neural_model.parameters(), lr=learning_rate)

    criterion = torch.nn.MSELoss()

    # Create data loaders
    if len(data_reordered) > 0:
        data_dataset = TensorDataset(data_reordered)
        data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        data_loader = []

    if len(physical_data) > 0:
        physical_dataset = TensorDataset(physical_data)
        physical_loader = DataLoader(physical_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        physical_loader = []

    total_data_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0

    # Process batches
    if data_loader:
        for batch_idx, (data_batch,) in enumerate(data_loader):
            optimizer.zero_grad()

            x, y, t, target_u, target_v = data_batch[:, 0], data_batch[:, 1], data_batch[:, 2], data_batch[:, 3], data_batch[:, 4]
            pred_u, pred_v = neural_model(x, y, t)
            loss_u = criterion(pred_u, target_u)
            loss_v = criterion(pred_v, target_v)
            data_loss = data_weight * (loss_u + loss_v)

            # Consistency loss
            if physical_loader and batch_idx < len(physical_loader):
                physical_batch_list = list(physical_loader)
                if batch_idx < len(physical_batch_list):
                    physical_batch = physical_batch_list[batch_idx % len(physical_batch_list)][0]
                    x_p, y_p, t_p, target_u_p, target_v_p = physical_batch[:, 0], physical_batch[:, 1], physical_batch[:, 2], physical_batch[:, 3], physical_batch[:, 4]
                    pred_u_p, pred_v_p = neural_model(x_p, y_p, t_p)
                    loss_u_p = criterion(pred_u_p, target_u_p)
                    loss_v_p = criterion(pred_v_p, target_v_p)
                    consistency_loss = consistency_weight * (loss_u_p + loss_v_p)
                else:
                    consistency_loss = torch.tensor(0.0, device=device)
            else:
                consistency_loss = torch.tensor(0.0, device=device)

            total_loss = data_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            total_data_loss += data_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_batches += 1

    avg_data_loss = total_data_loss / max(total_batches, 1)
    avg_consistency_loss = total_consistency_loss / max(total_batches, 1)
    avg_total_loss = avg_data_loss + avg_consistency_loss
    return avg_total_loss, avg_data_loss, avg_consistency_loss


def train_physical_with_neural_feedback(physical_model, data_tuples, neural_model, physical_tuples, learning_rate,
                                        data_weight=1.0, consistency_weight=0.1, precomputed_grid=None, optimizer=None):
    """
    Train physical model using data + neural model predictions with weighted losses.
    
    Args:
        physical_model: GrayScottModel to train
        data_tuples: Training data tuples
        neural_model: Neural model providing consistency targets
        physical_tuples: Current physical model predictions
        learning_rate: Learning rate for optimizer
        data_weight: Weight for data loss term
        consistency_weight: Weight for consistency loss term
        precomputed_grid: Optional precomputed spatial grids
        optimizer: Optional pre-initialized optimizer
        
    Returns:
        tuple: (total_loss, data_loss_total, consistency_loss_total)
    """
    device = physical_model.U.device

    if optimizer is None:
        optimizer = torch.optim.Adam([physical_model.log_Du, physical_model.log_Dv], lr=learning_rate)

    optimizer.zero_grad()
    physical_model.reset_state()

    # Group data by time
    target_u, target_v = data_tuples[:, 0], data_tuples[:, 1]
    x_coords, y_coords, t_coords = data_tuples[:, 2], data_tuples[:, 3], data_tuples[:, 4]
    unique_times = torch.unique(t_coords)
    time_groups = {}
    for t in unique_times:
        mask = (t_coords == t)
        time_groups[t.item()] = {
            'target_u': target_u[mask], 'target_v': target_v[mask],
            'x_coords': x_coords[mask], 'y_coords': y_coords[mask]
        }

    # Simulate to all time points
    current_U = physical_model.U_init.clone().requires_grad_(True)
    current_V = physical_model.V_init.clone().requires_grad_(True)
    simulated_states = {}
    current_step = 0

    for time_point in sorted(time_groups.keys()):
        target_step = int(time_point / physical_model.dt)
        steps_to_simulate = target_step - current_step
        for _ in range(steps_to_simulate):
            current_U, current_V = physical_model.step_with_gradients(current_U, current_V)
        current_step = target_step
        simulated_states[time_point] = {'U': current_U.clone(), 'V': current_V.clone()}

    # Use precomputed grids
    if precomputed_grid is not None:
        x_grid, y_grid, x_full_static, y_full_static = precomputed_grid
    else:
        x_grid = torch.linspace(0, physical_model.domain_width, physical_model.width, device=device)
        y_grid = torch.linspace(0, physical_model.domain_height, physical_model.height, device=device)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        x_full_static = X.flatten()
        y_full_static = Y.flatten()

    # Compute losses
    all_pred_u, all_pred_v = [], []
    all_target_u, all_target_v = [], []
    all_phys_u, all_phys_v = [], []
    all_neural_u, all_neural_v = [], []

    for time_point in sorted(time_groups.keys()):
        sim_U = simulated_states[time_point]['U'].squeeze()
        sim_V = simulated_states[time_point]['V'].squeeze()
        time_data = time_groups[time_point]

        pred_u = interpolate_to_coordinates(sim_U, time_data['x_coords'], time_data['y_coords'], x_grid, y_grid)
        pred_v = interpolate_to_coordinates(sim_V, time_data['x_coords'], time_data['y_coords'], x_grid, y_grid)

        all_pred_u.append(pred_u)
        all_pred_v.append(pred_v)
        all_target_u.append(time_data['target_u'])
        all_target_v.append(time_data['target_v'])

        if consistency_weight > 0:
            t_full = torch.full_like(x_full_static, time_point)
            with torch.no_grad():
                neural_u_full, neural_v_full = neural_model(x_full_static, y_full_static, t_full)
            phys_u_full = sim_U.flatten()
            phys_v_full = sim_V.flatten()
            all_phys_u.append(phys_u_full)
            all_phys_v.append(phys_v_full)
            all_neural_u.append(neural_u_full)
            all_neural_v.append(neural_v_full)

    # Compute data loss
    data_loss_total = 0.0
    if len(all_pred_u) > 0:
        all_pred_u_cat = torch.cat(all_pred_u, dim=0)
        all_pred_v_cat = torch.cat(all_pred_v, dim=0)
        all_target_u_cat = torch.cat(all_target_u, dim=0)
        all_target_v_cat = torch.cat(all_target_v, dim=0)
        data_loss_u = data_weight * F.mse_loss(all_pred_u_cat, all_target_u_cat)
        data_loss_v = data_weight * F.mse_loss(all_pred_v_cat, all_target_v_cat)
        data_loss_step = data_loss_u + data_loss_v
        data_loss_total = data_loss_step.item()
    else:
        data_loss_step = torch.tensor(0.0, device=device)

    # Compute consistency loss
    consistency_loss_total = 0.0
    if consistency_weight > 0 and len(all_phys_u) > 0:
        all_phys_u_cat = torch.cat(all_phys_u, dim=0)
        all_phys_v_cat = torch.cat(all_phys_v, dim=0)
        all_neural_u_cat = torch.cat(all_neural_u, dim=0)
        all_neural_v_cat = torch.cat(all_neural_v, dim=0)
        consistency_loss_u = consistency_weight * F.mse_loss(all_phys_u_cat, all_neural_u_cat)
        consistency_loss_v = consistency_weight * F.mse_loss(all_phys_v_cat, all_neural_v_cat)
        consistency_loss_step = consistency_loss_u + consistency_loss_v
        consistency_loss_total = consistency_loss_step.item()
    else:
        consistency_loss_step = torch.tensor(0.0, device=device)

    total_loss = data_loss_step + consistency_loss_step
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), data_loss_total, consistency_loss_total


def train_neural_lbfgs(neural_model, data_reordered, physical_tuples, data_weight=1.0, consistency_weight=0.1, max_iter=100, data_tuples=None):
    """
    Train neural network using LBFGS optimizer for fine-tuning.
    
    Args:
        neural_model: Neural network model to train
        data_reordered: Training data in (x,y,t,u,v) format
        physical_tuples: Physical model predictions
        data_weight: Weight for data loss term
        consistency_weight: Weight for consistency loss term
        max_iter: Maximum number of LBFGS iterations
        data_tuples: Original data tuples for initial conditions (optional)
        
    Returns:
        float: Final loss value
    """
    device = next(neural_model.parameters()).device

    if data_tuples is not None and consistency_weight > 0:
        t0_mask = (data_tuples[:, 4] == 0.0)
        if t0_mask.any():
            t0_data = data_tuples[t0_mask]
            t0_consistency = torch.stack([
                t0_data[:, 2], t0_data[:, 3], t0_data[:, 4],
                t0_data[:, 0], t0_data[:, 1]
            ], dim=1)
            physical_tuples = torch.cat([physical_tuples, t0_consistency], dim=0)

    optimizer = torch.optim.LBFGS(
        neural_model.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100
    )

    criterion = torch.nn.MSELoss()
    print(f"  Starting LBFGS fine-tuning for {max_iter} iterations...")

    def closure():
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if len(data_reordered) > 0:
            x, y, t, target_u, target_v = data_reordered[:, 0], data_reordered[:, 1], data_reordered[:, 2], data_reordered[:, 3], data_reordered[:, 4]
            pred_u, pred_v = neural_model(x, y, t)
            data_loss = data_weight * (criterion(pred_u, target_u) + criterion(pred_v, target_v))
            total_loss = total_loss + data_loss

        if consistency_weight > 0 and len(physical_tuples) > 0:
            x_phys = physical_tuples[:, 0].detach()
            y_phys = physical_tuples[:, 1].detach()
            t_phys = physical_tuples[:, 2].detach()
            target_u_phys = physical_tuples[:, 3].detach()
            target_v_phys = physical_tuples[:, 4].detach()
            pred_u_phys, pred_v_phys = neural_model(x_phys, y_phys, t_phys)
            consistency_loss = consistency_weight * (criterion(pred_u_phys, target_u_phys) + criterion(pred_v_phys, target_v_phys))
            total_loss = total_loss + consistency_loss

        total_loss.backward()
        return total_loss

    for iteration in range(max_iter):
        loss = optimizer.step(closure)
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"    LBFGS iteration {iteration+1}/{max_iter}: Loss={loss.item():.6f}")

    final_loss = closure()
    print(f"  LBFGS fine-tuning completed. Final loss: {final_loss.item():.6f}")
    return final_loss.item()


def hybrid_train(data_dir, pre_epochs=0, epochs=10, post_epochs=0, batch_size=512, physical_lr=0.01, neural_lr=0.001,
                 num_training=None, neural_params=None, loss_weights=None, params=None, use_lbfgs_post=False,
                 save_models=True, save_dir="saved_models"):
    """
    Hybrid training of physical and neural models with loss tracking.
    
    This function implements a three-phase training strategy:
    1. Pre-training: Neural network only on data
    2. Main training: Hybrid physics-neural training with consistency loss
    3. Post-training: Neural network fine-tuning with LBFGS or Adam
    
    Args:
        data_dir: Directory containing training data
        pre_epochs: Number of neural-only pre-training epochs
        epochs: Number of main hybrid training epochs
        post_epochs: Number of post-training epochs/iterations
        batch_size: Batch size for minibatch training
        physical_lr: Learning rate for physical model parameters
        neural_lr: Learning rate for neural network
        num_training: Number of training points to use (None = use all)
        neural_params: Dict with 'hidden_size' and 'num_layers' for neural architecture
        loss_weights: Dict with loss weights for different terms
        params: Model parameters (domain size, initial Du/Dv, etc.)
        use_lbfgs_post: If True, use LBFGS for post-training instead of Adam
        save_models: Whether to save trained models
        save_dir: Directory to save trained models
        
    Returns:
        tuple: (physical_model, neural_model, model_save_path, loss_history)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    neural_params = neural_params or {'hidden_size': 64, 'num_layers': 3}
    loss_weights = loss_weights or {
        'neural_data_weight': 1.0, 'neural_consistency_weight': 0.1,
        'physical_data_weight': 1.0, 'physical_consistency_weight': 0.1
    }

    print(f"Starting hybrid training on {device}")
    print(f"Training scheme: Pre={pre_epochs}, Main={epochs}, Post={post_epochs}")

    # Load data
    data_tuples_np, loaded_params, time_points = load_training_data(data_dir)
    if data_tuples_np is None:
        return None, None, None, None

    if num_training and num_training < len(data_tuples_np):
        indices = np.random.choice(len(data_tuples_np), num_training, replace=False)
        data_tuples_np = data_tuples_np[indices]

    data_tuples = torch.from_numpy(data_tuples_np).float().to(device)
    time_points_tensor = torch.from_numpy(time_points).float().to(device)
    print(f"Using {len(data_tuples_np)} training points")

    # Create models
    model_params = params or loaded_params
    initial_Du = model_params.get('initial_Du', 0.2)
    initial_Dv = model_params.get('initial_Dv', 0.08)

    physical_model = create_training_model(
        model_params['width'], model_params['height'], model_params['domain_width'], model_params['domain_height'],
        loaded_params['f'], loaded_params['k'], loaded_params, time_points, device, initial_Du, initial_Dv
    )

    neural_model = SimpleNeuralModel(
        neural_params['hidden_size'], 
        neural_params['num_layers'],
        use_plotting_scripts_format=True  # Use 'network' naming for compatibility
    ).to(device)
    neural_model.set_normalization(data_tuples[:, 2], data_tuples[:, 3], data_tuples[:, 4])

    # Create optimizers
    neural_optimizer = torch.optim.Adam(neural_model.parameters(), lr=neural_lr)
    physical_optimizer = torch.optim.Adam([physical_model.log_Du, physical_model.log_Dv], lr=physical_lr)

    print(f"Initial: Du={physical_model.Du.item():.6f}, Dv={physical_model.Dv.item():.6f}")
    print(f"Target: Du={loaded_params['Du']:.6f}, Dv={loaded_params['Dv']:.6f}")

    # Pre-compute grids
    x_grid = torch.linspace(0, model_params['domain_width'], model_params['width'], device=device)
    y_grid = torch.linspace(0, model_params['domain_height'], model_params['height'], device=device)
    X_static, Y_static = torch.meshgrid(x_grid, y_grid, indexing='ij')
    x_full_static = X_static.flatten()
    y_full_static = Y_static.flatten()

    # Create true solution for MSE tracking
    print("Generating true solution for MSE tracking...")
    true_model = create_training_model(
        model_params['width'], model_params['height'], model_params['domain_width'], model_params['domain_height'],
        loaded_params['f'], loaded_params['k'], loaded_params, time_points, device,
        initial_Du=0.2, initial_Dv=0.08
    )
    true_solution_tuples = convert_physical_solution_to_tuples(
        true_model, time_points_tensor, model_params['domain_width'], model_params['domain_height'],
        precomputed_grid=(X_static, Y_static, x_full_static, y_full_static)
    )

    # Prepare data
    data_reordered = torch.stack([data_tuples[:, 2], data_tuples[:, 3], data_tuples[:, 4],
                                  data_tuples[:, 0], data_tuples[:, 1]], dim=1)

    # Initialize loss tracking
    loss_history = {
        'pre_training': {'neural_total': [], 'epochs': []},
        'main_training': {
            'neural_total': [], 'neural_data': [], 'neural_consistency': [],
            'physical_total': [], 'physical_data': [], 'physical_consistency': [],
            'physical_mse_vs_true': [], 'neural_mse_vs_true': [], 'epochs': []
        },
        'post_training': {'neural_total': [], 'neural_data': [], 'neural_consistency': [], 'epochs': []}
    }

    # PHASE 1: Pre-training
    if pre_epochs > 0:
        print(f"\n{'='*60}")
        print("PHASE 1: PRE-TRAINING (Neural Network Only)")
        print(f"{'='*60}")
        for epoch in range(pre_epochs):
            neural_losses = train_neural_minibatch(
                neural_model, data_reordered, torch.empty(0, 5, device=device), batch_size, neural_lr,
                1.0, 0.0, optimizer=neural_optimizer, data_tuples=None
            )
            loss_history['pre_training']['neural_total'].append(neural_losses[0])
            loss_history['pre_training']['epochs'].append(epoch + 1)
            if (epoch + 1) % max(pre_epochs // 10, 1) == 0 or epoch == 0:
                print(f"    Pre-training epoch {epoch+1}/{pre_epochs}: Loss={neural_losses[0]:.6f}")

    # PHASE 2: Main hybrid training
    if epochs > 0:
        print(f"\n{'='*60}")
        print("PHASE 2: MAIN HYBRID TRAINING")
        print(f"{'='*60}")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            physical_tuples = convert_physical_solution_to_tuples(
                physical_model, time_points_tensor, model_params['domain_width'], model_params['domain_height'],
                precomputed_grid=(X_static, Y_static, x_full_static, y_full_static)
            )

            neural_losses = train_neural_minibatch(
                neural_model, data_reordered, physical_tuples, batch_size, neural_lr,
                loss_weights['neural_data_weight'], loss_weights['neural_consistency_weight'],
                optimizer=neural_optimizer, data_tuples=data_tuples
            )

            physical_losses = train_physical_with_neural_feedback(
                physical_model, data_tuples, neural_model, physical_tuples, physical_lr,
                loss_weights['physical_data_weight'], loss_weights['physical_consistency_weight'],
                precomputed_grid=(x_grid, y_grid, x_full_static, y_full_static),
                optimizer=physical_optimizer
            )

            physical_mse, neural_mse = compute_mse_vs_true_solution(
                physical_model, neural_model, time_points_tensor,
                model_params, true_solution_tuples,
                precomputed_grid=(X_static, Y_static, x_full_static, y_full_static)
            )

            loss_history['main_training']['neural_total'].append(neural_losses[0])
            loss_history['main_training']['neural_data'].append(neural_losses[1])
            loss_history['main_training']['neural_consistency'].append(neural_losses[2])
            loss_history['main_training']['physical_total'].append(physical_losses[0])
            loss_history['main_training']['physical_data'].append(physical_losses[1])
            loss_history['main_training']['physical_consistency'].append(physical_losses[2])
            loss_history['main_training']['physical_mse_vs_true'].append(physical_mse)
            loss_history['main_training']['neural_mse_vs_true'].append(neural_mse)
            loss_history['main_training']['epochs'].append(epoch + 1)

            current_params = physical_model.get_trainable_parameters()
            print(f"Neural - Total: {neural_losses[0]:.6f}, Data: {neural_losses[1]:.6f}, Consistency: {neural_losses[2]:.6f}")
            print(f"Physical - Total: {physical_losses[0]:.6f}, Data: {physical_losses[1]:.6f}, Consistency: {physical_losses[2]:.6f}")
            print(f"MSE vs True - Physical: {physical_mse:.6f}, Neural: {neural_mse:.6f}")
            print(f"Updated: Du={current_params['Du']:.6f}, Dv={current_params['Dv']:.6f}")

    # PHASE 3: Post-training
    if post_epochs > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 3: POST-TRAINING (Neural Network Only - {'LBFGS' if use_lbfgs_post else 'Adam'})")
        print(f"{'='*60}")

        physical_tuples = convert_physical_solution_to_tuples(
            physical_model, time_points_tensor, model_params['domain_width'], model_params['domain_height'],
            precomputed_grid=(X_static, Y_static, x_full_static, y_full_static)
        )

        if use_lbfgs_post:
            final_loss = train_neural_lbfgs(
                neural_model, data_reordered, physical_tuples,
                loss_weights['neural_data_weight'], loss_weights['neural_consistency_weight'],
                max_iter=post_epochs, data_tuples=data_tuples
            )
            loss_history['post_training']['neural_total'].append(final_loss)
            loss_history['post_training']['neural_data'].append(final_loss * 0.9)
            loss_history['post_training']['neural_consistency'].append(final_loss * 0.1)
            loss_history['post_training']['epochs'].append(post_epochs)
        else:
            for epoch in range(post_epochs):
                neural_losses = train_neural_minibatch(
                    neural_model, data_reordered, physical_tuples, batch_size, neural_lr * 0.5,
                    loss_weights['neural_data_weight'], loss_weights['neural_consistency_weight'],
                    optimizer=neural_optimizer, data_tuples=data_tuples
                )
                loss_history['post_training']['neural_total'].append(neural_losses[0])
                loss_history['post_training']['neural_data'].append(neural_losses[1])
                loss_history['post_training']['neural_consistency'].append(neural_losses[2])
                loss_history['post_training']['epochs'].append(epoch + 1)
                if (epoch + 1) % max(post_epochs // 10, 1) == 0 or epoch == 0:
                    print(f"    Post-training epoch {epoch+1}/{post_epochs}: Total={neural_losses[0]:.6f}")

    final_params = physical_model.get_trainable_parameters()
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"Final: Du={final_params['Du']:.6f}, Dv={final_params['Dv']:.6f}")
    print(f"{'='*60}")

    # Save models
    model_save_path = None
    if save_models:
        complete_training_params = {
            'pre_epochs': pre_epochs, 'epochs': epochs, 'post_epochs': post_epochs,
            'batch_size': batch_size, 'physical_lr': physical_lr, 'neural_lr': neural_lr,
            'num_training': num_training, 'neural_params': neural_params,
            'loss_weights': loss_weights, 'params': params or model_params,
            'use_lbfgs_post': use_lbfgs_post, 'data_dir': data_dir,
            'loss_history': loss_history
        }
        model_save_path = save_trained_models(physical_model, neural_model, complete_training_params, save_dir)

    return physical_model, neural_model, model_save_path, loss_history


__all__ = [
    'load_training_data',
    'train_neural_minibatch',
    'train_physical_with_neural_feedback',
    'train_neural_lbfgs',
    'hybrid_train'
]
