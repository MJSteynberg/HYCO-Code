"""
Model loading utilities for Gray-Scott models.
Handles loading saved PINN, HYCO, and NN models with their configurations.
"""

import os
import pickle
import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional

from shared.pinn import GrayScottPINN, GrayScottNeuralNetwork
from shared.model import SimpleNeuralModel, GrayScottModel


class SimpleScaler:
    """Scaler class compatible with training pipeline."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
        
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        s[s == 0] = 1.0
        self.scale_ = s
        self.fitted = True
        return self
        
    def transform(self, X):
        if not self.fitted: 
            raise RuntimeError("Scaler not fitted")
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X):
        if not self.fitted: 
            raise RuntimeError("Scaler not fitted")
        X = np.asarray(X)
        return X * self.scale_ + self.mean_


def load_pinn_model_and_scalers(model_dir: str) -> Tuple[GrayScottPINN, Dict]:
    """
    Load a saved PINN model and its associated scalers.

    Args:
        model_dir: Directory where the model and scalers are saved

    Returns:
        Tuple of (model, scalers_dict)
    """
    model_path = os.path.join(model_dir, 'model_state.pth')
    scalers_path = os.path.join(model_dir, 'scalers.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model state file not found: {model_path}")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Scalers file not found: {scalers_path}")

    # Load metadata to get the correct architecture
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        num_hidden = metadata['architecture']['num_hidden']
        num_layers = metadata['architecture']['num_layers']
        learn_diffusion = metadata.get('learn_diffusion', True)
    else:
        # Default values if metadata not available
        num_hidden = 256
        num_layers = 4
        learn_diffusion = True

    model = GrayScottPINN(
        num_hidden=num_hidden, 
        num_layers=num_layers, 
        learn_diffusion=learn_diffusion
    )

    # Determine device and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # Load scalers with backward compatibility
    try:
        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'gray_scott_pinn':
                    if name in ['SimpleScaler', 'StandardScaler']:
                        return SimpleScaler
                    if name == 'GrayScottPINN':
                        return GrayScottPINN
                if module == 'sklearn.preprocessing' and name == 'StandardScaler':
                    try:
                        from sklearn.preprocessing import StandardScaler as SkStd
                        return SkStd
                    except Exception:
                        return SimpleScaler
                return super().find_class(module, name)

        with open(scalers_path, 'rb') as f:
            unpickler = CompatUnpickler(f)
            scalers = unpickler.load()

            # Validate scalers
            for key, dim in (('input', 3), ('output', 2)):
                if key not in scalers:
                    raise AttributeError(f"Missing scaler key: {key}")
                scaler = scalers[key]
                if not hasattr(scaler, 'transform'):
                    raise AttributeError(f"Scaler for '{key}' has no transform method")
                test_data = np.zeros((1, dim))
                scaler.transform(test_data)

    except Exception as e:
        print(f"Warning: Could not load scalers ({e}). Using identity scalers.")
        scalers = {
            'input': SimpleScaler(),
            'output': SimpleScaler()
        }
        scalers['input'].mean_ = np.array([0.0, 0.0, 0.0])
        scalers['input'].scale_ = np.array([1.0, 1.0, 1.0])
        scalers['input'].fitted = True
        scalers['output'].mean_ = np.array([0.0, 0.0])
        scalers['output'].scale_ = np.array([1.0, 1.0])
        scalers['output'].fitted = True

    print(f"Model and scalers loaded from {model_dir} on device: {device}")
    return model, scalers


def evaluate_pinn_model(
    model_dir: str, 
    data_dir: str, 
    time_snapshots: list
) -> dict:
    """
    Load a trained PINN model and return predictions at specified times.
    
    Args:
        model_dir: Directory containing saved PINN model
        data_dir: Directory containing data for spatial grid
        time_snapshots: List of time points to evaluate
        
    Returns:
        Dictionary mapping time -> {'u': prediction_array}
    """
    from shared.data_loader import load_tuple_data
    
    print("Evaluating PINN model...")
    model, scalers = load_pinn_model_and_scalers(model_dir)
    data = load_tuple_data(data_dir)
    data_tuples = data['data_tuples']

    x_min, x_max = data_tuples[:, 2].min(), data_tuples[:, 2].max()
    y_min, y_max = data_tuples[:, 3].min(), data_tuples[:, 3].max()
    resolution = 64
    x_coords = np.linspace(x_min, x_max, resolution)
    y_coords = np.linspace(y_min, y_max, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    pinn_predictions = {}
    device = next(model.parameters()).device
    
    for t_val in time_snapshots:
        x_flat = X_grid.flatten()
        y_flat = Y_grid.flatten()
        t_flat = np.full_like(x_flat, t_val)
        coords = np.column_stack([x_flat, y_flat, t_flat])

        coords_norm = scalers['input'].transform(coords)
        coords_tensor = torch.FloatTensor(coords_norm).to(device)

        with torch.no_grad():
            pred_norm = model(coords_tensor)
            pred_phys = scalers['output'].inverse_transform(pred_norm.cpu().numpy())

        u_pred = pred_phys[:, 0].reshape(X_grid.shape)
        pinn_predictions[t_val] = {'u': u_pred}

    return pinn_predictions


def _safe_tensor_to_device(value, device):
    """Safely convert saved value back to tensor on device."""
    if value is None:
        return None
    elif torch.is_tensor(value):
        return value.to(device)
    else:
        return torch.tensor(value, device=device)


def load_hybrid_models(
    model_path: str, 
    device: str
) -> Tuple[Optional[GrayScottModel], Optional[SimpleNeuralModel], Optional[Dict]]:
    """
    Load previously trained physical and neural models (HYCO or NN).
    
    Args:
        model_path: Directory containing model files
        device: Device to load models on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (physical_model, neural_model, training_params)
    """
    print(f"Loading models from: {model_path}")
    required_files = ['physical_model.pt', 'neural_model.pt', 'training_params.json']
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"Warning: Required file not found: {file}. Skipping model loading.")
            return None, None, None

    # Load training parameters
    with open(os.path.join(model_path, "training_params.json"), 'r') as f:
        training_params = json.load(f)

    # Load physical model
    physical_state = torch.load(
        os.path.join(model_path, "physical_model.pt"), 
        map_location=device, 
        weights_only=False
    )
    loaded_Du = torch.exp(physical_state['log_Du']).item()
    loaded_Dv = torch.exp(physical_state['log_Dv']).item()
    
    physical_model = GrayScottModel(
        width=physical_state['width'],
        height=physical_state['height'],
        domain_width=physical_state['domain_width'],
        domain_height=physical_state['domain_height'],
        Du=loaded_Du, 
        Dv=loaded_Dv,
        f=physical_state['f'],
        k=physical_state['k'],
        dt=physical_state['dt'],
        device=device
    )

    # Load neural model
    neural_state = torch.load(
        os.path.join(model_path, "neural_model.pt"), 
        map_location=device, 
        weights_only=False
    )
    neural_model = SimpleNeuralModel(
        neural_state['hidden_size'],
        neural_state['num_layers'],
        use_plotting_scripts_format=True
    ).to(device)
    
    neural_model.load_state_dict(neural_state['state_dict'])
    
    # Restore normalization parameters
    normalization_params = neural_state.get('normalization', {})
    if 'x_mean' in normalization_params and normalization_params['x_mean'] is not None:
        neural_model.x_mean = _safe_tensor_to_device(
            normalization_params['x_mean'], device
        ).item()
    if 'x_std' in normalization_params and normalization_params['x_std'] is not None:
        neural_model.x_std = _safe_tensor_to_device(
            normalization_params['x_std'], device
        ).item()
    if 'y_mean' in normalization_params and normalization_params['y_mean'] is not None:
        neural_model.y_mean = _safe_tensor_to_device(
            normalization_params['y_mean'], device
        ).item()
    if 'y_std' in normalization_params and normalization_params['y_std'] is not None:
        neural_model.y_std = _safe_tensor_to_device(
            normalization_params['y_std'], device
        ).item()
    if 't_mean' in normalization_params and normalization_params['t_mean'] is not None:
        neural_model.t_mean = _safe_tensor_to_device(
            normalization_params['t_mean'], device
        ).item()
    if 't_std' in normalization_params and normalization_params['t_std'] is not None:
        neural_model.t_std = _safe_tensor_to_device(
            normalization_params['t_std'], device
        ).item()
    
    neural_model.hidden_size = neural_state['hidden_size']
    neural_model.num_layers = neural_state['num_layers']
    neural_model.eval()

    print("Models loaded successfully!")
    return physical_model, neural_model, training_params


def load_experiment_models(experiment_dir: str) -> Dict:
    """
    Load both sets of trained models from the experiment directory.
    
    Args:
        experiment_dir: Directory containing consistency_0.0 and consistency_1.0 subdirectories
        
    Returns:
        Dictionary with 'consistency_0' and 'consistency_1' model sets
    """
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load consistency = 1.0 models (Hybrid)
    consistency_1_path = os.path.join(experiment_dir, "consistency_1.0")
    physical_1, neural_1, params_1 = load_hybrid_models(consistency_1_path, device)
    if physical_1 and neural_1:
        models['consistency_1'] = {
            'physical': physical_1, 
            'neural': neural_1, 
            'params': params_1
        }
        print(f"✓ Successfully loaded HYCO models from: {consistency_1_path}")

    # Load consistency = 0.0 models (Neural Network only)
    consistency_0_path = os.path.join(experiment_dir, "consistency_0.0")
    physical_0, neural_0, params_0 = load_hybrid_models(consistency_0_path, device)
    if physical_0 and neural_0:
        models['consistency_0'] = {
            'physical': physical_0, 
            'neural': neural_0, 
            'params': params_0
        }
        print(f"✓ Successfully loaded NN models from: {consistency_0_path}")

    return models
