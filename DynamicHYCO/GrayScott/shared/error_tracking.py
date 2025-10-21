"""
Comprehensive error tracking for Gray-Scott PINN training.
Based on the ERROR folder implementation patterns.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os


class ErrorTracker:
    """
    Tracks multiple types of errors during PINN training:
    1. MSE (Mean Squared Error) - Data fitting accuracy
    2. L2 Norm Error - Solution field accuracy relative to true solution
    3. Parameter Error - Learned parameter accuracy (Du, Dv, f, k)
    4. Physics Loss - PDE residual compliance
    5. Data Loss - Training data fitting
    6. Boundary/Initial Condition losses
    """
    
    def __init__(self, true_params: Dict = None, save_dir: str = "results"):
        """
        Initialize error tracker.
        
        Args:
            true_params: Dictionary with true parameter values {'Du': 0.2, 'Dv': 0.08, 'f': 0.018, 'k': 0.051}
            save_dir: Directory to save error tracking files
        """
        self.true_params = true_params or {}
        self.save_dir = save_dir
        
        # Error history storage
        self.error_history = {
            'epoch': [],
            'mse_normalized': [],      # MSE in normalized space
            'mse_physical': [],        # MSE in physical space
            'l2_norm_error': [],       # L2 norm relative error
            'l2_norm_u': [],           # L2 norm error for U component
            'l2_norm_v': [],           # L2 norm error for V component
            'parameter_error': [],     # Combined parameter error
            'du_error': [],            # Du parameter error
            'dv_error': [],            # Dv parameter error
            'f_error': [],             # f parameter error (if learned)
            'k_error': [],             # k parameter error (if learned)
            'physics_loss': [],        # PDE residual loss
            'data_loss': [],           # Data fitting loss
            'ic_loss': [],             # Initial condition loss
            'bc_loss': [],             # Boundary condition loss
            'total_loss': [],          # Total combined loss
            'u_mse': [],               # U component MSE
            'v_mse': [],               # V component MSE
            'u_relative_error': [],    # U component relative error
            'v_relative_error': [],    # V component relative error
        }
        
    def compute_mse_errors(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                          scalers: Dict = None) -> Dict[str, float]:
        """
        Compute MSE errors in both normalized and physical space.
        
        Args:
            y_pred: Predicted solutions (normalized)
            y_true: True solutions
            scalers: Dictionary containing input/output scalers
            
        Returns:
            Dictionary with MSE error metrics
        """
        errors = {}
        
        # MSE in normalized space
        mse_norm = torch.mean((y_pred - y_true)**2).item()
        errors['mse_normalized'] = mse_norm
        
        # Convert to physical space if scalers available
        if scalers and 'output' in scalers:
            out_scaler = scalers['output']
            y_pred_phys = y_pred.cpu().numpy() * out_scaler.scale_ + out_scaler.mean_
            y_true_phys = y_true.cpu().numpy() * out_scaler.scale_ + out_scaler.mean_
            
            # MSE in physical space
            u_mse = np.mean((y_pred_phys[:, 0] - y_true_phys[:, 0])**2)
            v_mse = np.mean((y_pred_phys[:, 1] - y_true_phys[:, 1])**2)
            mse_phys = u_mse + v_mse
            
            errors['mse_physical'] = mse_phys
            errors['u_mse'] = u_mse
            errors['v_mse'] = v_mse
            
            # Relative errors
            u_rel = np.sqrt(u_mse) / (np.std(y_true_phys[:, 0]) + 1e-8)
            v_rel = np.sqrt(v_mse) / (np.std(y_true_phys[:, 1]) + 1e-8)
            errors['u_relative_error'] = u_rel
            errors['v_relative_error'] = v_rel
        else:
            errors['mse_physical'] = mse_norm
            errors['u_mse'] = mse_norm / 2
            errors['v_mse'] = mse_norm / 2
            errors['u_relative_error'] = 0.0
            errors['v_relative_error'] = 0.0
            
        return errors
    
    def compute_l2_errors(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Compute L2 norm errors for solution fields.
        
        Args:
            y_pred: Predicted solutions
            y_true: True solutions
            
        Returns:
            Dictionary with L2 error metrics
        """
        errors = {}
        
        # Overall L2 relative error
        l2_pred = torch.linalg.norm(y_pred, ord=2, dim=0)
        l2_true = torch.linalg.norm(y_true, ord=2, dim=0)
        l2_diff = torch.linalg.norm(y_pred - y_true, ord=2, dim=0)
        
        l2_rel_u = (l2_diff[0] / (l2_true[0] + 1e-8)).item()
        l2_rel_v = (l2_diff[1] / (l2_true[1] + 1e-8)).item()
        l2_rel_total = (torch.linalg.norm(l2_diff) / (torch.linalg.norm(l2_true) + 1e-8)).item()
        
        errors['l2_norm_error'] = l2_rel_total
        errors['l2_norm_u'] = l2_rel_u
        errors['l2_norm_v'] = l2_rel_v
        
        return errors
    
    def compute_parameter_errors(self, learned_params: Dict) -> Dict[str, float]:
        """
        Compute parameter estimation errors.
        
        Args:
            learned_params: Dictionary with learned parameter values
            
        Returns:
            Dictionary with parameter error metrics
        """
        errors = {}
        param_errors = []
        
        for param_name in ['Du', 'Dv', 'f', 'k']:
            if param_name in learned_params and param_name in self.true_params:
                true_val = self.true_params[param_name]
                learned_val = learned_params[param_name]
                
                # Relative error
                rel_error = abs(learned_val - true_val) / (abs(true_val) + 1e-8)
                errors[f'{param_name.lower()}_error'] = rel_error
                param_errors.append(rel_error)
            else:
                errors[f'{param_name.lower()}_error'] = 0.0
        
        # Combined parameter error (L2 norm of relative errors)
        if param_errors:
            errors['parameter_error'] = np.linalg.norm(param_errors)
        else:
            errors['parameter_error'] = 0.0
            
        return errors
    
    def update_errors(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor,
                     learned_params: Dict = None, scalers: Dict = None,
                     losses: Dict = None):
        """
        Update all error metrics for the current epoch.
        
        Args:
            epoch: Current training epoch
            y_pred: Predicted solutions
            y_true: True solutions
            learned_params: Dictionary with learned parameters
            scalers: Dictionary with data scalers
            losses: Dictionary with loss components
        """
        # Store epoch
        self.error_history['epoch'].append(epoch)
        
        # Compute MSE errors
        mse_errors = self.compute_mse_errors(y_pred, y_true, scalers)
        for key, value in mse_errors.items():
            self.error_history[key].append(value)
        
        # Compute L2 errors
        l2_errors = self.compute_l2_errors(y_pred, y_true)
        for key, value in l2_errors.items():
            self.error_history[key].append(value)
        
        # Compute parameter errors
        if learned_params:
            param_errors = self.compute_parameter_errors(learned_params)
            for key, value in param_errors.items():
                self.error_history[key].append(value)
        else:
            for key in ['parameter_error', 'du_error', 'dv_error', 'f_error', 'k_error']:
                self.error_history[key].append(0.0)
        
        # Store loss components
        if losses:
            for loss_type in ['physics_loss', 'data_loss', 'ic_loss', 'bc_loss', 'total_loss']:
                value = losses.get(loss_type, 0.0)
                if torch.is_tensor(value):
                    value = value.item()
                self.error_history[loss_type].append(value)
        else:
            for loss_type in ['physics_loss', 'data_loss', 'ic_loss', 'bc_loss', 'total_loss']:
                self.error_history[loss_type].append(0.0)
    
    def save_errors_to_csv(self, filename: str = None):
        """
        Save error history to CSV file.
        
        Args:
            filename: Output filename. If None, generates timestamp-based name.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"error_tracking_{timestamp}.csv"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Ensure directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.error_history)
        df.to_csv(filepath, index=False)
        print(f"Error tracking saved to: {filepath}")
        
        return filepath
    
    def get_latest_errors(self) -> Dict[str, float]:
        """
        Get the most recent error values.
        
        Returns:
            Dictionary with latest error metrics
        """
        if not self.error_history['epoch']:
            return {}
        
        latest_errors = {}
        for key, values in self.error_history.items():
            if values:
                latest_errors[key] = values[-1]
        
        return latest_errors
    
    def print_error_summary(self, epoch: int = None):
        """
        Print a summary of current errors.
        
        Args:
            epoch: Specific epoch to summarize. If None, uses latest.
        """
        if epoch is None:
            if not self.error_history['epoch']:
                print("No error data available.")
                return
            epoch_idx = -1
            epoch = self.error_history['epoch'][-1]
        else:
            try:
                epoch_idx = self.error_history['epoch'].index(epoch)
            except ValueError:
                print(f"Epoch {epoch} not found in error history.")
                return
        
        print(f"\n=== Error Summary - Epoch {epoch} ===")
        print(f"MSE (Physical):     {self.error_history['mse_physical'][epoch_idx]:.2e}")
        print(f"L2 Error:           {self.error_history['l2_norm_error'][epoch_idx]:.3f}")
        print(f"Parameter Error:    {self.error_history['parameter_error'][epoch_idx]:.3f}")
        
        if self.error_history['du_error'][epoch_idx] > 0:
            print(f"  Du Error:         {self.error_history['du_error'][epoch_idx]:.3f}")
        if self.error_history['dv_error'][epoch_idx] > 0:
            print(f"  Dv Error:         {self.error_history['dv_error'][epoch_idx]:.3f}")
        
        print(f"Physics Loss:       {self.error_history['physics_loss'][epoch_idx]:.2e}")
        print(f"Data Loss:          {self.error_history['data_loss'][epoch_idx]:.2e}")
        print(f"Total Loss:         {self.error_history['total_loss'][epoch_idx]:.2e}")
