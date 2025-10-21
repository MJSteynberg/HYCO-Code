#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import DataLoader_Scalar
from models.generate_data import heat
import torch
from torch.utils.data import DataLoader, Subset
from models.plot import plot_results_separate
from models.FEM import AdvectionDiffusion, Heat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import torch
import torch.nn as nn
from models.training import interpolate_phys_solution
from torch.nn import functional as F
from models.NN import SpatioTemporalNN
import copy

device = torch.device('cuda')



class HeatNNTrainer:
    def __init__(self,
                 spatiotemporal_nn,
                 physics_model,
                 optimizer_nn,
                 optimizer_phys,
                 scheduler_nn,
                 scheduler_phys,
                 device,
                 grid,
                 print_freq=50, 
                 interaction=True, 
                 data_high=None):
        self.spatiotemporal_nn = spatiotemporal_nn.to(device)
        self.physics_model = physics_model.to(device)
        self.optimizer_nn = optimizer_nn
        self.optimizer_phys = optimizer_phys
        self.scheduler_nn = scheduler_nn
        self.scheduler_phys = scheduler_phys
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.grid = grid
        self.interaction = interaction
        self.data_high = data_high

    def train(self, u_data, u0, num_epochs, hybrid=True, lambdas=None):
        """
        Train the hybrid model
        
        Args:
            u_data: Ground truth data (T, N, 3) where last dim is [x, y, u]
            u0: Initial condition for physics solver
            num_epochs: Number of training epochs
            indices: Training indices
            alpha_real: True physics parameters
            alpha_init: Initial physics parameters
            hybrid: Whether to use hybrid training
        """
        mse_errors_syn = torch.zeros(num_epochs).to(self.device)
        mse_errors_phys = torch.zeros(num_epochs).to(self.device)
        p_errors_syn = torch.zeros(num_epochs).to(self.device)
        p_errors_phys = torch.zeros(num_epochs).to(self.device)
        l2_errors_syn = torch.zeros(num_epochs).to(self.device)
        l2_errors_phys = torch.zeros(num_epochs).to(self.device)
        params_history = []
        
        self.start_time = time.time()
        
        if hybrid:
            # Phase 1: Pre-train neural network on trajectories
            print("Phase 1: Pre-training neural network...")
            self._pretrain_nn(u_data, num_pretrain_epochs=0)
            
            # Phase 2: Hybrid training
            print("Phase 2: Hybrid training...")
            for epoch in range(num_epochs):
                loss, loss_phys, mse_syn, mse_phys, p_syn, p_phys, l2_syn, l2_phys = self._train_epoch_hybrid(
                    u_data, u0, epoch, lambdas)

                mse_errors_syn[epoch] = mse_syn
                mse_errors_phys[epoch] = mse_phys
                p_errors_syn[epoch] = p_syn
                p_errors_phys[epoch] = p_phys
                l2_errors_syn[epoch] = l2_syn
                l2_errors_phys[epoch] = l2_phys
                params_history.append([p.clone().detach() for p in self.physics_model.parameters()])
                
                if (epoch + 1) % self.print_freq == 0:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    current_params = [p.item() if p.numel() == 1 else p.tolist() 
                                    for p in self.physics_model.parameters()]
                    print(f"Epoch {epoch + 1}/{num_epochs} | "
                          f"Loss: {loss:.3e} | Loss_phys: {loss_phys:.3e} | "
                          f"MSE_syn: {mse_syn:.3e} | MSE_phys: {mse_phys:.3e} | "
                          f"Time: {elapsed_time:.1f} min | "
                          f"Params: {current_params}")
            # Return final parameters
            final_params = torch.cat([p.flatten() for p in self.physics_model.parameters()])
            return mse_errors_syn.detach().cpu().numpy(), mse_errors_phys.detach().cpu().numpy(), p_errors_phys.detach().cpu().numpy(), p_errors_syn.detach().cpu().numpy(), l2_errors_phys.detach().cpu().numpy(), l2_errors_syn.detach().cpu().numpy(), final_params.detach().cpu().numpy()
        else:
            # Physics-only training
            for epoch in range(num_epochs):
                loss, mse_phys, p_phys, l2_phys = self._train_epoch_physics_only(u_data, u0)
                mse_errors_phys[epoch] = mse_phys
                p_errors_phys[epoch] = p_phys
                l2_errors_phys[epoch] = l2_phys
                params_history.append([p.clone().detach() for p in self.physics_model.parameters()])

                if (epoch + 1) % self.print_freq == 0:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    current_params = [p.item() if p.numel() == 1 else p.tolist() 
                                    for p in self.physics_model.parameters()]
                    print(f"Epoch {epoch + 1}/{num_epochs} | "
                          f"Loss: {loss:.3e} | MSE: {mse_phys:.3e} | "
                          f"Time: {elapsed_time:.1f} min | "
                          f"Params: {current_params}")
                    
            # Return final parameters
            final_params = torch.cat([p.flatten() for p in self.physics_model.parameters()])
            return mse_errors_phys.detach().cpu().numpy(), p_errors_phys.detach().cpu().numpy(), l2_errors_phys.detach().cpu().numpy(), final_params.detach().cpu().numpy()
        
        

    def _pretrain_nn(self, u_data, num_pretrain_epochs=0):
        """Pre-train the neural network to learn the spatiotemporal field"""
        T, N, _ = u_data.shape
        
        # Create time tensor for all time steps
        t_all = torch.linspace(0, 1, T).to(self.device)  # normalized time
        
        for epoch in range(num_pretrain_epochs):
            # Use the entire dataset
            total_loss = 0
            num_batches = 0
            
            # Process all time steps
            for t_idx in range(T):
                # Get all spatial points at current time step
                x = u_data[t_idx, :, 0]  # all x coordinates at time t_idx
                y = u_data[t_idx, :, 1]  # all y coordinates at time t_idx
                u_true = u_data[t_idx, :, 2]  # all field values at time t_idx
                t = t_all[t_idx].expand_as(x)  # broadcast time to match batch size
                
                # Forward pass
                u_pred = self.spatiotemporal_nn(x, y, t).squeeze()
                
                # Loss
                loss = self.loss_func(u_pred, u_true)
                total_loss += loss.item()
                num_batches += 1
                
                # Backward pass
                self.optimizer_nn.zero_grad()
                loss.backward()
                self.optimizer_nn.step()
            
            # Average loss over all batches
            avg_loss = total_loss / num_batches
            
            if epoch % 100 == 0:
                print(f"Pre-train epoch {epoch}: Avg Loss = {avg_loss:.6f}")
    
    def _train_epoch_hybrid(self, u_data, u0, epoch, lambdas):
        """Single epoch of hybrid training"""
        T, N, _ = u_data.shape
        
        # 2. Generate ghost points for consistency training
        num_ghost = 20 if epoch < 1500 else 800 
        ghost = 3 * torch.rand(num_ghost, 2).to(self.device) - 3  # Random ghost points in [-3, 3]
        ghost_forward = ghost.unsqueeze(0).expand(T, -1, -1)  # Shape: (T, num_ghost, 2)

        ghost_x = ghost_forward[:, :, 0].reshape(-1)  # Shape: (T * num_ghost,)
        ghost_y = ghost_forward[:, :, 1].reshape(-1)  # Shape:
        ghost_t = torch.linspace(0, 1, T).to(self.device).unsqueeze(1).expand(-1, num_ghost).reshape(-1)  # Shape: (T * num_ghost,)



        # 6. VECTORIZED Data fitting loss (NN vs ground truth data)
        # Flatten spatial and temporal dimensions
        x_all = u_data[:, :, 0].flatten()  # Shape: (T*N,)
        y_all = u_data[:, :, 1].flatten()  # Shape: (T*N,)
        u_true_all = u_data[:, :, 2].flatten()  # Shape: (T*N,)
        
        # Create time coordinates for all points
        t_data = torch.linspace(0, 1, T).to(self.device)
        t_all = t_data.unsqueeze(1).expand(T, N).flatten()  # Shape: (T*N,)
        
        # Single forward pass for all data points
        nn_pred_all = self.spatiotemporal_nn(x_all, y_all, t_all).squeeze()
        
        # Compute loss
        loss_data = self.loss_func(nn_pred_all, u_true_all)
        phys_solution = self.physics_model(u0)  # Shape: (T, nX, nX)
        
        def loss_p():
            """Compute total loss for physics model"""
            self.optimizer_phys.zero_grad()
            # 1. Compute physics solution
            phys_solution_ = self.physics_model(u0)  # Shape: (T, nX, nX)
            # 3. Get NN predictions at ghost points
            nn_pred = self.spatiotemporal_nn(ghost_x, ghost_y, ghost_t).squeeze()
            nn_pred = nn_pred.reshape(T, num_ghost)[1:]  # Shape: (T, num_ghost)
            
            # 4. Interpolate physics solution at ghost points
            # 7. Physics loss (physics vs ground truth)
            phys_pred_ = interpolate_phys_solution(ghost_forward[1:], phys_solution_) 
            interpolated_phys_data = interpolate_phys_solution(u_data[1:], phys_solution_)
            # 5. Compute consistency loss (NN vs Physics)
            loss_consistency = self.loss_func(nn_pred, phys_pred_)
            loss_phys = self.loss_func(interpolated_phys_data, u_data[1:, :, 2])
            loss = 100 * loss_phys + lambdas[0] * loss_consistency + self.physics_model.penalization()
            loss.backward()
            return loss
        # 8. Total loss
        def loss_nn():
            """Compute total loss for neural network"""
            self.optimizer_nn.zero_grad()
            # 1. Compute physics solution
            
            # 3. Get NN predictions at ghost points
            nn_pred = self.spatiotemporal_nn(ghost_x, ghost_y, ghost_t).squeeze()
            nn_pred = nn_pred.reshape(T, num_ghost)[1:]  # Shape: (T, num_ghost)
            
            # 4. Interpolate physics solution at ghost points
            # 7. Physics loss (physics vs ground truth)
            
            phys_pred = interpolate_phys_solution(ghost_forward[1:], phys_solution) 
            # 5. Compute consistency loss (NN vs Physics)
            loss_consistency = self.loss_func(nn_pred, phys_pred)
            loss = 100 * loss_data + lambdas[1] * loss_consistency
            loss.backward()  # Retain graph for physics loss
            return loss
        

        
        # 9. Optimization

        self.optimizer_nn.step(loss_nn)
        self.optimizer_phys.step(loss_p)
        self.scheduler_nn.step()
        self.scheduler_phys.step()
        
        
        # 10. Compute MSE for monitoring
        with torch.no_grad():
            interpolated_phys_data = interpolate_phys_solution(u_data[1:], phys_solution)
            loss_phys = self.loss_func(interpolated_phys_data, u_data[1:, :, 2])
            if self.data_high is not None:
                mse_syn = loss_data.item() 
                mse_phys = loss_phys.item()
                p_syn = 0
                p_phys = self._compute_phys_ploss()
                l2_syn = self._compute_nn_mse()
                l2_phys = self._compute_physics_mse(phys_solution)

            else:
                mse_syn = loss_data.item()
                mse_phys = loss_phys.item()
        
        return loss_data.item(), loss_phys.item(), mse_syn, mse_phys, p_syn, p_phys, l2_syn, l2_phys

    
    def _train_epoch_physics_only(self, u_data, u0):
        """Single epoch of physics-only training"""
        # Compute physics solution
        phys_solution = self.physics_model(u0)
        
        interpolated_phys_data = interpolate_phys_solution(u_data[1:], phys_solution)
        loss_phys = self.loss_func(interpolated_phys_data, u_data[1:, :, 2])

        loss_total = 100 * loss_phys + self.physics_model.penalization()
        
        # Optimization
        self.optimizer_phys.zero_grad()
        loss_total.backward()
        self.optimizer_phys.step()
        self.scheduler_phys.step()
        
        # MSE
        with torch.no_grad():
            mse_phys = loss_phys.item()
            p_phys = self._compute_phys_ploss()
            l2_phys = self._compute_physics_mse(phys_solution) if self.data_high is not None else 0.0

        return loss_total.item(), mse_phys, p_phys, l2_phys

    def _compute_nn_mse(self):
        """VECTORIZED: Compute MSE for neural network against high-res data"""
        if self.data_high is not None:
            T_high, H_high, W_high = self.data_high.shape
            
            # Create a grid that matches the high-res data spatial resolution
            x_high = torch.linspace(-3, 3, W_high).to(self.device)
            y_high = torch.linspace(-3, 3, H_high).to(self.device)
            grid_x, grid_y = torch.meshgrid(x_high, y_high, indexing='ij')
            x_flat = grid_x.reshape(-1)  # Shape: (H_high * W_high,)
            y_flat = grid_y.reshape(-1)  # Shape: (H_high * W_high,)
            
            # Create time steps that match the high-res data
            t_high = torch.linspace(0, 1, T_high).to(self.device)
            
            # VECTORIZED: Create all spatiotemporal coordinates at once
            # Expand spatial coordinates for all time steps
            x_all = x_flat.unsqueeze(0).expand(T_high, -1).flatten()  # Shape: (T_high * H_high * W_high,)
            y_all = y_flat.unsqueeze(0).expand(T_high, -1).flatten()  # Shape: (T_high * H_high * W_high,)
            
            # Expand temporal coordinates for all spatial points
            t_all = t_high.unsqueeze(1).expand(-1, H_high * W_high).flatten()  # Shape: (T_high * H_high * W_high,)
            
            # Single forward pass for ALL spatiotemporal points
            nn_pred_all = self.spatiotemporal_nn(x_all, y_all, t_all).squeeze()
            
            # Reshape predictions to match data structure
            nn_pred_reshaped = nn_pred_all.reshape(T_high, H_high, W_high)
            
            # Compute MSE across all points at once
            mse = torch.linalg.norm(
                torch.linalg.norm(nn_pred_reshaped.reshape(T_high, -1) - 
                                  self.data_high.reshape(T_high, -1), ord=2, dim=1)
            ) / torch.linalg.norm(
                torch.linalg.norm(self.data_high.reshape(T_high, -1), ord=2, dim=1)
            )
            
            return mse.item()

        return 0.0
    def _compute_phys_ploss(self):
        params = self.physics_model.alpha.detach().cpu().numpy()
        ampl = params[0:2]
        centres = params[2:6]
        
        alpha_real = np.array([3, 2.5, 1, -2, 1, -2, 1.0, 1.0])

        p_loss =  np.linalg.norm(ampl - alpha_real[:2]) / np.linalg.norm(alpha_real[:2]) + \
                     np.linalg.norm(centres - alpha_real[2:6]) / np.linalg.norm(alpha_real[2:6])
        return p_loss

    def _compute_physics_mse(self, phys_solution):
        """Compute MSE for physics model against high-res data"""
        if self.data_high is not None:
            data_interp = F.interpolate(
                self.data_high.unsqueeze(1), 
                size=phys_solution.shape[1:], 
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            
            mse = torch.linalg.norm(
                torch.linalg.norm(phys_solution.reshape(len(phys_solution), -1) - 
                                data_interp.reshape(len(data_interp), -1), ord=2, dim=1)
            ) / torch.linalg.norm(
                torch.linalg.norm(data_interp.reshape(len(data_interp), -1), ord=2, dim=1)
            )
            return mse.item()
        return 0.0

    def save_model(self, model_path_nn, model_path_phys, hyperparams):
        """Save the neural network, physics model, and hyperparameters to files."""
        torch.save({
            'state_dict': self.spatiotemporal_nn.state_dict(),
            'hyperparameters': hyperparams
        }, model_path_nn)
        torch.save({
            'state_dict': self.physics_model.state_dict(),
            'hyperparameters': hyperparams
        }, model_path_phys)
        print(f"Models and hyperparameters saved to {model_path_nn} and {model_path_phys}")

    def load_model(self, model_path_nn, model_path_phys):
        """Load the neural network, physics model, and hyperparameters from files."""
        nn_checkpoint = torch.load(model_path_nn, map_location=self.device)
        self.spatiotemporal_nn.load_state_dict(nn_checkpoint['state_dict'])
        phys_checkpoint = torch.load(model_path_phys, map_location=self.device)
        self.physics_model.load_state_dict(phys_checkpoint['state_dict'])
        print(f"Models loaded from {model_path_nn} and {model_path_phys}")
        return nn_checkpoint['hyperparameters']  # Return hyperparameters


def setup(split, nX, L, folder, num_gaussians, alpha = None, hidden_dim = 1000, learning_rate = 1e-3, num_epochs = 2000, interaction = True, flag = "full"):
    # Load data and split
    data = DataLoader_Scalar(device, folder, flag)
    train_size = int(split*data.length_u())
    indices = torch.randperm(data.length_u())[:train_size]

    # High definition solution
    data_high = np.load(folder + f'/phys_{flag}.npy')
    data_high = torch.from_numpy(data_high).float().to(device)

    # Obtain variables from data
    u_train = data.u[:, indices, :]
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    data_dim = data.u.shape[2]
    T = data.t[-1]
    num_steps = data.length_t()
    dt = data.t[1] - data.t[0]

    # create models, optimizers and schedulers
    syn = SpatioTemporalNN(hidden_dim=hidden_dim, num_layers=4, activation='relu').to(device)
    phys = Heat(device, L, nX, dt, num_steps, num_gaussians, alpha = alpha)

    optimizer_node = torch.optim.Adam(syn.parameters(), lr=learning_rate)
    optimizer_phys = torch.optim.Adam(phys.parameters(), lr=1e-3, betas = (0.5, 0.99999))

    scheduler_node = torch.optim.lr_scheduler.OneCycleLR(optimizer_node, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_phys = torch.optim.lr_scheduler.OneCycleLR(optimizer_phys, max_lr=1e-2, steps_per_epoch=1, epochs=num_epochs)

    # Print number of parameters
    print(f"Number of parameters in SpatioTemporalNN: {sum(p.numel() for p in syn.parameters() if p.requires_grad)}")

    # create trainer 
    trainer = HeatNNTrainer(
        spatiotemporal_nn=syn,
        physics_model=phys,
        optimizer_nn=optimizer_node,
        optimizer_phys=optimizer_phys,
        scheduler_nn=scheduler_node,
        scheduler_phys=scheduler_phys,
        device=device,
        grid=grid,
        print_freq=50, 
        interaction=interaction, 
        data_high=data_high
    )

    return u_train, indices, u0, trainer

def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)], dim=1)
    x, y = grid[:, 0], grid[:, 1]
    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped



if __name__ == '__main__':

    for flag in ['quarter', 'half', 'full']:
        print("---------------------------------")
        print("--------Generating data----------")
        print("---------------------------------")
        # clear the folder
        # Generate data
        if flag == 'full':
            heat(flag="full", L = 6, folder = 'data/heat/table')
            lambdas = (1e-1, 1e2)
        elif flag == 'half':
            heat(flag="half", L = 6, folder = 'data/heat/table')
            lambdas = (1, 1e1)
        elif flag == 'quarter':
            heat(flag="quarter", L = 6, folder = 'data/heat/table')
            lambdas = (1e1, 1e1)
        else:
            raise ValueError("Invalid flag value. Choose from ['full', 'half', 'quarter']")


        # Common parameters: 
        split = 0.6
        nX = 18
        L = 6
        folder = 'data/heat/table'
        num_gaussians = 2
        alpha_real = torch.tensor([3, 2.5, 1, -2, 1, -2, 1.0, 1.0]).float()

        alpha = torch.tensor([1, 1, 1.4, -1.8, -1.3, -1.1, 1.0, 1.0]).float().to(device) # [Amplitude, x0, y0, sigma]
        hidden_dim = 256
        num_layers = 4
        learning_rate = 1e-3
        num_epochs = 3000

        u_train, indices, u0, trainer_phys = setup(split, nX, L, folder, num_gaussians, alpha.clone(), hidden_dim, learning_rate, num_epochs, interaction = False, flag=flag)
        _,_,_, trainer_hybrid = setup(split, nX, L, folder, num_gaussians, alpha.clone(), hidden_dim, learning_rate, num_epochs, interaction = True, flag=flag)
        print("--------------------------------")
        print("--------Setup finished----------")

        # Define hyperparameters
        hyperparams = {
            'split': split,
            'nX': nX,
            'L': L,
            'folder': folder,
            'num_gaussians': num_gaussians,
            'alpha': alpha.tolist(),
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'interaction': False
        }

        print("--------------------------------")
        print("------- Hybrid training: -------")
        print("--------------------------------")
        t1_hyb = time.time()
        mse_errors_syn, mse_errors_phys, p_errors_phys, p_errors_syn, l2_errors_phys, l2_errors_syn, final_params  = trainer_hybrid.train(u_train, u0, num_epochs, hybrid=True, lambdas=lambdas)
        t2_hyb = time.time()

        print("--------------------------------")
        print("------- Physics training: ------")
        print("--------------------------------")
        t1_phys = time.time()
        mse_errors_phys_only, p_errors_phys_only, l2_errors_phys_only, final_params_phys_only = trainer_phys.train(u_train, u0, num_epochs, hybrid=False)
        t2_phys = time.time()


        # Combine all error arrays into one DataFrame
        all_errors = pd.DataFrame({
            'mse_hybrid_syn': mse_errors_syn.flatten(),
            'mse_hybrid_phys': mse_errors_phys.flatten(),
            'mse_phys_only': mse_errors_phys_only.flatten(),
            'p_error_hybrid_phys': p_errors_phys.flatten(),
            'p_error_hybrid_syn': p_errors_syn.flatten() if p_errors_syn is not None else np.zeros_like(p_errors_phys.flatten()),
            'p_error_phys_only': p_errors_phys_only.flatten(),
            'l2_error_hybrid_phys': l2_errors_phys.flatten(),
            'l2_error_hybrid_syn': l2_errors_syn.flatten(),
            'l2_error_phys_only': l2_errors_phys_only.flatten()
        })
        
        # Get the current date and time to include in the filename
        # Save all parameters to parameters folder using date and time in name
        params_real = alpha_real.clone().detach().cpu().numpy()
        params = pd.DataFrame(np.stack([params_real.flatten(), final_params.flatten(), final_params_phys_only.flatten()]).T, columns=["params_real", "params_hybrid", "params_phys"])
        train_indices = pd.DataFrame(indices, columns = ['training_indices'])
        
        # Get the current date and time to include in the filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save the DataFrame to CSV with the timestamp in the filename
        params.to_csv(f'parameters/heat/table/param_{timestamp}_{flag}.csv', index=False)
        train_indices.to_csv(f'parameters/heat/table/index_{timestamp}_{flag}.csv', index=False)
        all_errors.to_csv(f'parameters/heat/table/errors_{timestamp}_{flag}.csv', index=False)  
        # save the timings
        timings = pd.DataFrame(np.stack([np.array([t2_hyb - t1_hyb, t2_hyb - t1_hyb])]).T, columns=["timings"])
        timings.to_csv(f'parameters/heat/table/timings_{timestamp}_{flag}.csv', index=False)

        # save the models
        torch.save(trainer_hybrid.spatiotemporal_nn.state_dict(), f'parameters/heat/table/syn_{timestamp}_{flag}.pt')
        torch.save(trainer_hybrid.physics_model.state_dict(), f'parameters/heat/table/phys_{timestamp}_{flag}.pt')


