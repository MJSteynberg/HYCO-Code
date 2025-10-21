import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import torch

def interpolate_phys_solution(data, phys_solution, nX=11):
    """
    Interpolate the phys solution onto the trajectory points for
    each time step using bilinear interpolation.

    Args:
    - data (tensor): A tensor of shape (T, N, 3)
                        containing the trajectories (x, y, u).
    - phys_solution (tensor): A tensor of shape (T, nX, nX)
                        containing the phys solution over time.

    Returns:
    - interpolated_solution (tensor): A tensor of shape (T, N)
                        containing the interpolated phys values
                        for each trajectory at each time step.
    """

    # Normalize query points across all time steps
    query_points = data[..., :2]  # Extract (x, y) coordinates
    query_normalized = query_points / 3
    query_normalized = query_normalized.unsqueeze(1)


    # Prepare phys_solution for batch processing
    grid_values = phys_solution.unsqueeze(1)  # Shape: (T, 1, H, W)
    
    # Perform batch grid sampling
    interpolated = nn.functional.grid_sample(
        grid_values, query_normalized, mode='bilinear', align_corners=True
    )  # Output shape: (T, 1, N, 1)
    # Squeeze to match output shape
    interpolated_solution = interpolated.squeeze(1).squeeze(1)  # Shape: (T, N)

    return interpolated_solution



class Trainer:
    def __init__(self,
                 model_node,
                 model_phys,
                 optimizer_node,
                 optimizer_phys,
                 scheduler_anode,
                 scheduler_phys,
                 device,
                 grid,
                 print_freq=50, 
                 interaction = True, 
                 data_high = None, 
                 u0 = None):
        self.model_node = model_node.to(device)
        self.model_phys = model_phys.to(device)
        self.optimizer_node = optimizer_node
        self.optimizer_phys = optimizer_phys
        self.scheduler_anode = scheduler_anode
        self.scheduler_phys = scheduler_phys
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.grid = grid
        self.interaction = interaction 
        self.data_high = data_high 
        self.u0 = u0
        x = torch.linspace(-3, 3, 50).to(device)
        y = torch.linspace(-3, 3, 50).to(device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        grid_high = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1)
        mask = grid_high[:, 0] ** 2 + grid_high[:, 1] ** 2 < 3**2
        grid_high = grid_high[mask]
        u0_init = interpolate_phys_solution(grid_high.unsqueeze(0), u0.unsqueeze(0))
        self.init_high = torch.cat([grid_high, u0_init.T], dim=1)
        

    def train(self, u, num_epochs):
        u_train = (u[0, :, :].to(self.device))
        u_target = u[1:, :, :].to(self.device)
        errors_syn = torch.zeros(num_epochs).to(self.device)
        errors_phys = torch.zeros(num_epochs).to(self.device)
        gp = 20
        self.start_time = time.time()
        if self.interaction:
            loss = 1
            epoch = 0
            #train trajectories first
            self.optimizer_node.param_groups[0]['lr'] = 1e-2
            while loss > 5e-3:
                loss = self._train_epoch_trajectory(u_train, u_target)
                epoch += 1
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1} | Loss: {loss:.3e} | Time: {elapsed_time:.1f} min")

            # Train model
            self.optimizer_node.param_groups[0]['lr'] = 1e-3
            
            
            for epoch in range(num_epochs):
                loss, loss_FD, mse_syn, mse_phys = self._train_epoch(u_train, u_target, gp)
                errors_syn[epoch] = mse_syn
                errors_phys[epoch] = mse_phys
                if epoch == 1500:
                    print("Changing ghost points to 800") 
                    gp = 800
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.3e} | Loss_FD: {loss_FD:.3e}| MSE_syn: {mse_syn:.3e} | MSE_phys: {mse_phys:.3e} | Time: {elapsed_time:.1f} min | "
                        f"Params: {[x for x in self.model_phys.parameters()]}")

            print(f"Total time: {time.time() - self.start_time:.1f} sec")
            
        else:
            for epoch in range(num_epochs):
                loss, mse = self._train_epoch_nointeraction(u_target)
                errors_phys[epoch] = mse
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.3e} | MSE: {mse:.3e}| Time: {elapsed_time:.1f} min | "
                        f"Params: {[x for x in self.model_phys.parameters()]}")
                    
                
            print(f"Total time: {time.time() - self.start_time:.1f} sec")

        return np.array([x.detach().cpu().numpy() for x in self.model_phys.parameters()]), errors_syn.cpu().numpy(), errors_phys.cpu().numpy()
    
        
    def _train_epoch_trajectory(self, u_train, u_target):
        traj = self.model_node.g(u_train)
        
        
        # Calculate the loss
        loss = (100 * self.loss_func(traj, u_target[:,:,:2]))
        

        # Optimizer steps
        self.optimizer_node.zero_grad()
        loss.backward()
        self.optimizer_node.step()

        return loss.item()
    
    def _train_epoch_nointeraction(self, u_target):
        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(self.u0)

        interpolated_phys_target = interpolate_phys_solution(u_target, phys_solution)
        

        # Calculate the loss
        loss_FD = self.loss_func(interpolated_phys_target, u_target[:, :, 2])
        loss = 100 * loss_FD + 1e6 * self.model_phys.penalization()
        # Optimizer steps
        self.optimizer_phys.zero_grad()
        loss.backward()
        self.optimizer_phys.step()
        self.scheduler_phys.step()

        # Compute mse on a grid with high resolution data
        if self.data_high is not None:
            with torch.no_grad():
                data_interp = F.interpolate(self.data_high.unsqueeze(1), size=(phys_solution.shape[1:]), mode='bilinear', align_corners=True)
                data_interp = data_interp.squeeze(1)  # shape (99, 100, 100)
                mse_phys = torch.linalg.norm(torch.linalg.norm(phys_solution.reshape(99, -1) - data_interp.reshape(99, -1), ord=2, dim=1)) / torch.linalg.norm(torch.linalg.norm(data_interp.reshape(99, -1), ord=2, dim=1))

        else:
            mse_phys = 0

        return loss.item(), mse_phys
        
        
        
    
    def _train_epoch(self, u_train, u_target, gp = 20):
        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(self.u0)

        # Compute trajectory predicted by NODE
        traj = self.model_node.f(u_train)

        # # Find node prediction on random points to compare with fd
        forward_random_points = 0.7*(3 * torch.rand((gp, 2)).to(self.device) - 3)
        u0_init = interpolate_phys_solution(forward_random_points.unsqueeze(0), self.u0.unsqueeze(0))
        init = torch.cat((forward_random_points, u0_init.T), dim=1)
        grid_traj_forward = self.model_node.f(init)

        # Interpolate phys at trajectories
        interpolated_phys_traj = interpolate_phys_solution(grid_traj_forward, phys_solution)

        
        
        interpolated_phys_target = interpolate_phys_solution(u_target, phys_solution)
        

        # Calculate the loss
        if interpolated_phys_target.all() < 1e2:
            loss_FD = self.loss_func(interpolated_phys_target, u_target[:, :, 2])
            loss_hybrid = self.loss_func(interpolated_phys_traj, grid_traj_forward[:,:,2])
        else:
            loss_FD = self.model_phys.stabilize()
            loss_hybrid = 0
        loss = (100 * self.loss_func(traj, u_target)
                 + self.model_phys.penalization()
                + 100 * loss_FD)
       
        loss_1 = loss + 1e2 * loss_hybrid
        loss_2 = loss + 1e-2 * loss_hybrid
        # Optimizer steps
        self.optimizer_node.zero_grad()
        loss_1.backward()
        self.optimizer_node.step(loss_1)
        self.optimizer_phys.zero_grad()
        loss_2.backward()
        self.optimizer_phys.step(loss_2)
        self.scheduler_anode.step()
        self.scheduler_phys.step()

        # Compute mse on a grid with high resolution data
        if self.data_high is not None:
            with torch.no_grad():
                predicted_high = self.model_node.f(self.init_high)
                interpolated_high = interpolate_phys_solution(predicted_high, self.data_high)

                mse_syn = torch.linalg.norm(torch.linalg.norm(predicted_high[:, :, 2] - interpolated_high, ord=2, dim=1)) / torch.linalg.norm(torch.linalg.norm(interpolated_high, ord=2, dim=1))
                data_interp = F.interpolate(self.data_high.unsqueeze(1), size=(phys_solution.shape[1:]), mode='bilinear', align_corners=True)
                data_interp = data_interp.squeeze(1)  # shape (99, 100, 100)
                mse_phys = torch.linalg.norm(torch.linalg.norm(phys_solution.reshape(99, -1) - data_interp.reshape(99, -1), ord=2, dim=1)) / torch.linalg.norm(torch.linalg.norm(data_interp.reshape(99, -1), ord=2, dim=1))
        else:
            mse = 0

        return loss.item(), loss_FD.item(), mse_syn, mse_phys
