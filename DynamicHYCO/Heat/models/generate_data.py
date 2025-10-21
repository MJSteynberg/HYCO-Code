import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from models.FEM import AdvectionDiffusion, Heat
from models.training import interpolate_phys_solution
import numpy as np

# Define the ODE system
class SimpleHarmonicOscillator(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state)

class SimpleHarmonicOscillator2(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state - torch.tensor([1.5, 1.5], dtype=torch.float32))

class Oscillator1(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state - torch.tensor([0, 0], dtype=torch.float32))

class Oscillator2(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state - torch.tensor([0.75, 0.75], dtype=torch.float32))

class Oscillator3(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state - torch.tensor([1.5, 1.5], dtype=torch.float32))
    
class Stationary(torch.nn.Module):
    def forward(self, t, state):
        return torch.stack([0, 0], dim=0)

# Solve the ODE
def solve_ode_with_odeint(x0, y0, t, ode_func):
    """
    Solves the ODE system x' = y, y' = -x using torchdiffeq's odeint.
    
    Args:
    - x0, y0 (float): Initial conditions for x and y.
    - t (torch.Tensor): Time points for evaluation.
    
    Returns:
    - torch.Tensor: Solution (len(t), 2) where [:, 0] is x(t) and [:, 1] is y(t).
    """
    # Initial conditions as a torch.Tensor
    initial_state = torch.tensor([x0, y0], dtype=torch.float32)
    
    
    # Solve the ODE
    solution = odeint(ode_func, initial_state, t)
    return solution

def heat(flag = "full", L = 6.0, folder = 'data/heat/table'):
    import os
    # Check if files exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    data_path = f'{folder}/data_{flag}.npy'
    phys_path = f'{folder}/phys_{flag}.npy'
    heat_sol_path = f'{folder}/heat_solution_{flag}.npy'
    if os.path.exists(data_path) and os.path.exists(phys_path) and os.path.exists(heat_sol_path):
        print(f"Data files already exist in {folder} for flag '{flag}', skipping generation.")
        return

    num_steps = 10000
    N = 100
    T = 1
    num_gaussians = 2
    alpha = torch.tensor([3, 2.5, 1, -2, 1, -2, 1.0, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]

    dt = T / num_steps
    sqrt_num_sensors = 10 # 
    device = "cpu"

    heat = Heat(device, L, N, dt, num_steps + 1, num_gaussians, alpha = alpha)

    # Create a sum of gaussians initial condition
    x = torch.linspace(-L//2,L//2, N)
    y = torch.linspace(-L//2,L//2, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    heat_solution = heat(u0)
    
    hs = heat_solution.detach().numpy()
    np.save(heat_sol_path, hs)

    if flag == "full":
        # Create sensors in [-3,3]
        a = torch.linspace(-3, 3, sqrt_num_sensors)
        b = torch.linspace(-3, 3, sqrt_num_sensors)
        x0, y0 = torch.meshgrid(a, b, indexing='ij')
        x0 = x0.flatten()
        y0 = y0.flatten()
        ode_func = Oscillator1()
    elif flag == "half":
        # Create sensors in [-1.5,3]
        a = torch.linspace(-1.5, 3, sqrt_num_sensors)
        b = torch.linspace(-1.5, 3, sqrt_num_sensors)
        x0, y0 = torch.meshgrid(a, b, indexing='ij')
        x0 = x0.flatten()
        y0 = y0.flatten()
        ode_func = Oscillator2()
    else:
        # Create sensors in [-1.5,1.5]
        a = torch.linspace(0, 3, sqrt_num_sensors)
        b = torch.linspace(0, 3, sqrt_num_sensors)
        x0, y0 = torch.meshgrid(a, b, indexing='ij')
        x0 = x0.flatten()
        y0 = y0.flatten()
        ode_func = Oscillator3()
        

    data_pts = x0.shape[0]

    data = torch.empty((num_steps, data_pts, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(data_pts):
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t, ode_func)

    data[:,:,2] = interpolate_phys_solution(data, heat_solution)
    
    #extract every fifth data point
    data = data[::100,:,:]
    heat_solution = heat_solution.detach().numpy()[::100,:,:]
    heat_solution = heat_solution[1:]
    # save the data 
    
    np.save(f'{folder}/data_{flag}.npy', data.detach().numpy())
    np.save(f'{folder}/phys_{flag}.npy', heat_solution)
    