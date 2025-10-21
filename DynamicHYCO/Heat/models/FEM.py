import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
    

class Heat(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, num_gaussians, alpha = None):
        super(Heat, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.num_params = 4 * num_gaussians # pos_x, pos_y, sigma, amplitude
        self.num_gaussians = num_gaussians
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape[0] != (self.num_params):
                print(self.alpha.shape, self.num_params)
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((self.num_params), dtype=torch.float32, device=self.device))

        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = 0.1*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        x, y = torch.meshgrid(x, y)
        for i in range(self.num_gaussians):
            diffusion_map += torch.exp(torch.log(self.alpha[i])) * torch.exp(-((x - self.alpha[self.num_gaussians + i]) ** 2 + (y - self.alpha[2*self.num_gaussians + i]) ** 2))
        diffusion_map = torch.clamp(diffusion_map, min=0.0, max=2.5)
  
        return diffusion_map
    
    def penalization(self):
        return 0

    def step(self, T):
        dx2 = self.dx ** 2

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        alpha = self.diffusion_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.dt * alpha * (T_xx + T_yy)

        # Apply Dirichlet boundary conditions
        T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = 0.0

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
class AdvectionDiffusion(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = None, kappa = None):
        super(AdvectionDiffusion, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.num_params_alpha = 4 * num_gaussians_alpha # pos_x, pos_y, sigma, amplitude
        self.num_gaussians_alpha = num_gaussians_alpha
        self.num_params_kappa = 4 * num_gaussians_kappa # pos_x, pos_y, sigma, amplitude
        self.num_gaussians_kappa = num_gaussians_kappa
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape[0] != (self.num_params_alpha):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(0.1*torch.randn((self.num_params_alpha), dtype=torch.float32, device=self.device) + 1)

        if kappa is not None:
            self.kappa = nn.Parameter(kappa)
            if self.kappa.shape[0] != (self.num_params_kappa):
                raise ValueError("Shape of kappa does not match the parameters.")
        else:
            self.kappa = nn.Parameter(0.1*torch.randn((self.num_params_alpha), dtype=torch.float32, device=self.device) + 1)
        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = 0.1*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians_alpha):
            diffusion_map += self.alpha[i] * torch.exp(-((x - self.alpha[self.num_gaussians_alpha + i]) ** 2 + (y - self.alpha[2*self.num_gaussians_alpha + i]) ** 2))
        return diffusion_map
    
    def create_advection_map(self):
        # interpolate alpha to the grid
        advection_map = 0.1*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians_kappa):
            advection_map += self.kappa[i] * torch.exp(-((x - self.kappa[self.num_gaussians_kappa + i]) ** 2 + (y - self.kappa[2*self.num_gaussians_kappa + i]) ** 2))
        return advection_map
    
    def penalization(self):
        return 1e6*(torch.relu(-self.alpha[0]) + torch.relu(-self.kappa[0]))
    
    def stabilize(self):
        return 0.1*(torch.nn.MSELoss()(self.kappa[0], -self.kappa[0]) + torch.nn.MSELoss()(self.alpha[0], -self.alpha[0]))

    def step(self, T):
        
        # t is not time, just time step
        dx2 = self.dx ** 2
        dx = self.dx

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        T_x = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dx)
        T_y = (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
        
        alpha = self.diffusion_map[1:-1, 1:-1]
        kappa = self.advection_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        

         # Burgers' term + Diffusion + Reaction
        T_new[1:-1, 1:-1] = (T[1:-1, 1:-1]
            - self.dt * kappa * (T[1:-1, 1:-1] * T_x + T[1:-1, 1:-1] * T_y)   # Advection (Burgers')
            + self.dt * alpha * (T_xx + T_yy)  # Diffusion
        )

        # Apply Neumann boundary conditions
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]
        

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        self.advection_map = self.create_advection_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
    
"""
Incompressible Euler step function from Chatgpt:



def step(self, rho, u):
    # Copy the inputs for updating
    rho_new = rho.copy()
    u_new = u.copy()

    dx = self.dx
    dt = self.dt

    # Compute finite differences
    rho_x = (rho[2:] - rho[:-2]) / (2 * dx)
    u_x = (u[2:] - u[:-2]) / (2 * dx)
    rho_xx = (rho[:-2] - 2 * rho[1:-1] + rho[2:]) / dx**2
    u_xx = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2

    # Compute source terms
    a_rho_x = (self.a_func[2:] * rho[2:] - self.a_func[:-2] * rho[:-2]) / (2 * dx)
    drag = 0.5 * self.theta * u[1:-1] * np.abs(u[1:-1]) / rho[1:-1]
    gravity = self.g * rho[1:-1] * np.sin(self.alpha)

    # Continuity equation
    rho_new[1:-1] = rho[1:-1] - dt / (2 * dx) * (u[2:] - u[:-2])

    # Momentum equation
    convection_term = (
        u[1:-1] / rho[1:-1] * (1 + 1 / rho[1:-1]) * u_x
        - u[1:-1]**2 / rho[1:-1]**2 * rho_x
    )
    u_new[1:-1] = (
        u[1:-1]
        - dt * (convection_term + a_rho_x + drag + gravity)
    )

    # Neumann boundary conditions
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]

    return rho_new, u_new


"""