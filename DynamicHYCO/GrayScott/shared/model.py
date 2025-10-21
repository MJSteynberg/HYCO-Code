import torch
import numpy as np
import torch.nn as nn


class GrayScottModel(nn.Module):
    def __init__(self, width=256, height=256, domain_width=1.0, domain_height=1.0,
                 Du=0.16, Dv=0.08, f=0.02, k=0.051, dt=None, device='cpu'):
        """
        Initialize Gray-Scott reaction-diffusion model as PyTorch nn.Module

        Parameters:
        - width, height: Grid resolution (number of points)
        - domain_width, domain_height: Physical domain size
        - Du, Dv: Diffusion rates for U and V (trainable parameters)
        - f: Feed rate
        - k: Kill rate
        - dt: Time step
        - device: 'cpu' or 'cuda'
        """
        super(GrayScottModel, self).__init__()

        self.width = width
        self.height = height
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.device = device

        # Calculate grid spacing
        self.dx = domain_width / (width - 1)
        self.dy = domain_height / (height - 1)
        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2

        # Register diffusion parameters as trainable parameters
        # Use log-space parameterization to ensure positivity and better optimization
        Du = max(0.001, min(Du, 1.0))  # Clamp between 0.001 and 1.0
        Dv = max(0.001, min(Dv, 1.0))  # Clamp between 0.001 and 1.0

        self.log_Du = nn.Parameter(torch.log(torch.tensor(Du, dtype=torch.float32)), requires_grad=True)
        self.log_Dv = nn.Parameter(torch.log(torch.tensor(Dv, dtype=torch.float32)),
                                   requires_grad=True)  # Corrected initialization

        # Other parameters (can also be made trainable if needed)
        self.f = nn.Parameter(torch.tensor(f, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32), requires_grad=False)

        # Store initial values for reset functionality
        self.initial_Du = Du
        self.initial_Dv = Dv
        self.initial_f = f
        self.initial_k = k

        if dt is None:
            max_D = max(Du, Dv)
            # More conservative stability condition
            self.dt = 0.1 * min(self.dx2, self.dy2) / (4 * max_D)
        else:
            self.dt = dt

        # Initialize concentrations - use nn.Parameter for state if you want gradients through time
        # For most applications, buffers are sufficient
        self.initial_width = width
        self.initial_height = height
        self.reset_state()

    @property
    def Du(self):
        """Get actual Du value (constrained to reasonable range)"""
        return torch.clamp(torch.exp(self.log_Du), 0.001, 1.0)

    @property
    def Dv(self):
        """Get actual Dv value (constrained to reasonable range)"""
        return torch.clamp(torch.exp(self.log_Dv), 0.001, 1.0)  # Corrected access

    def reset_state(self):
        """Reset the simulation state to initial conditions"""
        # Initialize concentrations
        U_init = torch.ones((1, 1, self.height, self.width), dtype=torch.float32)
        V_init = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32)

        # Add initial perturbation in the center
        center_x, center_y = self.width // 2, self.height // 2
        radius_physical = 0.1 * min(self.domain_width, self.domain_height)
        radius_grid = radius_physical / min(self.dx, self.dy)

        y, x = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing='ij')
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_grid ** 2
        U_init[0, 0, mask] = 0.5
        V_init[0, 0, mask] = 0.5

        # Store initial state for training
        self.U_init = U_init.to(self.device)
        self.V_init = V_init.to(self.device)

        # For inference, use buffers
        if hasattr(self, 'U'):
            self.U.data.copy_(U_init.to(self.device))
            self.V.data.copy_(V_init.to(self.device))
        else:
            self.register_buffer('U', U_init.to(self.device))
            self.register_buffer('V', V_init.to(self.device))

    def laplacian(self, Z):
        """
        Calculate 2D Laplacian using torch.roll for periodic boundary conditions.
        This is a fully differentiable and robust way to handle periodic boundaries.
        """
        # Z shape: (batch, channels, height, width) = (1, 1, H, W)

        # Shifted tensors for neighbors (periodic boundaries using torch.roll)
        Z_center = Z
        Z_up = torch.roll(Z, shifts=(-1,), dims=2)  # Shift up (y-1)
        Z_down = torch.roll(Z, shifts=(1,), dims=2)  # Shift down (y+1)
        Z_left = torch.roll(Z, shifts=(-1,), dims=3)  # Shift left (x-1)
        Z_right = torch.roll(Z, shifts=(1,), dims=3)  # Shift right (x+1)

        # Second derivatives (finite difference approximation)
        d2_dx2 = (Z_right - 2 * Z_center + Z_left) / self.dx2
        d2_dy2 = (Z_down - 2 * Z_center + Z_up) / self.dy2

        # Laplacian is sum of second derivatives
        laplacian = d2_dx2 + d2_dy2

        return laplacian

    def forward(self, steps=1, return_trajectory=False, training_mode=False):
        """
        Perform forward pass (multiple time steps)

        Args:
            steps: Number of simulation steps
            return_trajectory: If True, return full trajectory; if False, return final state
            training_mode: If True, use gradient-enabled computation
        """
        if training_mode:
            # For training, start with initial state that can track gradients
            U_current = self.U_init.clone().requires_grad_(True)
            V_current = self.V_init.clone().requires_grad_(True)

            if return_trajectory:
                trajectory_U = [U_current]
                trajectory_V = [V_current]

            for _ in range(steps):
                U_current, V_current = self.step_with_gradients(U_current, V_current)
                if return_trajectory:
                    trajectory_U.append(U_current)
                    trajectory_V.append(V_current)

            # Update buffers for visualization (detached)
            self.U.data.copy_(U_current.detach())
            self.V.data.copy_(V_current.detach())

            if return_trajectory:
                return torch.stack(trajectory_U), torch.stack(trajectory_V)
            else:
                return U_current, V_current
        else:
            # For inference, use buffer-based computation
            if return_trajectory:
                trajectory_U = []
                trajectory_V = []

            for _ in range(steps):
                U_new, V_new = self.step()
                if return_trajectory:
                    trajectory_U.append(U_new.clone())
                    trajectory_V.append(V_new.clone())

            if return_trajectory:
                return torch.stack(trajectory_U), torch.stack(trajectory_V)
            else:
                return self.U, self.V

    def step_with_gradients(self, U_current, V_current):
        """Perform one time step with gradient tracking for training"""
        # Clamp values to prevent numerical instability
        U_current = torch.clamp(U_current, 0.0, 2.0)
        V_current = torch.clamp(V_current, 0.0, 2.0)

        # Calculate Laplacians
        Lu = self.laplacian(U_current)
        Lv = self.laplacian(V_current)

        # Clamp Laplacians to prevent extreme values
        Lu = torch.clamp(Lu, -10.0, 10.0)
        Lv = torch.clamp(Lv, -10.0, 10.0)

        # Gray-Scott equations using the trainable parameters
        uvv = U_current * V_current * V_current
        du = self.Du * Lu - uvv + self.f * (1 - U_current)
        dv = self.Dv * Lv + uvv - (self.f + self.k) * V_current

        # Clamp derivatives to prevent instability
        du = torch.clamp(du, -1.0, 1.0)
        dv = torch.clamp(dv, -1.0, 1.0)

        # Update states
        U_new = U_current + du * self.dt
        V_new = V_current + dv * self.dt

        # Clamp final values
        U_new = torch.clamp(U_new, 0.0, 2.0)
        V_new = torch.clamp(V_new, 0.0, 2.0)

        return U_new, V_new

    def step(self):
        """Perform one time step of the Gray-Scott equations for inference"""
        # Calculate Laplacians
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)

        # Gray-Scott equations using the current (possibly optimized) parameters
        uvv = self.U * self.V * self.V
        du = self.Du * Lu - uvv + self.f * (1 - self.U)
        dv = self.Dv * Lv + uvv - (self.f + self.k) * self.V

        # Create NEW tensors (not in-place)
        U_new = self.U + du * self.dt
        V_new = self.V + dv * self.dt

        # Update the buffers - detach to prevent accumulating gradients over multiple steps during inference
        self.U.data.copy_(U_new.detach())
        self.V.data.copy_(V_new.detach())

        # Return the new states for the caller (detached)
        return self.U, self.V

    def reset(self):
        """Reset the simulation to initial conditions and parameters to initial values"""
        # Reset parameters to initial values
        self.log_Du.data = torch.log(torch.tensor(self.initial_Du, dtype=torch.float32, device=self.device))
        self.log_Dv.data = torch.log(torch.tensor(self.initial_Dv, dtype=torch.float32, device=self.device))
        self.f.data = torch.tensor(self.initial_f, dtype=torch.float32, device=self.device)
        self.k.data = torch.tensor(self.initial_k, dtype=torch.float32, device=self.device)

        # Reset state (U and V buffers)
        self.reset_state()

    def get_numpy_arrays(self):
        """Get U and V as numpy arrays for visualization"""
        return self.U[0, 0].cpu().detach().numpy(), self.V[0, 0].cpu().detach().numpy()

    def get_coordinates(self):
        """Get physical coordinate arrays"""
        x = np.linspace(0, self.domain_width, self.width)
        y = np.linspace(0, self.domain_height, self.height)
        return np.meshgrid(x, y)

    def set_parameters(self, Du=None, Dv=None, f=None, k=None):
        """Update model parameters (for direct setting, not for optimization)"""
        if Du is not None:
            self.log_Du.data = torch.log(torch.tensor(Du, dtype=torch.float32, device=self.device))
        if Dv is not None:
            self.log_Dv.data = torch.log(torch.tensor(Dv, dtype=torch.float32, device=self.device))
        if f is not None:
            self.f.data = torch.tensor(f, dtype=torch.float32, device=self.device)
        if k is not None:
            self.k.data = torch.tensor(k, dtype=torch.float32, device=self.device)

    def get_stability_info(self):
        """Get stability information for the current parameters"""
        Du_val = self.Du.item()
        Dv_val = self.Dv.item()
        max_D = max(Du_val, Dv_val)
        min_dx2 = min(self.dx2, self.dy2)
        stability_limit = min_dx2 / (4 * max_D)

        return {
            'Du': Du_val,
            'Dv': Dv_val,
            'dx': self.dx,
            'dy': self.dy,
            'dt': self.dt,
            'stability_limit': stability_limit,
            'is_stable': self.dt < stability_limit,
            'recommended_dt': stability_limit * 0.8
        }

    def get_trainable_parameters(self):
        """Get dictionary of current trainable parameter values"""
        return {
            'Du': self.Du.item(),
            'Dv': self.Dv.item(),
            'f': self.f.item(),  # Include f and k for completeness
            'k': self.k.item(),
            'log_Du': self.log_Du.item(),
            'log_Dv': self.log_Dv.item(),
        }



class SimpleNeuralModel(nn.Module):
    """
    Simple neural network that learns the mapping (x,y,t) -> (u,v).

    This is a minimal feedforward network that directly approximates
    the Gray-Scott solution without simulating the PDE.
    Compatible with both plotting_scripts (network) and your PINN (net) naming conventions.
    """

    def __init__(self, hidden_size=128, num_layers=4, use_plotting_scripts_format=False):
        """
        Initialize the neural network.

        Args:
            hidden_size: Number of neurons in hidden layers
            num_layers: Number of hidden layers
            use_plotting_scripts_format: If True, use 'network' naming, else use 'net' naming
        """
        super(SimpleNeuralModel, self).__init__()

        # Input: (x, y, t) -> 3 dimensions
        # Output: (u, v) -> 2 dimensions

        layers = []

        # Input layer with larger initial layer for more capacity
        layers.append(nn.Linear(3, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers with residual connections
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, 2))

        # Use different naming conventions based on the format needed
        if use_plotting_scripts_format:
            self.network = nn.Sequential(*layers)
        else:
            self.net = nn.Sequential(*layers)

        # Initialize weights properly
        self._initialize_weights()

        # Store normalization parameters (will be set during training)
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.t_mean = 0.0
        self.t_std = 1.0

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform for ReLU activations
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y, t):
        """
        Forward pass of the neural network.

        Args:
            x: x coordinates (tensor)
            y: y coordinates (tensor)
            t: time coordinates (tensor)

        Returns:
            (u, v): Predicted concentrations
        """
        # Normalize inputs
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        t_norm = (t - self.t_mean) / self.t_std

        # Stack inputs
        inputs = torch.stack([x_norm, y_norm, t_norm], dim=-1)

        # Forward through network (handle both naming conventions)
        if hasattr(self, 'network'):
            outputs = self.network(inputs)
        else:
            outputs = self.net(inputs)

        # Split outputs into u and v
        u = outputs[..., 0]
        v = outputs[..., 1]


        return u, v

    def set_normalization(self, x_coords, y_coords, t_coords):
        """Set normalization parameters based on training data."""
        self.x_mean = torch.mean(x_coords).item()
        self.x_std = torch.std(x_coords).item() + 1e-8
        self.y_mean = torch.mean(y_coords).item()
        self.y_std = torch.std(y_coords).item() + 1e-8
        self.t_mean = torch.mean(t_coords).item()
        self.t_std = torch.std(t_coords).item() + 1e-8
