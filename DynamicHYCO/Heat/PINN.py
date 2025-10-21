import torch
import torch.nn as nn
from data.dataloaders import DataLoader_Scalar
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
import numpy as np

from models.generate_data import heat

class PINN2D(nn.Module):
    def __init__(self, layers, parameters, kappa):
        """
        layers: list specifying the number of neurons in each layer
        a, b: advection speeds in x and y directions
        D: diffusion coefficient
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(PINN2D, self).__init__()
        self.params = nn.Parameter(parameters)
        self._kappa = kappa
        
        # Build a fully connected neural network
        self.linears = nn.ModuleList().to(self.device)
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        
        # Xavier initialization
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        # Concatenate inputs
        out = torch.cat((x, y, t), dim=1)
        out = torch.tanh(self.linears[0](out))
        # Pass through network
        for i in range(1, len(self.linears) - 1):
            out = out + torch.tanh(self.linears[i](out))
        out = self.linears[-1](out)
        return out
    
    def kappa(self, x, y):
        return self._kappa(x, y, self.params)
    
    def eta(self, x, y):
        return self._eta(x, y, self.params)
    

    def pde_residual(self, x, y, t):
        """
        Physics loss for the 2D advection-diffusion PDE:
        u_t + a*u_x + b*u_y - D*(u_xx + u_yy) = 0
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, y, t)
        kappa = self.kappa(x, y)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        uk_xx = torch.autograd.grad(kappa*u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        uk_yy = torch.autograd.grad(kappa*u_y, y, torch.ones_like(u_y), create_graph=True)[0]
       
        
        return u_t - uk_xx - uk_yy
    
    def initial_condition_loss(self, x, y):
        # Sample points in the spatial domain
        t = torch.zeros_like(x)
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u_0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))

        # Model prediction at t=0
        u_pred = self.forward(x, y, t)

        return u_pred - u_0
    
    def boundary_condition_loss(self, x, y, t, L):
        # Model prediction on the boundary
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u_pred = self.forward(x, y, t)
        
        # Identify top/bottom boundary where y = ±L/2
        # Identify left/right boundary where x = ±L/2
        eps = 1e-7
        is_top_bottom = (torch.abs(torch.abs(y) - (L/2)) < eps)
        is_left_right = (torch.abs(torch.abs(x) - (L/2)) < eps)
        
        # Dirichlet zero boundary condition
        u_bc = torch.zeros_like(u_pred)
        u_bc[is_top_bottom] = 0
        u_bc[is_left_right] = 0
        # Compute the boundary condition error

        return u_pred - u_bc

def train_pinn(model, optimizer, L, T, x_train, y_train, t_train, u_train, num_epochs=10000, flag='full'):
    """
    Example training loop using random collocation points.
    For a real application, supply boundary/initial condition
    losses and domain collocation points.
    """
    alpha_real = np.array([3, 2.5, 1, -2, 1, -2, 1.0, 1.0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-3, 3, 50).to(device)
    y = torch.linspace(-3, 3, 50).to(device)
    t = torch.linspace(0, T, 100).to(device)[1:]
    grid_t, grid_x, grid_y = torch.meshgrid(t, x, y, indexing='ij')


    # High definition solution
    data_high = np.load(folder + f'/phys_{flag}.npy')
    data_high = torch.from_numpy(data_high).float().to(device)
    mse = torch.zeros(num_epochs).to(device)
    p_norm = torch.zeros(num_epochs).to(device)
    l2_norm = torch.zeros(num_epochs).to(device)
    

    for epoch in range(num_epochs):
        # Sample points for initial condition
        N = 800*100
        x0 = torch.rand(N, 1).to(device)*L - L/2
        y0 = torch.rand(N, 1).to(device)*L - L/2
        # Sample points for residual
        x = torch.rand(N, 1).to(device)*L - L/2
        y = torch.rand(N, 1).to(device)*L - L/2
        t = torch.rand(N, 1).to(device)*T
        # Sample points for boundary condition
        x_bc_1 = torch.rand(N//4, 1).to(device)*L - L/2
        y_bc_1 = torch.full_like(x_bc_1, -L/2).to(device)
        x_bc_2 = torch.rand(N//4, 1).to(device)*L - L/2
        y_bc_2 = torch.full_like(x_bc_2, L/2).to(device)
        x_bc_3 = torch.full_like(x_bc_1, -L/2).to(device)
        y_bc_3 = torch.rand(N//4, 1).to(device)*L - L/2
        x_bc_4 = torch.full_like(x_bc_1, L/2).to(device)
        y_bc_4 = torch.rand(N//4, 1).to(device)*L - L/2
        x_bc = torch.cat((x_bc_1, x_bc_2, x_bc_3, x_bc_4), dim=0)
        y_bc = torch.cat((y_bc_1, y_bc_2, y_bc_3, y_bc_4), dim=0)
        n_bc = x_bc.shape[0]
        t_bc = torch.rand(n_bc, 1).to(device)*T

        
        # IC loss
        ic_loss = model.initial_condition_loss(x0, y0)
        # PDE loss
        res = model.pde_residual(x, y, t)
        # BC loss
        bc_loss = model.boundary_condition_loss(x_bc, y_bc, t_bc, L)

        # Compute loss
        loss_phys = torch.mean(res**2) + torch.mean(ic_loss**2) + torch.mean(bc_loss**2) 

        u_pred = model(x_train, y_train, t_train)
        loss_data = 10*torch.mean((u_pred - u_train)**2)
       
        model.params.requires_grad = True
        loss = loss_data + loss_phys

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            # Calculate the mse on the high definition grid and solution
            data_interp = F.interpolate(data_high.unsqueeze(1), size=(grid_x.shape[1], grid_x.shape[2]), mode='bilinear', align_corners=True)
            data_interp = data_interp.squeeze(1)  # shape (99, 100, 100)
            u_pred = model(grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_t.reshape(-1,1))
            # reshape u_pred to be (99, 10000)
            u_pred = u_pred.reshape(data_interp.shape[0], -1)
            # make sure it is reshaped in terms of times
            l2 = torch.linalg.norm(torch.linalg.norm(u_pred - data_interp.reshape(data_interp.shape[0], -1), ord=2, dim=1)) / torch.linalg.norm(torch.linalg.norm(data_interp.reshape(data_interp.shape[0], -1), ord=2, dim=1))
            l2_norm[epoch] = l2.item()
            mse[epoch] = torch.mean((u_pred - data_interp.reshape(data_interp.shape[0], -1))**2).item()
            params = model.params.cpu().detach().numpy()
            ampl = params[0:2] 
            centers = params[2:6]
            p_norm[epoch] = np.linalg.norm(ampl - alpha_real[0:2]) / np.linalg.norm(alpha_real[0:2]) + np.linalg.norm(centers - alpha_real[2:6]) / np.linalg.norm(alpha_real[2:6])

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, loss_data: {loss_data.item()}, loss_phys: {loss_phys.item()}, params: {model.params.cpu().detach().numpy()}")
    return mse.detach().cpu().numpy(), p_norm.detach().cpu().numpy(), l2_norm.detach().cpu().numpy()
import matplotlib.pyplot as plt
import numpy as np

def visualize_pinn_solution(model, L, t_value=0.5, nx=50, ny=50):
    # Create a mesh for x and y
    x_vals = np.linspace(0, 1, nx)*L - L/2
    y_vals = np.linspace(0, 1, ny)*L - L/2
    X, Y = np.meshgrid(x_vals, y_vals)

    # Convert to torch tensors
    x_torch = torch.tensor(X.ravel(), dtype=torch.float32).unsqueeze(1)
    y_torch = torch.tensor(Y.ravel(), dtype=torch.float32).unsqueeze(1)
    t_torch = torch.full_like(x_torch, t_value)
    model = model.to(torch.device("cpu"))
    # Predict
    with torch.no_grad():
        u_pred = model(x_torch, y_torch, t_torch)

    U = u_pred.cpu().numpy().reshape(X.shape)

    # Plot a contour of the solution
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, U, levels=100, cmap='viridis', vmin = -1, vmax = 1)
    plt.colorbar(cp)
    plt.title(f"PINN solution at t={t_value}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"pinn_solution{t_value}.png")
    plt.close()

if __name__ == "__main__":
    for flag in ['half', 'quarter']:
        
        print("---------------------------------")
        print("--------Generating data----------")
        print("---------------------------------")
        # Generate data
        if flag == 'full':
            heat(flag="full", L = 6, folder = 'data/heat/table')
        elif flag == 'half':
            heat(flag="half", L = 6, folder = 'data/heat/table')
        elif flag == 'quarter':
            heat(flag="quarter", L = 6, folder = 'data/heat/table')
        else:
            raise ValueError("Invalid flag value. Choose from ['full', 'half', 'quarter']")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = "data/heat/table"
        data = DataLoader_Scalar(device, folder, flag)
        
        x_train, y_train, u_train = data.u[:,1:,0], data.u[:,1:,1], data.u[:,1:,2]
        t_train = torch.zeros_like(x_train.T).to(device)
        for i in range(t_train.shape[0]):
            t_train[:,i] = data.t[1:]

        def kappa(x, y, param):
            num_gaussians = int(param.shape[0] / 4)
            gaussian_map =0.1*torch.ones_like(x).to(device)
            for i in range(num_gaussians):
                gaussian_map += param[i] * torch.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
            return gaussian_map
        

        net_layers = [3, 256, 256, 256, 1]  # (x, y, t) -> u
        parameters = torch.tensor([1, 1, 1.4, -1.8, -1.3, -1.1, 1.0, 1.0]).float().to(device)
        L = 6
        T = 1
        num_epochs = 3000
        model = PINN2D(net_layers, parameters, kappa=kappa).to(device)
        # print number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in PINN: {num_params}")
        optimizer = torch.optim.Adam([
            {'params': model.linears.parameters(), 'lr': 1e-4},   # For the network weights and biases
            {'params': [model.params], 'lr': 5e-4}                  # For the physical model parameters
        ])
        import time 
        t1 = time.time()
        mse, p_norm, l2_norm = train_pinn(model, optimizer, L, T,
                    x_train.reshape(-1,1), y_train.reshape(-1,1),
                    t_train.reshape(-1,1), u_train.reshape(-1,1), num_epochs=num_epochs, flag=flag)
        t2 = time.time()
        print(f"Training time: {t2 - t1:.2f} seconds")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        parameters = np.concatenate((parameters[0:3].cpu().numpy(), np.array([1.0]), parameters[3:6].cpu().numpy(), np.array([1.0])))
        params = pd.DataFrame(parameters, columns=["params"])
        errors = pd.DataFrame({"mse": mse, "p_norm": p_norm, "l2_norm": l2_norm})
        params.to_csv(f'parameters/heat/table/param_pinn_{timestamp}_{flag}.csv', index=False)
        errors.to_csv(f'parameters/heat/table/error_pinn_{timestamp}_{flag}.csv', index=False)
        # save the timings
        timings = pd.DataFrame(np.stack([np.array([t2 - t1, t2 - t1])]).T, columns=["timings"])
        timings.to_csv(f'parameters/heat/table/timings_{timestamp}_{flag}.csv', index=False)

