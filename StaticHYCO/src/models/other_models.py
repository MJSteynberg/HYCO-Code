from flax import nnx
import jax
import jax.numpy as jnp
from typing import Callable, Sequence
import optax

class PINN(nnx.Module):
    def __init__(self, domain: tuple, model, parameters: jnp.ndarray,
                 forcing_func: Callable, kappa_func: Callable, eta_func: Callable, rngs: nnx.Rngs):
        self.domain = domain
        self.model = model
        self.parameters = parameters
        self.forcing_func = forcing_func
        self.kappa_func = kappa_func
        self.eta_func = eta_func
    
    def __call__(self, x: float, y: float) -> float:
        return self.model(x, y)
    
    def kappa(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.kappa_func(self.parameters, x, y)
    
    def eta(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.eta_func(self.parameters, x, y)
    
    def on_boundary(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        left   = jnp.isclose(x, self.domain[0])
        right  = jnp.isclose(x, self.domain[1])
        bottom = jnp.isclose(y, self.domain[0])
        top    = jnp.isclose(y, self.domain[1])
        return jnp.logical_or(jnp.logical_or(left, right),
                              jnp.logical_or(bottom, top))
    
    @nnx.jit
    def residual(self, x: float, y: float) -> float:
        # First, define functions to compute the derivatives
        def u_x(x, y):
            return nnx.grad(lambda x_in: self.model(x_in, y)[0], argnums=0)(x)
        
        def u_y(x, y):
            return nnx.grad(lambda y_in: self.model(x, y_in)[0], argnums=0)(y)
        
        # For div(kappa * grad(u)) we need derivatives of (kappa * u_x) and (kappa * u_y)
        def kappa_ux(x, y):
            return nnx.grad(lambda x_in: self.kappa(x_in, y) * u_x(x_in, y), argnums=0)(x)
        
        def kappa_uy(x, y):
            return nnx.grad(lambda y_in: self.kappa(x, y_in) * u_y(x, y_in), argnums=0)(y)
        
        # The residual is -div(kappa * grad(u)) + eta * u - f
        return -kappa_ux(x, y) - kappa_uy(x, y) + self.eta(x, y) * self.model(x, y)[0] - self.forcing_func(x, y)
    
    def create_collocation_points(self, n_in: int, n_b, rng) -> Sequence[jnp.ndarray]:  
        n_b = n_b//4
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        x_in = jax.random.uniform(rng1, (n_in, 1), minval=self.domain[0], maxval=self.domain[1])
        y_in = jax.random.uniform(rng2, (n_in, 1), minval=self.domain[0], maxval=self.domain[1])
        

        x_1 = jax.random.uniform(rng3, (n_b, 1), minval=self.domain[0], maxval=self.domain[1])
        y_11 = jnp.full((n_b, 1), self.domain[0])
        y_12 = jnp.full((n_b, 1), self.domain[1])  

        y_2 = jax.random.uniform(rng4, (n_b, 1), minval=self.domain[0], maxval=self.domain[1])
        x_21 = jnp.full((n_b, 1), self.domain[0])
        x_22 = jnp.full((n_b, 1), self.domain[1])


        x_b = jnp.concatenate([x_1, x_1, x_21, x_22], axis=0)
        y_b = jnp.concatenate([y_11, y_12, y_2, y_2], axis=0)
        return x_in.reshape(-1), y_in.reshape(-1), x_b.reshape(-1), y_b.reshape(-1)
        

###########################################################################################
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt

def forward():
    # Define the domain
    domain = (0.0, 1.0)

    # Define the problem functions
    def forcing_func(x, y):
        # Simple forcing function: f(x,y) = 2π²sin(πx)sin(πy)
        # This corresponds to the exact solution: u(x,y) = sin(πx)sin(πy)
        return 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def kappa_func(params, x, y):
        # Constant diffusion coefficient
        return 1.0

    def eta_func(params, x, y):
        # No reaction term
        return 0.0

    def exact_solution(x, y):
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    # Initialize the model
    model = FeedForwardNet(
        hidden_dims=(64, 64, 64),
        activation=jnp.tanh,
        output_dim=1,
        rngs=nnx.Rngs(0)
    )

    # For forward problem, we don't need parameters
    parameters = nnx.Param(jnp.array([]))

    # Create PINN
    pinn = PINN(
        domain=domain,
        model=model,
        parameters=parameters,
        forcing_func=forcing_func,
        kappa_func=kappa_func,
        eta_func=eta_func,
        rngs=nnx.Rngs(0)
    )

    # Define vmapped helper functions for scalar calls
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy)[0])(xs, ys)

    def vmapped_residual(m, xs, ys):
        return jax.vmap(lambda xx, yy: m.residual(xx, yy))(xs, ys)

    # Training setup with multi-transform optimizer
    tx = optax.adam(learning_rate=1e-3)
    opt = nnx.Optimizer(pinn, tx)

    @nnx.jit
    def loss_fn(model, x_in, y_in, x_b, y_b):
        # Physics loss (PDE residual)
        residuals = vmapped_residual(model, x_in, y_in)
        physics_loss = jnp.mean(residuals**2)
        
        # Boundary loss (u = 0 on the boundary)
        boundary_values = vmapped_model(model, x_b, y_b)
        boundary_loss = jnp.mean(boundary_values**2)
        
        # Total loss
        total_loss = physics_loss + 100.0 * boundary_loss
        return total_loss

    @nnx.jit
    def train_step(model, optimizer, x_in, y_in, x_b, y_b):
        loss_value, grads = nnx.value_and_grad(loss_fn)(model, x_in, y_in, x_b, y_b)
        optimizer.update(grads)
        return loss_value

    # Training loop
    n_epochs = 3000
    n_in = 400
    n_b = 400
    key = jax.random.PRNGKey(42)

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        x_in, y_in, x_b, y_b = pinn.create_collocation_points(n_in, n_b, subkey)
        loss = train_step(pinn, opt, x_in, y_in, x_b, y_b)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4e}")

    # Plot results
    grid_size = 50
    x_grid = jnp.linspace(domain[0], domain[1], grid_size)
    y_grid = jnp.linspace(domain[0], domain[1], grid_size)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    # Use vmapped_model for predictions
    Z_pred = jax.vmap(lambda xs: jax.vmap(lambda ys: model(xs, ys)[0])(Y[:, 0]))(X[0, :])
    Z_exact = jax.vmap(lambda xs: jax.vmap(lambda ys: exact_solution(xs, ys))(Y[:, 0]))(X[0, :])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(X, Y, Z_pred, shading='auto')
    plt.colorbar()
    plt.title("PINN Solution")

    plt.subplot(1, 3, 2)
    plt.pcolormesh(X, Y, Z_exact, shading='auto')
    plt.colorbar()
    plt.title("Exact Solution")

    plt.subplot(1, 3, 3)
    plt.pcolormesh(X, Y, jnp.abs(Z_pred - Z_exact), shading='auto')
    plt.colorbar()
    plt.title("Absolute Error")

    plt.tight_layout()
    plt.savefig("forward_poisson_results.png")
    plt.show()

def inverse():
    # Define the domain
    domain = (0.0, 1.0)

    # Define the problem functions with unknown parameters
    def forcing_func(x, y):
        # Simple forcing function
        a = 1.5
        c = 0.8
        return (2*a*jnp.pi**2 + c) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def kappa_func(params, x, y):
        # Spatially varying diffusion coefficient: kappa(x,y) = a + b*x^2
        a, b = params[0], params[1]
        return a

    def eta_func(params, x, y):
        # Spatially varying reaction coefficient: eta(x,y) = c + d*y^2
        c, d = params[2], params[3]
        return c

    # Ground truth parameters
    true_params = jnp.array([1.5, 2.0, 0.8, 1.2])  # [a, b, c, d]

    # Generate synthetic data
    def generate_helmholtz_data(params, num_points=100):
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)
        
        x_data = jax.random.uniform(key1, (num_points,), minval=domain[0], maxval=domain[1])
        y_data = jax.random.uniform(key2, (num_points,), minval=domain[0], maxval=domain[1])
        
        # Simple reference solution
        u_exact = jnp.sin(jnp.pi * x_data) * jnp.sin(jnp.pi * y_data)
        
        
        return x_data, y_data, u_exact

    # Initialize the model
    model = FeedForwardNet(
        hidden_dims=(150, 150, 150),
        activation=jnp.sin,
        output_dim=1,
        rngs=nnx.Rngs(0)
    )

    # Initial guess for parameters
    init_params = nnx.Param(jnp.array([1.0, 1.0, 0.5, 0.5]))  # [a, b, c, d]

    # Generate synthetic data
    x_data, y_data, u_data = generate_helmholtz_data(true_params, num_points=200)

    # Create PINN with initial parameters
    pinn = PINN(
        domain=domain,
        model=model,
        parameters=init_params,
        forcing_func=forcing_func,
        kappa_func=kappa_func,
        eta_func=eta_func,
        rngs=nnx.Rngs(0)
    )

    # Create multi-transform optimizer for handling both NN and physical parameters
    tx = optax.multi_transform(
        {
            "model": optax.adam(1e-3),
            "parameters": optax.adam(1e-3),  # Slower for physical parameters
        },
        nnx.State({
            "model": "model", 
            "parameters": "parameters"
        })
    )
    opt = nnx.Optimizer(pinn, tx)

    # Define vmapped helper functions
    def vmapped_model(m, xs, ys):
        return jax.vmap(lambda xx, yy: m(xx, yy)[0])(xs, ys)

    def vmapped_residual(m, xs, ys):
        return jax.vmap(lambda xx, yy: m.residual(xx, yy))(xs, ys)

    # Training functions
    @nnx.jit
    def train_step_nn_only(model, optimizer, x_in, y_in, x_b, y_b):
        """Train only the neural network parameters (not physical params)"""
        def loss_fn(m):
            residuals = vmapped_residual(m, x_in, y_in)
            physics_loss = jnp.mean(residuals**2)
            
            boundary_values = vmapped_model(m, x_b, y_b)
            boundary_loss = jnp.mean(boundary_values**2)
            
            return physics_loss + 100.0 * boundary_loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        
        # Zero out gradients for physical parameters
        grads['parameters'] = jax.tree.map(lambda g: 0.0, grads['parameters'])
        optimizer.update(grads)
        
        return loss

    @nnx.jit
    def train_step_full(model, optimizer, x_data, y_data, u_data, x_in, y_in, x_b, y_b):
        """Train both network and physical parameters"""
        def loss_fn(m):
            # Physics loss (PDE residual)
            residuals = vmapped_residual(m, x_in, y_in)
            physics_loss = jnp.mean(residuals**2)
            
            # Boundary loss (u = 0 on the boundary)
            boundary_values = vmapped_model(m, x_b, y_b)
            boundary_loss = jnp.mean(boundary_values**2)
            
            # Data loss
            predictions = vmapped_model(m, x_data, y_data)
            data_loss = jnp.mean((predictions - u_data)**2)
            
            return physics_loss + 100.0 * boundary_loss + 1000.0 * data_loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        
        # Calculate physics loss separately for monitoring
        physics_loss = jnp.mean(vmapped_residual(model, x_in, y_in)**2)
        
        return loss, physics_loss

    # Training loop
    n_epochs = 5000
    n_in = 400
    n_b = 400
    key = jax.random.PRNGKey(42)

    param_history = [pinn.parameters.value]
    loss_history = []
    physics_threshold = 1e-1  # Threshold to switch from NN-only to full training

    # Initial phase: train only neural network
    print("Phase 1: Training neural network only...")
    for epoch in range(3000):  # Limited initial epochs
        key, subkey = jax.random.split(key)
        x_in, y_in, x_b, y_b = pinn.create_collocation_points(n_in, n_b, subkey)
        physics_loss = train_step_nn_only(pinn, opt, x_in, y_in, x_b, y_b)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Physics Loss: {physics_loss:.4e}")
        
        # Check if we can progress to full training
        if physics_loss < physics_threshold:
            print(f"Physics loss below threshold at epoch {epoch}, switching to full training")
            break

    # Second phase: train both network and parameters
    print("Phase 2: Training both neural network and physical parameters...")
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        x_in, y_in, x_b, y_b = pinn.create_collocation_points(n_in, n_b, subkey)
        loss, physics_loss = train_step_full(pinn, opt, x_data, y_data, u_data, x_in, y_in, x_b, y_b)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4e}, Physics Loss: {physics_loss:.4e}")
            print(f"Parameters: {pinn.parameters.value}")
        
        if epoch % 100 == 0:
            param_history.append(pinn.parameters.value)
            loss_history.append(loss)

    # Plot parameter convergence
    param_history = jnp.array(param_history)
    param_names = ['a (κ)', 'b (κ)', 'c (η)', 'd (η)']

    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(param_history[:, i], label=f'{param_names[i]} (learned)')
        plt.axhline(y=true_params[i], color='r', linestyle='--', label=f'{param_names[i]} (true)')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Parameter value')
        plt.legend()
        plt.title(f'{param_names[i]} convergence')

    plt.tight_layout()
    plt.savefig("inverse_helmholtz_parameters.png")
    plt.show()

    print(f"True parameters: {true_params}")
    print(f"Learned parameters: {pinn.parameters.value}")

    # Plot the resulting kappa and eta functions
    grid_size = 50
    x_grid = jnp.linspace(domain[0], domain[1], grid_size)
    y_grid = jnp.linspace(domain[0], domain[1], grid_size)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    kappa_true = jax.vmap(jax.vmap(lambda x, y: kappa_func(true_params, x, y)))(X, Y)
    kappa_pred = jax.vmap(jax.vmap(lambda x, y: kappa_func(pinn.parameters.value, x, y)))(X, Y)

    eta_true = jax.vmap(jax.vmap(lambda x, y: eta_func(true_params, x, y)))(X, Y)
    eta_pred = jax.vmap(jax.vmap(lambda x, y: eta_func(pinn.parameters.value, x, y)))(X, Y)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(X, Y, kappa_true, shading='auto')
    plt.colorbar()
    plt.title("True κ(x,y)")

    plt.subplot(2, 2, 2)
    plt.pcolormesh(X, Y, kappa_pred, shading='auto')
    plt.colorbar()
    plt.title("Predicted κ(x,y)")

    plt.subplot(2, 2, 3)
    plt.pcolormesh(X, Y, eta_true, shading='auto')
    plt.colorbar()
    plt.title("True η(x,y)")

    plt.subplot(2, 2, 4)
    plt.pcolormesh(X, Y, eta_pred, shading='auto')
    plt.colorbar()
    plt.title("Predicted η(x,y)")

    plt.tight_layout()
    plt.savefig("inverse_helmholtz_coefficients.png")
    plt.show()

if __name__ == "__main__":
    from models.synthetic_model import FeedForwardNet, ResNetSynthetic
    #forward()
    inverse()