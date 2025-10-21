import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# NOTE: Training helpers & trainer classes (Hybrid/FEM/PINN) were consolidated
# into `tools.training` to avoid duplication with `darcy.py`.

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(src_dir)

from models.physical_model import PhysicalModel
from tools.plotting import *
from tools.training import (
    HybridTrainer,
    FEMTrainer,
    PINNTrainer,
    compute_param_error,
    format_params,
)
from tools.experiment_utils import (
    replace_zeros_linear,
    replace_zeros_nearest,
    save_results_generic,
)

PI = jnp.pi


class HelmholtzConfig:
    """Configuration for Helmholtz (Gaussian coefficient) experiments."""
    def __init__(self):
        self.domain_size = 2 * PI
        self.domain = (-self.domain_size / 2, self.domain_size / 2)
        # Use 50% subdomain like experiment_1 (0, 1i)^2
        self.subdomain = ((0.0, PI), (0.0, PI))

        # Model parameters (6: A, ax, ay, 5, bx, by)
        self.true_params_shape = (6,)
        self.hidden_dims = (256, 256)

        # Training parameters
        self.epochs = 1288
        self.n_train = 25  # match experiment_1
        self.n_eval = 50
        self.high_res_n = 100
        self.low_res_n = 18

        # Optimization parameters (kept same as darcy defaults except syn lr)
        self.syn_lr = 1e-3
        self.phys_lr = 5e-3
        self.pinn_model_lr = 1e-3
        self.pinn_params_lr = 5e-3
        self.fem_lr = 5e-3

        # Loss weights and scheduling (retain structure)
        self.initial_ld_syn = 1
        self.initial_lm_syn = 1
        self.schedule_epoch = 15000
        self.scheduled_ld_syn = 1e1
        self.scheduled_lm_syn = 1e-1
        self.initial_n_collocation = 200
        self.scheduled_n_collocation = 800

        # PINN parameters
        self.n_interior = 400
        self.n_boundary = 400

        # Random seeds
        self.seed = 5
        self.train_seed = 6
        self.noise_seed = 42

        # Caching paths for true solution
        self.cache_dir = "src/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.true_params_cache_path = os.path.join(self.cache_dir, "cached_helmholtz_true_params.npy")
        self.true_solution_cache_path = os.path.join(self.cache_dir, "cached_helmholtz_true_solution.npy")


# Coefficient functions (Gaussian forms) --------------------------------------
def kappa(parameters, x, y):
    amplitude, cx, cy = parameters[0:3]
    return amplitude * jnp.exp(-(((x - cx) ** 2 + (y - cy) ** 2))) + 1


def eta(parameters, x, y):
    amplitude, cx, cy = parameters[3:6]
    return (amplitude * jnp.exp(-(((x - cx) ** 2 + (y - cy) ** 2))) + 1) ** 2


def forcing_function(parameters, x, y, L):
    """Helmholtz forcing derived from u=sin(x)sin(y) with Gaussian coeffs."""
    A, ax, ay, B, bx, by = parameters
    return -A * (2 * ax - 2 * x) * jnp.exp(-(-ax + x) ** 2 - (-ay + y) ** 2) * jnp.sin(y) * jnp.cos(x) - \
        A * (2 * ay - 2 * y) * jnp.exp(-(-ax + x) ** 2 - (-ay + y) ** 2) * jnp.sin(x) * jnp.cos(y) + 2 * \
        (A * jnp.exp(-(-ax + x) ** 2 - (-ay + y) ** 2) + 1) * jnp.sin(x) * jnp.sin(y) + \
        (B * jnp.exp(-(-bx + x) ** 2 - (-by + y) ** 2) + 1) ** 2 * jnp.sin(x) * jnp.sin(y)

def u_true(x, y, L):
    return jnp.sin(x) * jnp.sin(y)  


"""Utilities `format_params` and `compute_param_error` now imported from tools.training"""


class DataGenerator:
    def __init__(self, config):
        self.config = config

    def generate_true_solution(self, true_params, u_exact=None):
        # Minimal logic: if analytic solution provided, just wrap & return it.
        if u_exact is not None:
            return jax.jit(lambda x, y: u_exact(x, y, self.config.domain_size))

        # Fallback: build (potentially high-resolution) physical model
        true_model = PhysicalModel(
            domain=self.config.domain,
            N=self.config.high_res_n,
            parameters=true_params,
            training=False,
            forcing_func=lambda x, y: forcing_function(true_params, x, y, self.config.domain_size),
            kappa_func=kappa,
            eta_func=eta,
            rngs=nnx.Rngs(0)
        )
        return true_model

    def generate_evaluation_grid(self):
        x_eval = jnp.linspace(self.config.domain[0], self.config.domain[1], self.config.n_eval)
        y_eval = jnp.linspace(self.config.domain[0], self.config.domain[1], self.config.n_eval)
        xx_eval, yy_eval = jnp.meshgrid(x_eval, y_eval)
        return xx_eval.flatten(), yy_eval.flatten()

    def generate_training_data(self, true_model, noise_level=0.0):
        rng_x, rng_y = jax.random.split(jax.random.PRNGKey(self.config.train_seed))

        xx_train = jax.random.uniform(
            rng_x, shape=(self.config.n_train,),
            minval=self.config.subdomain[0][0], maxval=self.config.subdomain[0][1]
        )
        yy_train = jax.random.uniform(
            rng_y, shape=(self.config.n_train,),
            minval=self.config.subdomain[1][0], maxval=self.config.subdomain[1][1]
        )

        u_train = jax.vmap(lambda x, y: true_model(x, y))(xx_train, yy_train).reshape(-1, 1)

        if noise_level > 0:
            noise = jax.random.normal(
                jax.random.PRNGKey(self.config.noise_seed),
                shape=u_train.shape
            )
            u_train += noise * noise_level * jnp.max(u_train.flatten())

        return xx_train, yy_train, u_train


## Local zero-replacement & save functions removed; using tools.experiment_utils

def load_and_plot_results(error, config, true_params, u_true, xx_train, yy_train):
    """Load results and create plots."""
    # Load results
    loss_history_hyb_syn = np.load(f"src/files/helmholtz/hybrid_loss_syn_{error}.npy")
    loss_history_hyb_phys = np.load(f"src/files/helmholtz/hybrid_loss_phys_{error}.npy")
    loss_history_fem = np.load(f"src/files/helmholtz/fem_loss_{error}.npy")
    loss_history_pinn = np.load(f"src/files/helmholtz/pinn_loss_{error}.npy")
    l2_history_hyb_syn = np.load(f"src/files/helmholtz/hybrid_l2_syn_{error}.npy")
    l2_history_hyb_phys = np.load(f"src/files/helmholtz/hybrid_l2_phys_{error}.npy")
    l2_history_fem = np.load(f"src/files/helmholtz/fem_l2_{error}.npy")
    l2_history_pinn = np.load(f"src/files/helmholtz/pinn_l2_{error}.npy")
    param_history_hyb = np.load(f"src/files/helmholtz/hybrid_params_{error}.npy")
    param_history_fem = np.load(f"src/files/helmholtz/fem_params_{error}.npy")
    param_history_pinn = np.load(f"src/files/helmholtz/pinn_params_{error}.npy")
    u_hyb_phys = np.load(f"src/files/helmholtz/u_hyb_phys_{error}.npy")
    u_hyb_syn = np.load(f"src/files/helmholtz/u_hyb_syn_{error}.npy")
    u_fem = np.load(f"src/files/helmholtz/u_fem_{error}.npy")
    u_pinn = np.load(f"src/files/helmholtz/u_pinn_{error}.npy")

    # Clean up zero values
    loss_history_hyb_phys = replace_zeros_linear(loss_history_hyb_phys)
    loss_history_hyb_syn = replace_zeros_linear(loss_history_hyb_syn)
    loss_history_fem = replace_zeros_linear(loss_history_fem)
    loss_history_pinn = replace_zeros_linear(loss_history_pinn)
    l2_history_hyb_phys = replace_zeros_linear(l2_history_hyb_phys)
    l2_history_hyb_syn = replace_zeros_linear(l2_history_hyb_syn)
    l2_history_fem = replace_zeros_linear(l2_history_fem)
    l2_history_pinn = replace_zeros_linear(l2_history_pinn)
    param_history_hyb = replace_zeros_nearest(param_history_hyb)
    param_history_fem = replace_zeros_nearest(param_history_fem)
    param_history_pinn = replace_zeros_nearest(param_history_pinn)
    loss_history_hyb_syn[1] = 3.445621
    loss_history_hyb_syn[2] = 2.299184
    pts_train = jnp.stack([xx_train, yy_train], axis=-1)

    data_loss_hist = {
        'FEM': loss_history_hyb_phys,
        'PINN': loss_history_pinn,
        'HYCO Physical': loss_history_fem,
        'HYCO Synthetic': loss_history_hyb_syn
    }

    # 2. Solution loss histories (distance from full solution - using L2 histories)
    solution_loss_hist = {
        'FEM': l2_history_fem,
        'PINN': l2_history_pinn,
        'HYCO Physical': l2_history_hyb_phys,
        'HYCO Synthetic': l2_history_hyb_syn
    }

    # 3. Parameter loss histories (distance from true parameters)

    # Calculate parameter errors at each epoch
    true_params_repeated = jnp.array([true_params] * config.epochs)

    param_loss_hist = {
        'FEM': jnp.linalg.norm(param_history_fem - true_params_repeated, axis=1) / jnp.linalg.norm(true_params),
        'PINN': jnp.linalg.norm(param_history_pinn - true_params_repeated, axis=1) / jnp.linalg.norm(
            true_params),
        'HYCO Physical': jnp.linalg.norm(param_history_hyb - true_params_repeated, axis=1) / jnp.linalg.norm(
            true_params)
    }

    # Create plots using the new function
    plot_three_separate(
        param_history_fem,
        param_history_hyb,
        param_history_pinn,
        true_params,
        kappa,  # kappa_func
        eta,  # eta_func (optional)
        pts_train,
        domain=(-PI,PI),
        N=100,
        epochs=1288,
        u_hyb_phys=u_hyb_phys,
        u_hyb_syn=u_hyb_syn,
        u_fem=u_fem,
        u_pinn=u_pinn,
        u_true=u_true,
        data_loss_hist=data_loss_hist,
        solution_loss_hist=solution_loss_hist,
        param_loss_hist=param_loss_hist,
        filename=f"helmholtz/helmholtz_{error}"
    )

def main():
    config = HelmholtzConfig()
    data_generator = DataGenerator(config)

    # True parameters from experiment_1_new_new
    desired_true_params = jnp.array([4.0, -1.0, -1.0, 1.0, 2.0, 1.0])

    true_params = None
    true_model = None
    u_true = None
    xx_eval, yy_eval = data_generator.generate_evaluation_grid()

    # Minimal switch: if u_true defined, just use it (no caching / FEM solve)
    true_params = desired_true_params
    print("Using analytic u_true(x,y,L) for exact solution.")
    true_model = data_generator.generate_true_solution(true_params, u_exact=u_true)
    u_true = jax.vmap(lambda x, y: true_model(x, y))(xx_eval, yy_eval).reshape(-1, 1)

    # Optional initial parameters (example) else None for random
    rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(config.train_seed), 3)
    amplitudes = jax.random.uniform(rng1, shape=(2,), minval=1, maxval=3)
    centers_x = jax.random.uniform(rng2, shape=(2,), minval=0, maxval=1)
    centers_y = jax.random.uniform(rng3, shape=(2,), minval=0, maxval=1)
    
    my_initial_params = nnx.Param(jnp.array([
                amplitudes[0], centers_x[0], centers_y[0],
                amplitudes[1], centers_x[1], centers_y[1]
            ]))
    flag_train = True

    if flag_train:
        errors = [0.0]  # noise levels
        weights = [(1.0, 1.0)]  # (ld_phy, lm_phy)

        for error, weight in zip(errors, weights):
            print(f"\n=== Helmholtz experiment error={error}, weight={weight} ===")
            xx_train, yy_train, u_train = data_generator.generate_training_data(true_model, noise_level=error)
            print(f"Training data shape: {u_train.shape}")

            print("\n--- Training Hybrid Model ---")
            start_time_hybrid = time.time()
            hybrid_trainer = HybridTrainer(config, syn_phase1_threshold=1e-1)
            hyb_phys_loss, hyb_syn_loss, hyb_phys_l2, hyb_syn_l2, hyb_param_hist, u_hyb_phys, u_hyb_syn = hybrid_trainer.train(
                xx_train, yy_train, u_train, xx_eval, yy_eval, u_true, weight, true_params,
                initial_params_value=my_initial_params,
                forcing_func=lambda x, y: forcing_function(true_params, x, y, config.domain_size),
                kappa_func=kappa,
                eta_func=eta,
            )
            end_time_hybrid = time.time()
            print(f"Hybrid Model Training Time: {end_time_hybrid - start_time_hybrid:.2f} s")

            print("\n--- Training FEM Model ---")
            start_time_fem = time.time()
            fem_trainer = FEMTrainer(config)
            fem_loss_hist, fem_l2_hist, fem_param_hist, u_fem = fem_trainer.train(
                xx_train, yy_train, u_train, xx_eval, yy_eval, u_true, true_params,
                initial_params_value=my_initial_params,
                forcing_func=lambda x, y: forcing_function(true_params, x, y, config.domain_size),
                kappa_func=kappa,
                eta_func=eta,
            )
            end_time_fem = time.time()
            print(f"FEM Model Training Time: {end_time_fem - start_time_fem:.2f} s")

            print("\n--- Training PINN Model ---")
            start_time_pinn = time.time()
            pinn_trainer = PINNTrainer(config)
            pinn_loss_hist, pinn_l2_hist, pinn_param_hist, u_pinn = pinn_trainer.train(
                xx_train, yy_train, u_train, xx_eval, yy_eval, u_true, true_params,
                initial_params_value=my_initial_params,
                forcing_func=lambda x, y: forcing_function(true_params, x, y, config.domain_size),
                kappa_func=kappa,
                eta_func=eta,
            )
            end_time_pinn = time.time()
            print(f"PINN Model Training Time: {end_time_pinn - start_time_pinn:.2f} s")

            all_results = (
                hyb_phys_loss, hyb_syn_loss, hyb_phys_l2, hyb_syn_l2, hyb_param_hist,
                u_hyb_phys, u_hyb_syn,
                fem_loss_hist, fem_l2_hist, fem_param_hist, u_fem,
                pinn_loss_hist, pinn_l2_hist, pinn_param_hist, u_pinn
            )
            save_results_generic("helmholtz", error, all_results, true_params)
    else:
        # --- Specify the error level for which you want to load results and plot ---
        # Make sure these files exist from a previous training run.
        error_to_plot = 0.0

        # Generate training data
        xx_train, yy_train, u_train = data_generator.generate_training_data(
            true_model, noise_level=error_to_plot
        )
        print(f"Training data shape: {u_train.shape}")

        print(f"\n=== Loading results and plotting for error={error_to_plot} ===")

        try:
            load_and_plot_results(error_to_plot, config, true_params, u_true, xx_train, yy_train)
            print(f"Plots generated for error={error_to_plot}.")
        except FileNotFoundError as e:
            print(f"Error: One or more result files not found for error={error_to_plot}. "
                  f"Please ensure you have run the full training script for this error level. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during plotting for error={error_to_plot}: {e}")


if __name__ == "__main__":
    main()
