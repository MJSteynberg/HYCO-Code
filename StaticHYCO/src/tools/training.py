"""Shared training utilities and trainer classes for PDE experiments.

This module factors out duplicated training logic from individual PDE scripts
(such as `darcy.py` and `helmholtz.py`). It provides:

- Utility functions: format_params, compute_param_error
- JIT'ed helpers: vmapped_model, train_step_data_only, train_step_hybrid
- Base class: ModelTrainer
- Trainer implementations: HybridTrainer, FEMTrainer, PINNTrainer

Design goals:
- Keep PDE-specific components (kappa, eta, forcing functions, DataGenerator,
  result saving / plotting) inside the PDE scripts.
- Allow customization of parameter initialization and phase-1 thresholds
  via constructor arguments or call-time parameters.
- Minimize changes required in existing scripts.

Usage example (Darcy):
    from tools.training import HybridTrainer
    hybrid_trainer = HybridTrainer(config)  # uses default phase-1 threshold

Usage example (Helmholtz with custom phase-1 threshold):
    hybrid_trainer = HybridTrainer(config, syn_phase1_threshold=1e-1,
                                   phys_param_init_fn=custom_init_fn)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import Callable, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_params(params: Any, precision: int = 4) -> str:
    """Return a nicely formatted string for a parameter vector (nnx.Param or array)."""
    if hasattr(params, 'value'):
        params = params.value
    return f"[{', '.join([f'{float(p):.{precision}f}' for p in jnp.ravel(params)])}]"


def compute_param_error(current_params: Any, true_params: jnp.ndarray) -> jnp.ndarray:
    """Compute relative L2 parameter error."""
    if hasattr(current_params, 'value'):
        current_params = current_params.value
    return jnp.linalg.norm(current_params - true_params) / jnp.linalg.norm(true_params)

# ---------------------------------------------------------------------------
# Low-level JIT helpers
# ---------------------------------------------------------------------------

@nnx.jit
def vmapped_model(model, xs, ys):
    """Evaluate a model over arrays xs, ys (shape: (N,)). Returns shape (N,1) or (N,)."""
    return jax.vmap(lambda xx, yy: model(xx, yy))(xs, ys)


@nnx.jit
def train_step_data_only(model, optimizer, x, y, u):
    """One optimization step using only data loss."""
    def loss_fn(m):
        u_pred = vmapped_model(m, x, y)
        return jnp.mean(optax.squared_error(u_pred, u))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def train_step_hybrid(model, model_other, optimizer, x, y, u,
                      x_collocation, y_collocation, lambda_data, lambda_match):
    """One optimization step with hybrid (data + matching) loss."""
    def loss_data(m):
        u_pred = vmapped_model(m, x, y)
        return jnp.mean(optax.squared_error(u_pred, u))

    def loss_match(m):
        u_pred = vmapped_model(m, x_collocation, y_collocation)
        u_pred_other = vmapped_model(model_other, x_collocation, y_collocation)
        return jnp.mean(optax.squared_error(u_pred, u_pred_other))

    def loss_fn(m):
        return lambda_data * loss_data(m) + lambda_match * loss_match(m)

    data_loss = loss_data(model)
    total_loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return total_loss, data_loss

# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def compute_l2_error(self, u_pred, u_true):
        return jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)

# ---------------------------------------------------------------------------
# Hybrid Trainer
# ---------------------------------------------------------------------------

class HybridTrainer(ModelTrainer):
    def __init__(self,
                 config,
                 phys_param_init_fn: Optional[Callable[[jax.random.PRNGKey], jnp.ndarray]] = None,
                 syn_activation: Callable = nnx.relu,
                 syn_phase1_threshold: Optional[float | Callable[[float, float], float]] = None):
        """Hybrid physics + synthetic model trainer.

        syn_phase1_threshold:
            - If None: uses 0.5 * max(ld_syn, lm_syn) (Darcy default behavior)
            - If float: uses this fixed value (Helmholtz used 1e-1)
            - If callable: called with (ld_syn, lm_syn) each epoch start of phase check.
        """
        super().__init__(config)
        self.phys_param_init_fn = phys_param_init_fn
        self.syn_activation = syn_activation
        self.syn_phase1_threshold = syn_phase1_threshold

    # -- internal helpers --------------------------------------------------
    def _init_physical_params(self, true_params_shape):
        if self.phys_param_init_fn is not None:
            return nnx.Param(self.phys_param_init_fn(jax.random.PRNGKey(self.config.train_seed)))
        # Fallback uniform init
        return nnx.Param(jax.random.uniform(
            jax.random.PRNGKey(self.config.train_seed),
            shape=true_params_shape, minval=-1, maxval=1
        ))

    def create_models(self, true_params, initial_params_value=None, forcing_func=None, kappa_func=None, eta_func=None):
        from models.physical_model import PhysicalModel  # local import to avoid cycles
        from models.synthetic_model import ResNetSynthetic

        synthetic_model = ResNetSynthetic(
            hidden_dims=self.config.hidden_dims,
            activation=self.syn_activation,
            output_dim=1,
            rngs=nnx.Rngs(0)
        )

        if initial_params_value is not None:
            init_params = nnx.Param(jnp.asarray(initial_params_value))
        else:
            init_params = self._init_physical_params(self.config.true_params_shape)

        physical_model = PhysicalModel(
            domain=self.config.domain,
            N=self.config.low_res_n,
            parameters=init_params,
            training=True,
            forcing_func=forcing_func,
            kappa_func=kappa_func,
            eta_func=eta_func,
            rngs=nnx.Rngs(0)
        )
        return synthetic_model, physical_model

    # -- public training ---------------------------------------------------
    def train(self, xx_train, yy_train, u_train, xx_eval, yy_eval, u_true,
              weight_params, true_params,
              initial_params_value=None,
              forcing_func=None, kappa_func=None, eta_func=None):
        """Run training loop.

        weight_params: tuple (ld_phy, lm_phy)
        Returns tuple of histories + final predictions similar to previous scripts.
        """
        synthetic_model, physical_model = self.create_models(
            true_params, initial_params_value, forcing_func, kappa_func, eta_func
        )

        syn_opt = nnx.Optimizer(synthetic_model, optax.adam(self.config.syn_lr))
        phys_opt = nnx.Optimizer(physical_model, optax.adam(self.config.phys_lr))

        loss_history_phys = np.zeros(self.config.epochs)
        loss_history_syn = np.zeros(self.config.epochs)
        l2_history_phys = np.zeros(self.config.epochs)
        l2_history_syn = np.zeros(self.config.epochs)
        param_history = np.zeros((self.config.epochs, self.config.true_params_shape[0]))

        rng = jax.random.PRNGKey(self.config.train_seed)
        n_collocation = self.config.initial_n_collocation
        loss_syn_data = 1.0
        ld_syn = self.config.initial_ld_syn
        lm_syn = self.config.initial_lm_syn
        ld_phy, lm_phy = weight_params

        print(f"True parameters:    {format_params(true_params)}")
        print(f"Initial parameters: {format_params(physical_model.parameters)}")
        print(f"Initial param error: {compute_param_error(physical_model.parameters, true_params):.4f}")
        print("-" * 80)

        for epoch in range(self.config.epochs):
            # scheduling (kept for compatibility; may never trigger if schedule_epoch large)
            if epoch == self.config.schedule_epoch:
                n_collocation = self.config.scheduled_n_collocation
                ld_syn = self.config.scheduled_ld_syn
                lm_syn = self.config.scheduled_lm_syn
                print(f"Epoch {epoch}: Adjust hyperparameters n_collocation={n_collocation}, ld_syn={ld_syn:.1e}, lm_syn={lm_syn:.1e}")

            # Determine phase-1 threshold
            if self.syn_phase1_threshold is None:
                threshold_value = 0.1
            elif callable(self.syn_phase1_threshold):
                threshold_value = float(self.syn_phase1_threshold(ld_syn, lm_syn))
            else:
                threshold_value = float(self.syn_phase1_threshold)

            # Phase 1: synthetic model data-only
            if loss_syn_data > threshold_value:
                loss_syn_data = train_step_data_only(synthetic_model, syn_opt, xx_train, yy_train, u_train)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch:4d} | Phase 1 - Synthetic loss: {loss_syn_data:.4e}")
                continue

            # Phase 2: hybrid optimization
            rng, rng1, rng2 = jax.random.split(rng, 3)
            x_collocation = jax.random.uniform(rng1, shape=(n_collocation,),
                                               minval=self.config.domain[0], maxval=self.config.domain[1])
            y_collocation = jax.random.uniform(rng2, shape=(n_collocation,),
                                               minval=self.config.domain[0], maxval=self.config.domain[1])

            loss_syn, loss_syn_data = train_step_hybrid(
                synthetic_model, physical_model, syn_opt,
                xx_train, yy_train, u_train,
                x_collocation, y_collocation, ld_syn, lm_syn
            )

            loss_phy, loss_phy_data = train_step_hybrid(
                physical_model, synthetic_model, phys_opt,
                xx_train, yy_train, u_train,
                x_collocation, y_collocation, ld_phy, lm_phy
            )

            u_pred_syn = vmapped_model(synthetic_model, xx_eval, yy_eval).reshape(-1, 1)
            u_pred_phys = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)

            l2_syn = self.compute_l2_error(u_pred_syn, u_true)
            l2_phys = self.compute_l2_error(u_pred_phys, u_true)
            param_error = compute_param_error(physical_model.parameters, true_params)

            loss_history_syn[epoch] = loss_syn_data
            loss_history_phys[epoch] = loss_phy_data
            l2_history_syn[epoch] = l2_syn
            l2_history_phys[epoch] = l2_phys
            param_history[epoch] = physical_model.parameters.value

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Phys L2: {loss_phy_data:.4e} | Syn L2: {loss_syn_data:.4e} | Param Err: {param_error:.4e}")
                print(f"           | Current params: {format_params(physical_model.parameters)}")

        # Final metrics
        u_pred_phys = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)
        u_pred_syn = vmapped_model(synthetic_model, xx_eval, yy_eval).reshape(-1, 1)
        final_param_error = compute_param_error(physical_model.parameters, true_params)

        print("-" * 80)
        print(f"Final parameters:   {format_params(physical_model.parameters)}")
        print(f"True parameters:    {format_params(true_params)}")
        print(f"Final param error:  {final_param_error:.4e}")

        return (loss_history_phys, loss_history_syn, l2_history_phys, l2_history_syn, param_history,
                u_pred_phys, u_pred_syn)

# ---------------------------------------------------------------------------
# FEM Trainer (data-only optimization of physical parameters)
# ---------------------------------------------------------------------------

class FEMTrainer(ModelTrainer):
    def __init__(self, config, phys_param_init_fn: Optional[Callable[[jax.random.PRNGKey], jnp.ndarray]] = None):
        super().__init__(config)
        self.phys_param_init_fn = phys_param_init_fn

    def _init_physical_params(self, true_params_shape):
        if self.phys_param_init_fn is not None:
            return nnx.Param(self.phys_param_init_fn(jax.random.PRNGKey(self.config.train_seed)))
        return nnx.Param(jax.random.uniform(
            jax.random.PRNGKey(self.config.train_seed),
            shape=true_params_shape, minval=-1, maxval=1
        ))

    def train(self, xx_train, yy_train, u_train, xx_eval, yy_eval, u_true, true_params,
              initial_params_value=None, forcing_func=None, kappa_func=None, eta_func=None):
        from models.physical_model import PhysicalModel

        if initial_params_value is not None:
            init_params = nnx.Param(jnp.asarray(initial_params_value))
        else:
            init_params = self._init_physical_params(self.config.true_params_shape)

        physical_model = PhysicalModel(
            domain=self.config.domain,
            N=self.config.low_res_n,
            parameters=init_params,
            training=True,
            forcing_func=forcing_func,
            kappa_func=kappa_func,
            eta_func=eta_func,
            rngs=nnx.Rngs(0)
        )

        optimizer = nnx.Optimizer(physical_model, optax.adam(self.config.fem_lr))

        @nnx.jit
        def fem_train_step(model, opt, x, y, u):
            def loss_fn(m):
                u_pred = vmapped_model(m, x, y)
                return jnp.mean(optax.squared_error(u_pred, u))
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(grads)
            return loss

        loss_history = np.zeros(self.config.epochs)
        l2_history = np.zeros(self.config.epochs)
        param_history = np.zeros((self.config.epochs, self.config.true_params_shape[0]))

        print("FEM Training")
        print(f"True parameters:    {format_params(true_params)}")
        print(f"Initial parameters: {format_params(physical_model.parameters)}")
        print(f"Initial param error: {compute_param_error(physical_model.parameters, true_params):.4f}")
        print("-" * 80)

        for epoch in range(self.config.epochs):
            loss_phy = fem_train_step(physical_model, optimizer, xx_train, yy_train, u_train)
            u_pred = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)
            l2_error = self.compute_l2_error(u_pred, u_true)
            param_error = compute_param_error(physical_model.parameters, true_params)

            loss_history[epoch] = loss_phy
            l2_history[epoch] = l2_error
            param_history[epoch] = physical_model.parameters.value

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss_phy:.4e} | L2: {l2_error:.4e} | Param Err: {param_error:.4e}")
                print(f"           | Current params: {format_params(physical_model.parameters)}")

        final_param_error = compute_param_error(physical_model.parameters, true_params)
        print("-" * 80)
        print(f"Final parameters:   {format_params(physical_model.parameters)}")
        print(f"True parameters:    {format_params(true_params)}")
        print(f"Final param error:  {final_param_error:.4e}")

        u_pred = vmapped_model(physical_model, xx_eval, yy_eval).reshape(-1, 1)
        return loss_history, l2_history, param_history, u_pred

# ---------------------------------------------------------------------------
# PINN Trainer
# ---------------------------------------------------------------------------

class PINNTrainer(ModelTrainer):
    def __init__(self, config, phys_param_init_fn: Optional[Callable[[jax.random.PRNGKey], jnp.ndarray]] = None,
                 model_activation=nnx.tanh, freeze_params_phase1: bool = True):
        super().__init__(config)
        self.phys_param_init_fn = phys_param_init_fn
        self.model_activation = model_activation
        self.freeze_params_phase1 = freeze_params_phase1

    def _init_physical_params(self, true_params_shape):
        if self.phys_param_init_fn is not None:
            return nnx.Param(self.phys_param_init_fn(jax.random.PRNGKey(self.config.train_seed)))
        return nnx.Param(jax.random.uniform(
            jax.random.PRNGKey(self.config.train_seed),
            shape=true_params_shape, minval=-1, maxval=1
        ))

    def train(self, xx_train, yy_train, u_train, xx_eval, yy_eval, u_true, true_params,
              initial_params_value=None, forcing_func=None, kappa_func=None, eta_func=None):
        from models.synthetic_model import ResNetSynthetic
        from models.other_models import PINN

        if initial_params_value is not None:
            init_params = nnx.Param(jnp.asarray(initial_params_value))
        else:
            init_params = self._init_physical_params(self.config.true_params_shape)

        model = ResNetSynthetic(
            hidden_dims=self.config.hidden_dims,
            activation=self.model_activation,
            output_dim=1,
            rngs=nnx.Rngs(0)
        )

        pinn = PINN(
            domain=self.config.domain,
            model=model,
            parameters=init_params,
            forcing_func=forcing_func,
            kappa_func=kappa_func,
            eta_func=eta_func,
            rngs=nnx.Rngs(0)
        )

        tx = optax.multi_transform(
            {
                "model": optax.adam(self.config.pinn_model_lr),
                "parameters": optax.adam(self.config.pinn_params_lr),
            },
            nnx.State({"model": "model", "parameters": "parameters"})
        )
        optimizer = nnx.Optimizer(pinn, tx)

        @nnx.jit
        def vmapped_residual(m, xs, ys):
            return jax.vmap(lambda xx, yy: m.residual(xx, yy))(xs, ys)

        @nnx.jit
        def train_step_physics_only(model, opt, x_i, y_i, x_b, y_b):
            def loss_fn(m):
                u_res = vmapped_residual(m, x_i, y_i)
                u_b = vmapped_model(m, x_b, y_b)
                return optax.squared_error(u_res).mean() + 1e1 * optax.squared_error(u_b).mean()
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            if self.freeze_params_phase1:
                grads['parameters'] = jax.tree.map(lambda g: 0.0, grads['parameters'])
            opt.update(grads)
            return loss

        @nnx.jit
        def train_step_combined(model, opt, x, y, u, x_i, y_i, x_b, y_b):
            def loss_data(m):
                u_pred = vmapped_model(m, x, y)
                return optax.squared_error(u_pred, u).mean()
            def loss_physics(m):
                u_res = vmapped_residual(m, x_i, y_i)
                u_b = vmapped_model(m, x_b, y_b)
                return optax.squared_error(u_res).mean() + 1e1 * optax.squared_error(u_b).mean()
            def loss_fn(m):
                return loss_data(m) + loss_physics(m)
            data_loss = loss_data(model)
            total_loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(grads)
            return total_loss, data_loss

        loss_history = np.zeros(self.config.epochs)
        l2_history = np.zeros(self.config.epochs)
        param_history = np.zeros((self.config.epochs, self.config.true_params_shape[0]))

        loss_pinn = 1.0
        rng = jax.random.PRNGKey(self.config.train_seed)

        print("PINN Training")
        print(f"True parameters:    {format_params(true_params)}")
        print(f"Initial parameters: {format_params(pinn.parameters)}")
        print(f"Initial param error: {compute_param_error(pinn.parameters, true_params):.4f}")
        print("-" * 80)

        for epoch in range(self.config.epochs):
            rng, rng1 = jax.random.split(rng, 2)
            x_in, y_in, x_b, y_b = pinn.create_collocation_points(
                self.config.n_interior, self.config.n_boundary, rng1
            )

            if loss_pinn > 1:
                loss_pinn = train_step_physics_only(pinn, optimizer, x_in, y_in, x_b, y_b)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch:4d} | Phase 1 - Physics loss: {loss_pinn:.4e}")
            else:
                loss_val, loss_data = train_step_combined(
                    pinn, optimizer, xx_train, yy_train, u_train, x_in, y_in, x_b, y_b
                )
                u_pred = vmapped_model(pinn, xx_eval, yy_eval).reshape(-1, 1)
                l2_error = self.compute_l2_error(u_pred, u_true)
                param_error = compute_param_error(pinn.parameters, true_params)

                loss_history[epoch] = loss_data
                l2_history[epoch] = l2_error
                param_history[epoch] = pinn.parameters.value

                if epoch % 100 == 0:
                    print(f"Epoch {epoch:4d} | Loss: {loss_data:.4e} | L2: {l2_error:.4e} | Param Err: {param_error:.4e}")
                    print(f"           | Current params: {format_params(pinn.parameters)}")

        final_param_error = compute_param_error(pinn.parameters, true_params)
        print("-" * 80)
        print(f"Final parameters:   {format_params(pinn.parameters)}")
        print(f"True parameters:    {format_params(true_params)}")
        print(f"Final param error:  {final_param_error:.4e}")

        u_pred = vmapped_model(pinn, xx_eval, yy_eval).reshape(-1, 1)
        return loss_history, l2_history, param_history, u_pred

__all__ = [
    'format_params', 'compute_param_error', 'vmapped_model', 'train_step_data_only', 'train_step_hybrid',
    'ModelTrainer', 'HybridTrainer', 'FEMTrainer', 'PINNTrainer'
]
