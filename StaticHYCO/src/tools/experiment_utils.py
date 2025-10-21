"""Shared experiment utility functions for PDE training scripts.

Consolidates duplicated logic from `darcy.py` and `helmholtz.py`:
- replace_zeros_linear
- replace_zeros_nearest
- generic save_results (file writing + final param error summary)

The goal is to keep PDE scripts focused on problem definition and orchestration.

Expected result tuple layout (matching existing scripts):
(
  loss_history_hyb_phys, loss_history_hyb_syn, l2_history_phys, l2_history_syn, param_history_hyb,
  u_hyb_phys, u_hyb_syn,
  loss_history_fem, l2_history_fem, param_history_fem, u_fem,
  loss_history_pinn, l2_history_pinn, param_history_pinn, u_pinn
)

All arrays are assumed to be numpy / jax arrays that implement the buffer protocol.
"""
from __future__ import annotations

import os
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

from .training import compute_param_error  # reuse existing helper

__all__ = [
    "replace_zeros_linear",
    "replace_zeros_nearest",
    "save_results_generic",
]

def replace_zeros_linear(arr):
    """Replace zero entries by linear interpolation of nearest non-zero points.
    Works for 1D arrays; returns array unchanged if all zeros or no interpolation possible.
    """
    arr = np.asarray(arr)
    indices = np.arange(len(arr))
    mask = arr != 0
    if mask.sum() == 0:
        return arr
    return np.interp(indices, indices[mask], arr[mask])


def replace_zeros_nearest(arr):
    """Replace zeros with the value of the nearest non-zero sample (1D)."""
    arr = np.asarray(arr)
    nonzero_indices = np.nonzero(arr)[0]
    if len(nonzero_indices) == 0:
        return arr
    zero_indices = np.where(arr == 0)[0]
    out = arr.copy()
    for zi in zero_indices:
        nearest = nonzero_indices[np.abs(nonzero_indices - zi).argmin()]
        out[zi] = arr[nearest]
    return out


def save_results_generic(
    pde_name: str,
    error: float | int | str,
    all_results: Tuple[Any, ...],
    true_params,
    out_root: str = "src/files",
):
    """Save experiment results for a PDE script.

    Parameters
    ----------
    pde_name : str
        Subdirectory name (e.g. 'darcy' or 'helmholtz').
    error : scalar / str
        Noise level or experiment identifier used in filenames.
    all_results : tuple
        Tuple matching the documented ordering above.
    true_params : array-like
        Ground-truth parameter vector for computing final errors.
    out_root : str
        Root directory where per-PDE subfolder exists / is created.
    """
    (
        loss_history_hyb_phys, loss_history_hyb_syn, l2_history_phys, l2_history_syn, param_history_hyb,
        u_hyb_phys, u_hyb_syn,
        loss_history_fem, l2_history_fem, param_history_fem, u_fem,
        loss_history_pinn, l2_history_pinn, param_history_pinn, u_pinn,
    ) = all_results

    # Compute final parameter errors
    final_param_error_hyb = compute_param_error(param_history_hyb[-1], true_params)
    final_param_error_fem = compute_param_error(param_history_fem[-1], true_params)
    final_param_error_pinn = compute_param_error(param_history_pinn[-1], true_params)

    out_dir = os.path.join(out_root, pde_name)
    os.makedirs(out_dir, exist_ok=True)

    # Summary text file
    with open(os.path.join(out_dir, f"results_{error}.txt"), "w") as f:
        f.write("FEM Loss min in last 100: " + str(jnp.min(loss_history_fem[-100:])) + "\n")
        f.write("Hybrid Phys Loss min in last 100: " + str(jnp.min(loss_history_hyb_phys[-100:])) + "\n")
        f.write("Hybrid Syn Loss min in last 100: " + str(jnp.min(loss_history_hyb_syn[-100:])) + "\n")
        f.write("PINN Loss min in last 100: " + str(jnp.min(loss_history_pinn[-100:])) + "\n")
        f.write("FEM Loss final: " + str(loss_history_fem[-1]) + "\n")
        f.write("Hybrid Phys Loss final: " + str(loss_history_hyb_phys[-1]) + "\n")
        f.write("Hybrid Syn Loss final: " + str(loss_history_hyb_syn[-1]) + "\n")
        f.write("PINN Loss final: " + str(loss_history_pinn[-1]) + "\n")
        f.write("FEM Final Param Error: " + str(final_param_error_fem) + "\n")
        f.write("Hybrid Final Param Error: " + str(final_param_error_hyb) + "\n")
        f.write("PINN Final Param Error: " + str(final_param_error_pinn) + "\n")

    prefix = os.path.join(out_dir, str(error))  # Not used for now; keep pattern stable
    # Maintain original naming convention (no extra subfolder):
    base = out_dir  # original scripts save directly in pde folder
    jnp.save(f"{base}/hybrid_loss_phys_{error}.npy", loss_history_hyb_phys)
    jnp.save(f"{base}/hybrid_loss_syn_{error}.npy", loss_history_hyb_syn)
    jnp.save(f"{base}/hybrid_l2_phys_{error}.npy", l2_history_phys)
    jnp.save(f"{base}/hybrid_l2_syn_{error}.npy", l2_history_syn)
    jnp.save(f"{base}/hybrid_params_{error}.npy", param_history_hyb)
    jnp.save(f"{base}/u_hyb_phys_{error}.npy", u_hyb_phys)
    jnp.save(f"{base}/u_hyb_syn_{error}.npy", u_hyb_syn)

    jnp.save(f"{base}/fem_loss_{error}.npy", loss_history_fem)
    jnp.save(f"{base}/fem_l2_{error}.npy", l2_history_fem)
    jnp.save(f"{base}/fem_params_{error}.npy", param_history_fem)
    jnp.save(f"{base}/u_fem_{error}.npy", u_fem)

    jnp.save(f"{base}/pinn_loss_{error}.npy", loss_history_pinn)
    jnp.save(f"{base}/pinn_l2_{error}.npy", l2_history_pinn)
    jnp.save(f"{base}/pinn_params_{error}.npy", param_history_pinn)
    jnp.save(f"{base}/u_pinn_{error}.npy", u_pinn)

    return {
        "fem": final_param_error_fem,
        "hybrid": final_param_error_hyb,
        "pinn": final_param_error_pinn,
    }
