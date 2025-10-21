# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data.dataloaders import DataLoader_Scalar
import pandas as pd
import os

def gaussian(param, num_gaussians=1, N=100, L=8):
    gaussian_map = 0.1 * np.ones((N, N), dtype=np.float32)
    x = np.linspace(-L // 2, L // 2, N)
    y = np.linspace(-L // 2, L // 2, N)
    x, y = np.meshgrid(x, y, indexing='ij')
    for i in range(num_gaussians):
        gaussian_map += param[i] * np.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2 * num_gaussians + i]) ** 2))
    return gaussian_map


def plot_alpha_evolution(csv_file, pinn_csv_file, flag, trajectory_file=None, traj=False):
    """
    Creates a plot showing the final predictions for FD, HYCO, PINN, and True.
    Optionally overlays data trajectories in red.
    """
    df = pd.read_csv(csv_file)
    df_pinn = pd.read_csv(pinn_csv_file)

    alpha_phys = gaussian(df["params_phys"].values, num_gaussians=2)
    alpha_hyb = gaussian(df["params_hybrid"].values, num_gaussians=2)
    alpha_true = gaussian(df["params_real"].values, num_gaussians=2)
    alpha_pinn = gaussian(df_pinn["params"].values, num_gaussians=2)

    # Build a grid over the spatial domain.
    L_extent = 4
    N_x = alpha_true.shape[0]
    N_y = alpha_true.shape[1]
    x = np.linspace(-L_extent, L_extent, N_x)
    y = np.linspace(-L_extent, L_extent, N_y)
    xx, yy = np.meshgrid(x, y)

    # Load trajectory data if provided
    trajectories = None
    if trajectory_file and traj:
        trajectories = np.load(trajectory_file) 

    # Set a common range for the predictions.
    vmin = np.floor(alpha_true.min())
    vmax = np.ceil(alpha_true.max())

    fig = plt.figure(figsize=(16, 4))
    # Create a gridspec for 1 row and 4 columns (4 panels)
    gs_top = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 1, 1, 1, 0.2],
        left=0.1, right=0.93, top=0.85, bottom=0.1,
        wspace=0.07, hspace=0.07
    )

    # --- Row 1: Kappa maps ---
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, alpha_phys, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_xticks([]); ax0.set_yticks([])


    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, alpha_pinn, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_xticks([]); ax1.set_yticks([])


    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, alpha_hyb, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_xticks([]); ax2.set_yticks([])
    

    ax3 = fig.add_subplot(gs_top[0, 3])
    cf3 = ax3.contourf(xx, yy, alpha_true, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_xticks([]); ax3.set_yticks([])
    if trajectories is not None:
        for traj in range(trajectories.shape[1]):
            ax0.plot(trajectories[:, traj, 0], trajectories[:, traj, 1], 'red', linewidth=1)
            ax1.plot(trajectories[:, traj, 0], trajectories[:, traj, 1], 'red', linewidth=1)
            ax2.plot(trajectories[:, traj, 0], trajectories[:, traj, 1], 'red', linewidth=1)

    # Unified colorbar for row 1 (Kappa)
    ax_cb_top = fig.add_subplot(gs_top[0, 4])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb_top = fig.colorbar(mappable, cax=ax_cb_top)

    # --- Add column labels (above the grid) ---
    col_labels = ["FD", "PINN", "HYCO Physical", "True"]
    col_centers = [0.09 + 0.79 * (i + 0.5) / 4 for i in range(4)]
    for label, xc in zip(col_labels, col_centers):
        fig.text(xc, 0.9, label, ha="center", fontsize=18)

    # --- Add row labels (to the left of the grid) ---
    fig.text(0.0495, 0.49, r"$\kappa$", ha="center", fontsize=22)

    results_folder = 'figures/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    filename = f"{results_folder}heat_coefficients_{flag}_{traj}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Coefficient plot saved as {filename}")

if __name__ == "__main__":
    
    full_file = r'parameters\heat\table\param_2025-07-10_16-01-47_full.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_13-42-53_full.csv'
    trajectory_file = r'data\heat\table\data_full.npy'  # Example trajectory file
    flag = 'full'
    traj = True
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)

    full_file = r'parameters\heat\table\param_2025-07-10_15-36-56_half.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_13-58-13_half.csv'
    trajectory_file = r'data\heat\table\data_half.npy'  # Example trajectory file
    flag = 'half'
    traj = True
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)

    full_file = r'parameters\heat\table\param_2025-07-10_15-05-39_quarter.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_14-13-02_quarter.csv'
    trajectory_file = r'data\heat\table\data_quarter.npy'  # Example trajectory file
    flag = 'quarter'
    traj = True
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)

    full_file = r'parameters\heat\table\param_2025-07-10_16-01-47_full.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_13-42-53_full.csv'
    trajectory_file = r'data\heat\table\data_full.npy'  # Example trajectory file
    flag = 'full'
    traj = False
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)

    full_file = r'parameters\heat\table\param_2025-07-10_15-36-56_half.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_13-58-13_half.csv'
    trajectory_file = r'data\heat\table\data_half.npy'  # Example trajectory file
    flag = 'half'
    traj = False
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)

    full_file = r'parameters\heat\table\param_2025-07-10_15-05-39_quarter.csv'
    pinn_full_file = r'parameters\heat\table\param_pinn_2025-07-10_14-13-02_quarter.csv'
    trajectory_file = r'data\heat\table\data_quarter.npy'  # Example trajectory file
    flag = 'quarter'
    traj = False
    plot_alpha_evolution(full_file, pinn_full_file, flag, trajectory_file=trajectory_file, traj=traj)




