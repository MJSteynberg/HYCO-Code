import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def moving_avg_std(loss, window=10):
    """
    Returns the moving average, standard deviation, and x-axis (epoch indices)
    for a given loss history array.
    """
    if len(loss) < window:
        x = np.arange(len(loss))
        return x, loss, np.zeros_like(loss)

    avg = np.convolve(loss, np.ones(window) / window, mode='valid')
    std = np.array([np.std(loss[i:i + window]) for i in range(len(loss) - window + 1)])
    x = np.arange(window - 1, len(loss))
    return x, avg, std


def exponential_moving_average(loss, alpha=0.1):
    """
    Compute exponential moving average for smoother curves.
    Lower alpha = more smoothing
    """
    ema = np.zeros_like(loss)
    ema[0] = loss[0]
    for i in range(1, len(loss)):
        ema[i] = alpha * loss[i] + (1 - alpha) * ema[i - 1]
    return np.arange(len(loss)), ema


def savitzky_golay_smooth(loss, window_length=51, polyorder=3):
    """
    Apply Savitzky-Golay filter for smoothing while preserving features.
    """
    if len(loss) < window_length:
        window_length = len(loss) if len(loss) % 2 == 1 else len(loss) - 1
    if window_length < polyorder + 1:
        polyorder = window_length - 2

    smoothed = savgol_filter(loss, window_length, polyorder)
    return np.arange(len(loss)), smoothed


def gaussian_smooth(loss, sigma=10):
    """
    Apply Gaussian smoothing filter.
    Higher sigma = more smoothing
    """
    smoothed = gaussian_filter1d(loss.astype(float), sigma=sigma)
    return np.arange(len(loss)), smoothed


def percentile_filter(loss, window=50, percentile=50):
    """
    Apply rolling percentile filter (median by default).
    More robust to outliers than mean.
    """
    if len(loss) < window:
        return np.arange(len(loss)), loss

    filtered = np.zeros_like(loss)
    for i in range(len(loss)):
        start = max(0, i - window // 2)
        end = min(len(loss), i + window // 2 + 1)
        filtered[i] = np.percentile(loss[start:end], percentile)

    return np.arange(len(loss)), filtered


def plot_three_separate(
        phy_state_history,
        hyb_state_history,
        pinn_state_history,
        true_params,
        kappa_func,
        eta_func=None,
        pts_train=None,
        domain=(-3.0, 3.0),
        N=100,
        epochs=5000,
        u_hyb_phys=None,
        u_hyb_syn=None,
        u_fem=None,
        u_pinn=None,
        u_true=None,
        data_loss_hist=None,
        solution_loss_hist=None,
        param_loss_hist=None,
        filename="final_evolution",
        smoothing_method="percentile",  # New parameter
        smoothing_params=None,  # New parameter
        dpi=500  # New parameter for DPI
):
    """
    Generates and saves three separate plots:
    1. Coefficient predictions only
    2. Solution predictions only
    3. Three error plots (data distance, solution distance, parameter distance)

    smoothing_method: 'exponential', 'savgol', 'gaussian', 'percentile', or 'moving_avg'
    smoothing_params: dict with parameters for the chosen smoothing method
    dpi: resolution for saved images (default 300)
    """
    if smoothing_params is None:
        smoothing_params = {}

    # Adjust early stopping based on actual epochs
    early_stopping = min(epochs, 5000)

    # Extract final states
    final_phy = phy_state_history[early_stopping - 1]
    final_pinn = pinn_state_history[early_stopping - 1]
    final_hyb = hyb_state_history[early_stopping - 1]

    # Build shared grid for coefficient predictions
    x = jnp.linspace(domain[0], domain[1], N)
    y = jnp.linspace(domain[0], domain[1], N)
    xx, yy = jnp.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Compute kappa predictions
    kappa_values = {}
    kappa_values['FEM'] = kappa_func(final_phy, xx_flat, yy_flat)
    kappa_values['PINN'] = kappa_func(final_pinn, xx_flat, yy_flat)
    kappa_values['HYCO Physical'] = kappa_func(final_hyb, xx_flat, yy_flat)
    kappa_values['True'] = kappa_func(true_params, xx_flat, yy_flat)

    # Compute eta predictions if eta_func is provided
    eta_values = {}
    if eta_func is not None:
        eta_values['FEM'] = eta_func(final_phy, xx_flat, yy_flat)
        eta_values['PINN'] = eta_func(final_pinn, xx_flat, yy_flat)
        eta_values['HYCO Physical'] = eta_func(final_hyb, xx_flat, yy_flat)
        eta_values['True'] = eta_func(true_params, xx_flat, yy_flat)

    # Plot 1: Coefficients only
    _plot_coefficients_only(xx, yy, kappa_values, eta_values, pts_train, filename, dpi)

    # Plot 2: Solutions only (if provided)
    if all(u is not None for u in [u_hyb_phys, u_hyb_syn, u_fem, u_pinn, u_true]):
        _plot_solutions_only(u_fem, u_pinn, u_hyb_phys, u_hyb_syn, u_true, domain, filename, dpi)

    # Plot 3: Error plots (if provided)
    if all(loss is not None for loss in [data_loss_hist, solution_loss_hist, param_loss_hist]):
        _plot_errors_only(data_loss_hist, solution_loss_hist, param_loss_hist,
                          early_stopping, filename, smoothing_method, smoothing_params, dpi)


def _plot_coefficients_only(xx, yy, kappa_values, eta_values, pts_train, filename, dpi=500):
    """Plot coefficient predictions only"""

    # Determine colorbar range
    all_coeff_values = list(kappa_values.values())
    if eta_values:
        all_coeff_values.extend(eta_values.values())
    vmin = float(jnp.floor(jnp.min(jnp.array(all_coeff_values))))
    vmax = float(jnp.ceil(jnp.max(jnp.array(all_coeff_values))))

    # Create figure layout
    has_eta = len(eta_values) > 0
    n_rows = 2 if has_eta else 1
    fig_height = 8 if has_eta else 4

    fig = plt.figure(figsize=(15, fig_height), dpi=dpi)
    gs = fig.add_gridspec(
        n_rows, 5, width_ratios=[1, 1, 1, 1, 0.2],
        left=0.08, right=0.93, top=0.93, bottom=0.15,
        wspace=0.05, hspace=0.05
    )

    # Plot coefficients
    methods = ['FEM', 'PINN', 'HYCO Physical', 'True']
    axes = []

    # Kappa row
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])
        ax.contourf(xx, yy, kappa_values[method].reshape(xx.shape),
                    levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')  # Force square aspect ratio
        axes.append(ax)

    # Eta row (if exists)
    if has_eta:
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[1, i])
            ax.contourf(xx, yy, eta_values[method].reshape(xx.shape),
                        levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')  # Force square aspect ratio
            axes.append(ax)

    # Add training points if provided
    if pts_train is not None:
        for ax in axes:
            ax.scatter(pts_train[:, 0], pts_train[:, 1],
                       marker="o", c='#d62728', s=15)

    # Add colorbar
    ax_cb = fig.add_subplot(gs[:, 4])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=ax_cb)
    cb.set_ticks(np.arange(int(vmin), int(vmax) + 1))

    # Add labels - FIXED: Calculate proper column centers based on subplot positions
    for i, method in enumerate(methods):
        # Get the subplot position for the top row
        subplot_pos = gs[0, i].get_position(fig)
        # Calculate the center x-coordinate of the subplot
        center_x = subplot_pos.x0 + subplot_pos.width / 2
        fig.text(center_x, 0.96, method, ha="center", va="center", fontsize=18)

    # Row labels - IMPROVED: Better vertical centering
    if has_eta:
        # Calculate the vertical center of each row based on subplot positions
        kappa_row_pos = gs[0, 0].get_position(fig)
        eta_row_pos = gs[1, 0].get_position(fig)

        kappa_center_y = kappa_row_pos.y0 + kappa_row_pos.height / 2
        eta_center_y = eta_row_pos.y0 + eta_row_pos.height / 2

        fig.text(0.04, kappa_center_y, r"$\kappa$", ha="center", va="center", fontsize=25)
        fig.text(0.04, eta_center_y, r"$\eta$", ha="center", va="center", fontsize=25)
    else:
        # Single row case
        kappa_row_pos = gs[0, 0].get_position(fig)
        kappa_center_y = kappa_row_pos.y0 + kappa_row_pos.height / 2
        fig.text(0.04, kappa_center_y, r"$\kappa$", ha="center", va="center", fontsize=25)

    # Save plot
    coefficients_filename = f"src/results/{filename}_coefficients.png"
    plt.savefig(coefficients_filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _plot_solutions_only(u_fem, u_pinn, u_hyb_phys, u_hyb_syn, u_true, domain, filename, dpi=500):
    """Plot solutions only"""

    # Prepare solution data
    N_sol = int(np.sqrt(u_hyb_phys.shape[0]))
    sol_shape = (N_sol, N_sol)
    xx_sol = jnp.linspace(domain[0], domain[1], N_sol)
    yy_sol = jnp.linspace(domain[0], domain[1], N_sol)

    # Solution values
    solutions = {
        'FEM': u_fem.reshape(sol_shape),
        'PINN': u_pinn.reshape(sol_shape),
        'HYCO Physical': u_hyb_phys.reshape(sol_shape),
        'HYCO Synthetic': u_hyb_syn.reshape(sol_shape),
        'True': u_true.reshape(sol_shape)
    }

    # Determine color ranges
    vmin_sol = min(sol.min() for sol in solutions.values())
    vmax_sol = max(sol.max() for sol in solutions.values())

    # Create figure
    fig = plt.figure(figsize=(20, 4), dpi=dpi)
    gs = fig.add_gridspec(
        1, 6, width_ratios=[1, 1, 1, 1, 1, 0.2],
        left=0.06, right=0.94, top=0.93, bottom=0.15,
        wspace=0.04, hspace=0.04
    )

    # Plot solutions
    methods = ['FEM', 'PINN', 'HYCO Physical', 'HYCO Synthetic', 'True']

    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])
        ax.contourf(xx_sol, yy_sol, solutions[method], levels=200,
                    vmin=vmin_sol, vmax=vmax_sol, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')  # Force square aspect ratio
        ax.set_title(method, fontsize=20)

    # Add colorbar
    ax_cb_sol = fig.add_subplot(gs[0, 5])
    norm_sol = Normalize(vmin=vmin_sol, vmax=vmax_sol)
    mappable_sol = ScalarMappable(norm=norm_sol, cmap="viridis")
    mappable_sol.set_array([])
    fig.colorbar(mappable_sol, cax=ax_cb_sol)

    # Add row label - IMPROVED: Better vertical centering
    row_pos = gs[0, 0].get_position(fig)
    row_center_y = row_pos.y0 + row_pos.height / 2
    fig.text(0.03, row_center_y, r"$u_m$", ha="center", va="center", fontsize=25)

    # Save plot
    solutions_filename = f"src/results/{filename}_solutions.png"
    plt.savefig(solutions_filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _plot_errors_only(data_loss_hist, solution_loss_hist, param_loss_hist,
                      early_stopping, filename, smoothing_method="exponential",
                      smoothing_params=None, dpi=500):
    """Plot three error histories with enhanced smoothing options applied only to HYCO Synthetic"""

    plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.major.size': 3,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.width': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.labelpad': 10,
    })

    colors_nature = {
        'HYCO Physical': '#C40606',
        'HYCO Synthetic': '#F58220',
        'FEM': '#00A69D',
        'PINN': '#003F8D'
    }
    
    # Apply moving average smoothing
    window = 20  # Increased for smoother curves

    # Create figure with double column width for Nature format
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 5))  # ~183mm width (double column)
    smooth_func = moving_avg_std

    # Plot 1: Data distance error
    if isinstance(data_loss_hist, dict):
        for method, loss_hist in data_loss_hist.items():
            loss_to_plot = loss_hist[:early_stopping]
            print(method)
            if len(loss_to_plot) > 0:
                x_loss, avg_loss, _ = smooth_func(loss_to_plot, window=window)
                
                ax1.plot(x_loss, avg_loss, label=method, color=colors_nature[method])
    else:
        # Single array case
        loss_to_plot = data_loss_hist[:early_stopping]
        if len(loss_to_plot) > 0:
            x_loss, avg_loss, _ = moving_avg_std(loss_to_plot, window=1)
            ax1.plot(x_loss, avg_loss, label='Data Loss', color='blue', lw=2)

    ax1.set_yscale("log")
    ax1.set_ylabel(r"$e^m_d$ ", rotation=0)
    ax1.set_xticklabels([])
    ax1.set_xlim([0, early_stopping])
    ax1.legend(bbox_to_anchor=(0.82, 1.2), loc='upper left', frameon=True, fontsize=7)
    ax1.grid(False)

    # Plot 2: Solution distance error
    if isinstance(solution_loss_hist, dict):
        for method, loss_hist in solution_loss_hist.items():
            loss_to_plot = loss_hist[:early_stopping]
            if len(loss_to_plot) > 0:
                x_loss, avg_loss, _ = moving_avg_std(loss_to_plot, window=20)
                ax2.plot(x_loss, avg_loss, label=method, color=colors_nature[method])
    else:
        # Single array case
        loss_to_plot = solution_loss_hist[:early_stopping]
        if len(loss_to_plot) > 0:
            x_loss, avg_loss, _ = moving_avg_std(loss_to_plot, window=1)
            ax2.plot(x_loss, avg_loss, label='Solution Loss', color='red')

    ax2.set_yscale("log")
    ax2.set_ylabel(r"$e^m_s$", rotation=0)
    ax2.set_xticklabels([])
    ax2.set_xlim([0, early_stopping])
    ax2.grid(False)

    # Plot 3: Parameter distance error
    if isinstance(param_loss_hist, dict):
        for method, loss_hist in param_loss_hist.items():
            loss_to_plot = loss_hist[:early_stopping]
            if len(loss_to_plot) > 0:
                
                x_loss, avg_loss, _ = moving_avg_std(loss_to_plot, window=20)
                ax3.plot(x_loss, avg_loss, label=method, color=colors_nature[method])
    else:
        # Single array case
        loss_to_plot = param_loss_hist[:early_stopping]
        if len(loss_to_plot) > 0:
            x_loss, avg_loss, _ = moving_avg_std(loss_to_plot, window=20)
            ax3.plot(x_loss, avg_loss, label='Parameter Loss', color='green')

    ax3.set_yscale('log')  # Log scale for parameter errors
    ax3.set_yticks([1e-1])
    ax3.tick_params(axis='y', which='minor', labelleft=False)  # Hide minor tick labels but keep ticks
    ax3.set_ylabel("$e^m_p$", rotation=0)
    ax3.set_xlim([0, early_stopping])  # Set x-limits to match the number of epochs
    ax3.set_xlabel("Epochs")
    

    # Adjust layout
    plt.tight_layout()

    # Save plot
    errors_filename = f"src/results/{filename}_errors.png"
    # Save as high-res pdf for preview
    plt.savefig(f"{errors_filename}.pdf", dpi=1200, bbox_inches='tight', 
                format='pdf', facecolor='white')