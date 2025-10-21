"""
Error plotting script for Gray-Scott PINN training.
Based on ERROR/PLOT_ERRORS.py patterns.

Plots evolution of:
1. MSE errors (data fitting accuracy)
2. L2 errors (solution field accuracy) 
3. Parameter errors (learned parameter accuracy)

Designed to accommodate multiple model types (PINN, HYCO, NN) when available.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple
import glob
from datetime import datetime


def moving_avg_std(loss: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the moving average, standard deviation, and x-axis (epoch indices)
    for a given loss history array.
    
    Args:
        loss: Loss history array
        window: Window size for moving average
        
    Returns:
        Tuple of (x_indices, moving_avg, moving_std)
    """
    # Adjust window size if we don't have enough data points
    effective_window = min(window, max(1, len(loss) // 2))
    
    if len(loss) <= 2 or effective_window == 1:
        # Not enough points for meaningful moving average, return original
        x = np.arange(len(loss))
        return x, loss, np.zeros_like(loss)
    
    avg = np.convolve(loss, np.ones(effective_window)/effective_window, mode='valid')
    std = np.array([np.std(loss[i:i+effective_window]) for i in range(len(loss)-effective_window+1)])
    x = np.arange(effective_window-1, len(loss))
    return x, avg, std


def load_pinn_error_data(csv_path: str) -> pd.DataFrame:
    """
    Load PINN error data from CSV file.
    
    Args:
        csv_path: Path to the error tracking CSV file
        
    Returns:
        DataFrame with error data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PINN error file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded PINN error data: {len(df)} epochs from {csv_path}")
    
    # Verify required columns exist
    required_cols = ['epoch', 'mse_physical', 'l2_norm_error', 'parameter_error']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Available columns: {list(df.columns)}")
    
    return df


def find_latest_pinn_error_file(results_dir: str, experiment_name: str = None) -> str:
    """
    Find the most recent PINN error tracking file.
    
    Args:
        results_dir: Base results directory
        experiment_name: Specific experiment name, or None to search all
        
    Returns:
        Path to the most recent error file
    """
    if experiment_name:
        # Search in specific experiment directory
        search_pattern = os.path.join(results_dir, experiment_name, "error_tracking_pinn_*.csv")
    else:
        # Search in both subdirectories and root results directory
        search_patterns = [
            os.path.join(results_dir, "*", "error_tracking_pinn_*.csv"),
            os.path.join(results_dir, "error_tracking_pinn_*.csv")
        ]
        error_files = []
        for pattern in search_patterns:
            error_files.extend(glob.glob(pattern))
    
    if experiment_name:
        error_files = glob.glob(search_pattern)
    
    if not error_files:
        search_info = f"{search_pattern}" if experiment_name else f"{search_patterns}"
        raise FileNotFoundError(f"No PINN error files found matching: {search_info}")
    
    # Sort by modification time and return the most recent
    error_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = error_files[0]
    print(f"Using latest PINN error file: {latest_file}")
    
    return latest_file


def plot_gray_scott_errors(pinn_csv_path: str, 
                          hyco_csv_path: str = None,
                          nn_csv_path: str = None,
                          save_path: str = "figures/gray_scott_errors.png",
                          window: int = 10) -> None:
    """
    Creates a comprehensive plot showing the evolution of MSE, L2, and parameter errors
    for Gray-Scott PINN training, following ERROR folder patterns.
    
    Args:
        pinn_csv_path: Path to PINN error CSV file
        hyco_csv_path: Path to HYCO error CSV file (optional)
        nn_csv_path: Path to NN error CSV file (optional)
        save_path: Where to save the plot
        window: Window size for moving average smoothing
    """
    
    # Load PINN data (required)
    df_pinn = load_pinn_error_data(pinn_csv_path)
    
    # Load other model data if available
    df_hyco = None
    df_nn = None
    
    if hyco_csv_path and os.path.exists(hyco_csv_path):
        df_hyco = pd.read_csv(hyco_csv_path)
        print(f"Loaded HYCO error data: {len(df_hyco)} epochs")
    
    if nn_csv_path and os.path.exists(nn_csv_path):
        df_nn = pd.read_csv(nn_csv_path)
        print(f"Loaded NN error data: {len(df_nn)} epochs")
    
    # Create figure with 3 subplots (following ERROR folder layout)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Create gradient background for all subplots
    gradient = np.linspace(1, 0.6, 256)
    gradient = np.outer(gradient, gradient)
    
    # ===== MSE Errors Plot =====
    print("Plotting MSE errors...")
    
    # PINN data
    mse_pinn = df_pinn['mse_physical'].values
    x_mse_pinn, avg_mse_pinn, _ = moving_avg_std(mse_pinn, window=window)
    
    # Determine y-axis limits
    mse_max = 1.2 * avg_mse_pinn.max()
    
    # Add other models when available
    if df_hyco is not None and 'mse_physical' in df_hyco.columns:
        mse_hyco = df_hyco['mse_physical'].values
        x_mse_hyco, avg_mse_hyco, _ = moving_avg_std(mse_hyco, window=window)
        mse_max = max(mse_max, 1.2 * avg_mse_hyco.max())
    
    if df_nn is not None and 'mse_physical' in df_nn.columns:
        mse_nn = df_nn['mse_physical'].values
        x_mse_nn, avg_mse_nn, _ = moving_avg_std(mse_nn, window=window)
        mse_max = max(mse_max, 1.2 * avg_mse_nn.max())
    
    # Background gradient for MSE plot
    ax1.imshow(gradient, extent=[0, len(df_pinn), 0, mse_max], 
               aspect='auto', cmap='Blues', origin='lower', zorder=0, alpha=0.1)
    ax1.imshow(gradient, extent=[0, len(df_pinn), 0, mse_max], 
               aspect='auto', cmap='Grays', origin='lower', zorder=0, alpha=0.05)
    
    # Plot PINN
    ax1.plot(x_mse_pinn, avg_mse_pinn, label="PINN", lw=2, color='blue')
    
    # Plot other models if available
    if df_hyco is not None and 'mse_physical' in df_hyco.columns:
        ax1.plot(x_mse_hyco, avg_mse_hyco, label="HYCO", lw=2, color='red')
    
    if df_nn is not None and 'mse_physical' in df_nn.columns:
        ax1.plot(x_mse_nn, avg_mse_nn, label="NN", lw=2, color='green')
    
    ax1.set_ylabel("$e^m_d$", fontsize=18, rotation=0, labelpad=20)
    ax1.set_yscale("log")
    ax1.set_xticklabels([])
    ax1.set_xticks([])  # Hide x-ticks for MSE plot
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_title("Gray-Scott PINN Error Evolution", fontsize=16, pad=20)
    
    # ===== L2 Errors Plot =====
    print("Plotting L2 errors...")
    
    # PINN data
    l2_pinn = df_pinn['l2_norm_error'].values
    x_l2_pinn, avg_l2_pinn, _ = moving_avg_std(l2_pinn, window=window)
    
    # Determine y-axis limits
    l2_max = 1.2 * avg_l2_pinn.max()
    
    # Add other models when available
    if df_hyco is not None and 'l2_norm_error' in df_hyco.columns:
        l2_hyco = df_hyco['l2_norm_error'].values
        x_l2_hyco, avg_l2_hyco, _ = moving_avg_std(l2_hyco, window=window)
        l2_max = max(l2_max, 1.2 * avg_l2_hyco.max())
    
    if df_nn is not None and 'l2_norm_error' in df_nn.columns:
        l2_nn = df_nn['l2_norm_error'].values
        x_l2_nn, avg_l2_nn, _ = moving_avg_std(l2_nn, window=window)
        l2_max = max(l2_max, 1.2 * avg_l2_nn.max())
    
    # Background gradient for L2 plot
    ax2.imshow(gradient, extent=[0, len(df_pinn), 0, l2_max], 
               aspect='auto', cmap='Blues', origin='lower', zorder=0, alpha=0.1)
    ax2.imshow(gradient, extent=[0, len(df_pinn), 0, l2_max], 
               aspect='auto', cmap='Grays', origin='lower', zorder=0, alpha=0.05)
    
    # Plot PINN
    ax2.plot(x_l2_pinn, avg_l2_pinn, label="PINN", lw=2, color='blue')
    
    # Plot other models if available
    if df_hyco is not None and 'l2_norm_error' in df_hyco.columns:
        ax2.plot(x_l2_hyco, avg_l2_hyco, label="HYCO", lw=2, color='red')
    
    if df_nn is not None and 'l2_norm_error' in df_nn.columns:
        ax2.plot(x_l2_nn, avg_l2_nn, label="NN", lw=2, color='green')
    
    ax2.set_ylabel("$e^m_s$", fontsize=18, rotation=0, labelpad=20)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.set_yscale("log")
    ax2.set_xticklabels([])
    ax2.set_xticks([])  # Hide x-ticks for L2 plot
    
    # ===== Parameter Errors Plot =====
    print("Plotting parameter errors...")
    
    # PINN data
    p_error_pinn = df_pinn['parameter_error'].values
    x_p_pinn, avg_p_pinn, _ = moving_avg_std(p_error_pinn, window=window)
    
    # Determine y-axis limits
    p_max = 1.2 * avg_p_pinn.max()
    
    # Add other models when available
    if df_hyco is not None and 'parameter_error' in df_hyco.columns:
        p_error_hyco = df_hyco['parameter_error'].values
        x_p_hyco, avg_p_hyco, _ = moving_avg_std(p_error_hyco, window=window)
        p_max = max(p_max, 1.2 * avg_p_hyco.max())
    
    if df_nn is not None and 'parameter_error' in df_nn.columns:
        p_error_nn = df_nn['parameter_error'].values
        x_p_nn, avg_p_nn, _ = moving_avg_std(p_error_nn, window=window)
        p_max = max(p_max, 1.2 * avg_p_nn.max())
    
    # Background gradient for parameter plot
    ax3.imshow(gradient, extent=[0, len(df_pinn), 0, p_max], 
               aspect='auto', cmap='Blues', origin='lower', zorder=0, alpha=0.1)
    ax3.imshow(gradient, extent=[0, len(df_pinn), 0, p_max], 
               aspect='auto', cmap='Grays', origin='lower', zorder=0, alpha=0.05)
    
    # Plot PINN
    ax3.plot(x_p_pinn, avg_p_pinn, label="PINN", lw=2, color='blue')
    
    # Plot other models if available  
    if df_hyco is not None and 'parameter_error' in df_hyco.columns:
        ax3.plot(x_p_hyco, avg_p_hyco, label="HYCO", lw=2, color='red')
    
    if df_nn is not None and 'parameter_error' in df_nn.columns:
        ax3.plot(x_p_nn, avg_p_nn, label="NN", lw=2, color='green')
    
    ax3.set_xlabel("Epochs", fontsize=18)
    ax3.set_ylabel(r"$e^m_p$", fontsize=18, rotation=0, labelpad=20)
    ax3.set_yscale("log")
    ax3.legend(fontsize=12, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Error evolution plot saved as: {save_path}")
    plt.show()


def plot_parameter_evolution(pinn_csv_path: str, 
                           save_path: str = "figures/parameter_evolution.png") -> None:
    """
    Plot the evolution of learned parameters (Du, Dv) during training.
    
    Args:
        pinn_csv_path: Path to PINN error CSV file
        save_path: Where to save the plot
    """
    df = load_pinn_error_data(pinn_csv_path)
    
    # Check if parameter columns exist
    param_cols = ['du_error', 'dv_error']
    available_params = [col for col in param_cols if col in df.columns]
    
    if not available_params:
        print("Warning: No parameter error columns found. Skipping parameter evolution plot.")
        return
    
    fig, axes = plt.subplots(len(available_params), 1, figsize=(10, 3*len(available_params)))
    if len(available_params) == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'purple', 'orange']
    
    for i, param_col in enumerate(available_params):
        param_name = param_col.replace('_error', '').upper()
        
        epochs = df['epoch'].values
        param_errors = df[param_col].values
        
        axes[i].plot(epochs, param_errors, label=f"{param_name} Error", 
                    color=colors[i % len(colors)], linewidth=2)
        axes[i].set_ylabel(f"{param_name} Error", fontsize=12)
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel("Epochs", fontsize=12)
    plt.suptitle("Parameter Learning Evolution", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Parameter evolution plot saved as: {save_path}")
    plt.show()


def main():
    """
    Main function to generate error plots.
    
    Usage examples:
    1. Auto-find latest PINN error file:
       python plot_gray_scott_errors.py
    
    2. Specify specific error file:
       python plot_gray_scott_errors.py --pinn_csv path/to/error_file.csv
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot Gray-Scott PINN error evolution")
    parser.add_argument("--pinn_csv", type=str, default=None,
                       help="Path to PINN error CSV file")
    parser.add_argument("--hyco_csv", type=str, default=None,
                       help="Path to HYCO error CSV file")
    parser.add_argument("--nn_csv", type=str, default=None,
                       help="Path to NN error CSV file")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Base results directory")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Specific experiment name")
    parser.add_argument("--save_dir", type=str, default="figures",
                       help="Directory to save plots")
    parser.add_argument("--window", type=int, default=10,
                       help="Moving average window size")
    
    args = parser.parse_args()
    
    # Find PINN error file if not specified
    if args.pinn_csv is None:
        try:
            args.pinn_csv = find_latest_pinn_error_file(args.results_dir, args.experiment)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify --pinn_csv or ensure error tracking files exist.")
            return
    
    # Generate plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Main error evolution plot
    main_save_path = os.path.join(args.save_dir, f"gray_scott_errors_{timestamp}.png")
    plot_gray_scott_errors(
        pinn_csv_path=args.pinn_csv,
        hyco_csv_path=args.hyco_csv,
        nn_csv_path=args.nn_csv,
        save_path=main_save_path,
        window=args.window
    )
    
    # Parameter evolution plot
    param_save_path = os.path.join(args.save_dir, f"parameter_evolution_{timestamp}.png")
    plot_parameter_evolution(
        pinn_csv_path=args.pinn_csv,
        save_path=param_save_path
    )
    
    print(f"\n✓ All plots generated successfully!")
    print(f"  - Error evolution: {main_save_path}")
    print(f"  - Parameter evolution: {param_save_path}")


if __name__ == "__main__":
    main()
