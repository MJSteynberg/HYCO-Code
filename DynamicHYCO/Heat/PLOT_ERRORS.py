# Nature-style plot for errors evolution from CSV data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set Nature-style parameters
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

def moving_avg_std(loss, window=10):
    """
    Returns the moving average, standard deviation, and x-axis (epoch indices)
    for a given loss history array.
    """
    if len(loss) < window:
        x = np.arange(len(loss))
        return x, loss, np.zeros_like(loss)
    avg = np.convolve(loss, np.ones(window)/window, mode='valid')
    std = np.array([np.std(loss[i:i+window]) for i in range(len(loss)-window+1)])
    x = np.arange(window-1, len(loss))
    return x, avg, std

def plot_errors(csv_file, pinn_csv_file, flag):
    """
    Creates a Nature-style comprehensive plot showing the evolution of MSE, L2, and parameter errors.
    """
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    df = pd.read_csv(csv_file)
    df_pinn = pd.read_csv(pinn_csv_file)
    
    # Create figure with double column width for Nature format
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 5))  # ~183mm width (double column)
    
    # Define colors to match original figure
    colors = {
        'hyco_phys': '#C40606',          
        'hyco_syn': '#F58220',         
        'fd': '#00A69D',                
        'pinn': '#003F8D'           
    }
    
    # Apply moving average smoothing
    window = 20  # Increased for smoother curves
    
    # ===== MSE Errors Plot (Panel a) =====
    mse_hybrid_syn = df['mse_hybrid_syn'].values
    mse_hybrid_phys = df['mse_hybrid_phys'].values
    mse_phys_only = df['mse_phys_only'].values
    mse_pinn = df_pinn['mse'].values
    
    x_mse_syn, avg_mse_syn, _ = moving_avg_std(mse_hybrid_syn, window=window)
    x_mse_phys, avg_mse_phys, _ = moving_avg_std(mse_hybrid_phys, window=window)
    x_mse_only, avg_mse_only, _ = moving_avg_std(mse_phys_only, window=window)
    x_mse_pinn, avg_mse_pinn, _ = moving_avg_std(mse_pinn, window=window)
    
    ax1.plot(x_mse_syn, avg_mse_syn, label="HYCO Synthetic", 
              color=colors['hyco_syn'])
    ax1.plot(x_mse_phys, avg_mse_phys, label="HYCO Physical", 
              color=colors['hyco_phys'])
    ax1.plot(x_mse_only, avg_mse_only, label="FD", 
              color=colors['fd'])
    ax1.plot(x_mse_pinn, avg_mse_pinn, label="PINN", 
              color=colors['pinn'])
    
    ax1.set_yscale('log')  # Log scale for MSE
    ax1.set_ylabel("$e^m_d$", rotation=0)
    ax1.set_xticklabels([])  # Hide x-ticks for MSE
    ax1.set_xlim([0, len(df)])  # Set x-limits to match the number of epochs
    ax1.legend(bbox_to_anchor=(0.82, 1.2), loc='upper left', frameon=True, fontsize=7)
    
    
    # ===== L2 Errors Plot (Panel b) =====
    l2_hybrid_syn = df['l2_error_hybrid_syn'].values
    l2_hybrid_phys = df['l2_error_hybrid_phys'].values
    l2_phys_only = df['l2_error_phys_only'].values
    l2_pinn = df_pinn['l2_norm'].values
    
    x_l2_syn, avg_l2_syn, _ = moving_avg_std(l2_hybrid_syn, window=window)
    x_l2_phys, avg_l2_phys, _ = moving_avg_std(l2_hybrid_phys, window=window)
    x_l2_only, avg_l2_only, _ = moving_avg_std(l2_phys_only, window=window)
    x_l2_pinn, avg_l2_pinn, _ = moving_avg_std(l2_pinn, window=window)

    ax2.plot(x_l2_syn, avg_l2_syn, label="HYCO Synthetic", 
              color=colors['hyco_syn'])
    ax2.plot(x_l2_phys, avg_l2_phys, label="HYCO Physical", 
              color=colors['hyco_phys'])
    ax2.plot(x_l2_only, avg_l2_only, label="FD", 
              color=colors['fd'])
    ax2.plot(x_l2_pinn, avg_l2_pinn, label="PINN", 
              color=colors['pinn'])
    
    ax2.set_yscale('log')  # Log scale for L2
    ax2.set_ylabel("$e^m_s$", rotation=0)
    ax2.set_xticklabels([])  # Hide x-ticks for L2
    ax2.set_xlim([0, len(df)])  # Set x-limits to match the number of epochs

    # ===== Parameter Errors Plot (Panel c) =====
    p_error_hybrid_syn = df['p_error_hybrid_syn'].values
    p_error_hybrid_phys = df['p_error_hybrid_phys'].values
    p_error_phys_only = df['p_error_phys_only'].values
    p_error_pinn = df_pinn['p_norm'].values
    
    x_p_syn, avg_p_syn, _ = moving_avg_std(p_error_hybrid_syn, window=window)
    x_p_phys, avg_p_phys, _ = moving_avg_std(p_error_hybrid_phys, window=window)
    x_p_only, avg_p_only, _ = moving_avg_std(p_error_phys_only, window=window)
    x_p_pinn, avg_p_pinn, _ = moving_avg_std(p_error_pinn, window=window)
    
    ax3.plot(x_p_phys, avg_p_phys, label="HYCO Physical", 
              color=colors['hyco_phys'])
    ax3.plot(x_p_only, avg_p_only, label="FD", 
              color=colors['fd'])
    ax3.plot(x_p_pinn, avg_p_pinn, label="PINN", 
              color=colors['pinn'])
    
    ax3.set_yscale('log')  # Log scale for parameter errors
    ax3.set_yticks([1])
    ax3.set_yticklabels(['$10^0$'])
    ax3.tick_params(axis='y', which='minor', labelleft=False)  # Hide minor tick labels but keep ticks
    ax3.set_ylabel("$e^m_p$", rotation=0)
    ax3.set_xlim([0, len(df)])  # Set x-limits to match the number of epochs
    ax3.set_xlabel("Epochs")

    plt.tight_layout()
    # Create results folder
    results_folder = 'figures/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Save with Nature requirements
    filename_base = f"{results_folder}heat_error_{flag}"

    # Save as high-res pdf for preview
    plt.savefig(f"{filename_base}.pdf", dpi=1200, bbox_inches='tight', 
                format='pdf', facecolor='white')
    

if __name__ == "__main__":
    # Create only the quarter version in double-column format
    quarter_file = r'parameters\heat\table\errors_2025-07-10_17-35-52_quarter.csv'
    pinn_quarter_file = r'parameters\heat\table\error_pinn_2025-07-10_14-13-02_quarter.csv'
    flag = 'quarter'
    plot_errors(quarter_file, pinn_quarter_file, flag)