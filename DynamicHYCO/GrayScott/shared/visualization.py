"""
Visualization utilities for Gray-Scott model comparisons.
Contains all plotting and visualization functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def create_comprehensive_comparison_plot(
    hybrid_models, 
    pinn_predictions, 
    true_data, 
    target_times, 
    data_dir, 
    save_path="figures/comprehensive_comparison.png"
):
    """
    Create comprehensive comparison plot with predictions, with enhanced formatting.
    
    Args:
        hybrid_models: Dictionary containing consistency_0 and consistency_1 models
        pinn_predictions: Dictionary of PINN predictions at each time point
        true_data: Dictionary of ground truth data at each time point
        target_times: List of time points to visualize
        data_dir: Directory containing data
        save_path: Path to save the output figure
    """
    from shared.data_loader import evaluate_hybrid_models
    
    print("Creating comprehensive comparison plot...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Determine the number of models we have
    models_available = []
    if hybrid_models and 'consistency_1' in hybrid_models:
        models_available.extend(['HYCO Physical', 'HYCO Synthetic'])
    if pinn_predictions:
        models_available.append('PINN')
    if hybrid_models and 'consistency_0' in hybrid_models:
        models_available.append('NN')
    models_available.append('True')

    num_models = len(models_available)

    # Create the figure and subplots
    fig, axes = plt.subplots(
        num_models, len(target_times), 
        figsize=(6 * len(target_times), 5 * num_models), 
        squeeze=False
    )

    # Determine global vmin/vmax for consistent coloring across all plots
    global_u_min, global_u_max = 0, 1
    cmap = 'viridis_r'

    # Evaluate hybrid models if available
    predictions_c1 = None
    predictions_c0 = None
    
    if hybrid_models and 'consistency_1' in hybrid_models:
        predictions_c1 = evaluate_hybrid_models(
            hybrid_models['consistency_1']['physical'],
            hybrid_models['consistency_1']['neural'],
            data_dir, target_times
        )
    
    if hybrid_models and 'consistency_0' in hybrid_models:
        predictions_c0 = evaluate_hybrid_models(
            hybrid_models['consistency_0']['physical'],
            hybrid_models['consistency_0']['neural'],
            data_dir, target_times
        )

    # Plot predictions for each time point
    for col, time in enumerate(target_times):
        axes[0, col].set_title(f't = {time}', fontsize=20, fontweight='bold')

        row = 0
        
        # HYCO Physical (if available)
        if predictions_c1:
            im = axes[row, col].contourf(
                true_data[time]['x_coords'], 
                true_data[time]['y_coords'], 
                predictions_c1['physical'][time]['u'],
                levels=20, cmap=cmap, vmin=global_u_min, vmax=global_u_max
            )
            if col == 0:
                axes[row, col].set_ylabel(
                    'HYCO Physical', fontsize=18, rotation=90, 
                    ha='center', labelpad=20, fontweight='bold'
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            row += 1

            # HYCO Synthetic
            im = axes[row, col].contourf(
                true_data[time]['x_coords'], 
                true_data[time]['y_coords'], 
                predictions_c1['neural'][time]['u'],
                levels=20, cmap=cmap, vmin=global_u_min, vmax=global_u_max
            )
            if col == 0:
                axes[row, col].set_ylabel(
                    'HYCO Synthetic', fontsize=18, rotation=90, 
                    ha='center', labelpad=20, fontweight='bold'
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            row += 1

        # PINN Model (if available)
        if pinn_predictions:
            im = axes[row, col].contourf(
                true_data[time]['x_coords'], 
                true_data[time]['y_coords'], 
                pinn_predictions[time]['u'],
                levels=20, cmap=cmap, vmin=global_u_min, vmax=global_u_max
            )
            if col == 0:
                axes[row, col].set_ylabel(
                    'PINN', fontsize=18, rotation=90, 
                    ha='center', labelpad=20, fontweight='bold'
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            row += 1

        # NN (if available)
        if predictions_c0:
            im = axes[row, col].contourf(
                true_data[time]['x_coords'], 
                true_data[time]['y_coords'], 
                predictions_c0['neural'][time]['u'],
                levels=20, cmap=cmap, vmin=global_u_min, vmax=global_u_max
            )
            if col == 0:
                axes[row, col].set_ylabel(
                    'NN', fontsize=18, rotation=90, 
                    ha='center', labelpad=20, fontweight='bold'
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            row += 1

        # True
        im = axes[row, col].contourf(
            true_data[time]['x_coords'], 
            true_data[time]['y_coords'], 
            true_data[time]['u'],
            levels=20, cmap=cmap, vmin=global_u_min, vmax=global_u_max
        )
        if col == 0:
            axes[row, col].set_ylabel(
                'True', fontsize=18, rotation=90, 
                ha='center', labelpad=20, fontweight='bold'
            )
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    # Add a single colorbar for the entire plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar_ax.tick_params(labelsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.suptitle('Gray-Scott Model Comparison', fontsize=24, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {save_path}")
