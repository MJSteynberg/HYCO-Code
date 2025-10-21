"""
Main plotting script for Gray-Scott model comparison and visualization.
"""
import os

# Import from organized shared modules
from shared.data_loader import load_true_data_at_times
from shared.model_loader import load_experiment_models, evaluate_pinn_model
from shared.visualization import create_comprehensive_comparison_plot


# --- MAIN FUNCTION FOR EXECUTION ---
def main():
    """Main function for loading and comparing trained models."""
    print("="*80)
    print("GRAY-SCOTT MODEL COMPARISON AND VISUALIZATION")
    print("="*80)

    # Configuration
    data_dir = 'results/data2'
    pinn_model_dir = 'results/learnable_diffusion'
    hybrid_model_dir = "results/consistency_comparison"
    target_times = [0, 500, 1000, 1500, 2000]

    # Load models
    print("STEP 1: Loading Trained Models")
    print("-" * 40)
    hybrid_models = load_experiment_models(hybrid_model_dir)
    pinn_predictions = evaluate_pinn_model(pinn_model_dir, data_dir, target_times)

    if not hybrid_models or not pinn_predictions:
        print("Failed to load one or more models. Exiting.")
        return

    # Load ground truth data
    print("\nSTEP 2: Loading Ground Truth Data")
    print("-" * 40)
    true_data = load_true_data_at_times(data_dir, target_times)

    # Create comprehensive comparison plot
    print("\nSTEP 3: Creating Comprehensive Comparison Plot")
    print("-" * 40)

    save_path = f"figures/comprehensive_comparison.png"

    create_comprehensive_comparison_plot(
        hybrid_models, pinn_predictions, true_data, target_times,
        data_dir, save_path
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Comprehensive comparison plot saved to: {save_path}")

if __name__ == "__main__":
    main()
