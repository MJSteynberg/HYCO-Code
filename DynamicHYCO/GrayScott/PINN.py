"""
Experiment 2: Gray-Scott PINN with Learnable Diffusion Coefficients
This experiment learns both the solution and the diffusion parameters (Du, Dv).
"""

import os
import sys
from shared.config import TRAINING_CONFIG
from shared.pinn import train_pinn

def run_learnable_diffusion_experiment():
    """Run PINN training with learnable diffusion coefficients."""
    
    print("="*60)
    print("EXPERIMENT 2: LEARNABLE DIFFUSION COEFFICIENTS")
    print("="*60)
    
    # Configuration for learnable diffusion experiment
    config = {
        **TRAINING_CONFIG,
        
        # Key difference: Learn diffusion parameters
        'learn_diffusion': True,
        
        # Initial guesses for diffusion parameters (will be optimized)
        'init_Du': 0.10,  # Start with a slightly different value than true
        'init_Dv': 0.05,  # Start with a slightly different value than true
        
        # Training parameters optimized for parameter learning
        'epochs': 3000,  # More epochs needed for parameter learning
        'learning_rate': 1e-3,  # Network learning rate
        'diffusion_learning_rate': 1e-3,  # Slower learning for diffusion params
        'num_training_points': 5000,  # More data points for parameter learning
        'num_physics_points': 50000,  # More physics points for better regularization
        
        # Network architecture
        'num_hidden': 128,
        'num_layers': 4,
        
        # Loss weights 
        'lambda_physics': 1.0,
        'lambda_ic': 100.0,
        'lambda_bc': 0.0,
        
        # Warmup strategy for diffusion learning
        'diffusion_warmup_epochs': 0,  # Freeze diffusion params initially
        'unfreeze_diffusion_after_warmup': True,
        'disable_physics_during_warmup': True,  # Focus on data fitting first
        
        # L-BFGS fine-tuning strategy
        'lbfgs_mid_epochs': 0,  # Mid-training L-BFGS
        'lbfgs_mid_keep_diffusion_frozen': True,  
        'lbfgs_epochs': 500,  # Final L-BFGS
        'lbfgs_lr': 1,  # Conservative L-BFGS learning rate
        
        # Regularization
        'weight_decay': 0,  # Slightly stronger regularization
        
        # Hardware
        'use_gpu': True,
        
        # Error tracking
        'error_check_interval': 50,
        
        # Additional options for parameter learning
        'compute_data_loss_physical': True,  # Compute data loss in physical space
    }
    
    # Data and save directories
    data_dir = "data"
    experiment_name = "learnable_diffusion"
    base_save_dir = "results"
    
    print(f"Configuration:")
    print(f"  Learning diffusion: {config['learn_diffusion']}")
    print(f"  Initial Du: {config['init_Du']}")
    print(f"  Initial Dv: {config['init_Dv']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Network learning rate: {config['learning_rate']}")
    print(f"  Diffusion learning rate: {config['diffusion_learning_rate']}")
    print(f"  Training points: {config['num_training_points']}")
    print(f"  Physics points: {config['num_physics_points']}")
    print(f"  Network: {config['num_layers']} layers, {config['num_hidden']} hidden units")
    print(f"  Warmup epochs: {config['diffusion_warmup_epochs']}")
    print(f"  Save directory: {os.path.join(base_save_dir, experiment_name)}")
    print()
    
    # Run training
    print("Starting training...")
    model_path = train_pinn(data_dir, config, experiment_name, base_save_dir)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results directory: {os.path.join(base_save_dir, experiment_name)}")
    
    # Load and display final learned parameters
    try:
        import numpy as np
        results_dir = os.path.join(base_save_dir, experiment_name)
        
        # Load true parameters
        true_params = np.load("data/parameters.npy", allow_pickle=True).item()
        true_Du = true_params.get('Du', true_params.get('D_u', 'Unknown'))
        true_Dv = true_params.get('Dv', true_params.get('D_v', 'Unknown'))
        
        # Load learned parameters evolution
        Du_evolution = np.load(os.path.join(results_dir, "Du_evolution.npy"))
        Dv_evolution = np.load(os.path.join(results_dir, "Dv_evolution.npy"))
        
        learned_Du = Du_evolution[-1]
        learned_Dv = Dv_evolution[-1]
        
        print("\n" + "="*50)
        print("PARAMETER LEARNING RESULTS")
        print("="*50)
        print(f"True Du:    {true_Du}")
        print(f"Learned Du: {learned_Du:.6f}")
        print(f"Error Du:   {abs(float(true_Du) - learned_Du):.6f} ({abs(float(true_Du) - learned_Du)/float(true_Du)*100:.2f}%)")
        print()
        print(f"True Dv:    {true_Dv}")
        print(f"Learned Dv: {learned_Dv:.6f}")
        print(f"Error Dv:   {abs(float(true_Dv) - learned_Dv):.6f} ({abs(float(true_Dv) - learned_Dv)/float(true_Dv)*100:.2f}%)")
        print("="*50)
        
    except Exception as e:
        print(f"Could not load parameter evolution: {e}")
    
    return os.path.join(base_save_dir, experiment_name)

if __name__ == "__main__":
    # Run the learnable diffusion experiment
    results_dir = run_learnable_diffusion_experiment()
