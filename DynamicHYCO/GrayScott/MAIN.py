# -*- coding: utf-8 -*-
"""
MAIN.py - HYCO Gray-Scott Model Training Script

This script trains hybrid physics-neural models for the Gray-Scott reaction-diffusion system
with different consistency weights for comparison.
"""

import os
import torch

# Import shared modules
from shared.data_generator import generate_data
from shared.training import hybrid_train


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function - trains HYCO models with consistency weights 0.0 and 1.0."""
    print_banner("GRAY-SCOTT HYCO CONSISTENCY COMPARISON TRAINING")
    
    # =============================================================================
    # CONFIGURATION - Data Generation Parameters
    # =============================================================================
    data_params = {
        'width': 64,
        'height': 64,
        'domain_width': 100.0,
        'domain_height': 100.0,
        'steps': 5000,
        'save_interval': 100,
        'Du': 0.20,
        'Dv': 0.08,
        'f': 0.018,
        'k': 0.051,
        'use_gpu': torch.cuda.is_available(),
        'final_time': 2000.0,
    }
    
    # =============================================================================
    # CONFIGURATION - Training Parameters
    # =============================================================================
    # Base training parameters
    base_training_params = {
        'pre_epochs': 200,
        'epochs': 200,
        'post_epochs': 200,
        'batch_size': 1000,
        'physical_lr': 0.008,
        'neural_lr': 0.001,
        'num_training': 5000,
        'use_lbfgs_post': True,
        'neural_params': {
            'hidden_size': 128,
            'num_layers': 4
        },
        'loss_weights': {
            'neural_data_weight': 1.0,
            'physical_data_weight': 0.0,
            'physical_consistency_weight': 1.0
        },
        'params': {
            'width': 64,
            'height': 64,
            'domain_width': 100.0,
            'domain_height': 100.0,
            'initial_Du': 0.10,
            'initial_Dv': 0.05
        },
        'save_models': True
    }
    
    # =============================================================================
    # CONFIGURATION - Directories
    # =============================================================================
    data_dir = 'data'
    base_save_dir = os.path.join('results', 'consistency_comparison')
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # =============================================================================
    # STEP 1: Generate Training Data
    # =============================================================================
    print_banner("STEP 1: DATA GENERATION")
    
    print(f"\nData Parameters:")
    print(f"  Resolution: {data_params['width']}x{data_params['height']}")
    print(f"  Domain: {data_params['domain_width']}x{data_params['domain_height']}")
    print(f"  Time steps: {data_params['steps']}, Final time: {data_params['final_time']}")
    print(f"  Du={data_params['Du']}, Dv={data_params['Dv']}, f={data_params['f']}, k={data_params['k']}")
    
    # Generate data (will use existing if parameters match)
    generated_data_dir = generate_data(data_params, data_dir, force_regenerate=False)
    
    if not generated_data_dir:
        print("\nERROR: Data generation failed! Aborting.")
        return
    
    print(f"\nData ready in: {generated_data_dir}")
    
    # =============================================================================
    # STEP 2: Train Models with Different Consistency Weights
    # =============================================================================
    print_banner("STEP 2: CONSISTENCY COMPARISON TRAINING")
    
    # Define experiments (matching Colab script exactly)
    experiments = [
        {
            'name': 'consistency_1.0',
            'description': 'Neural consistency weight = 0.2',
            'neural_consistency_weight': 0.2
        },
        {
            'name': 'consistency_0.0',
            'description': 'Neural consistency weight = 0.0',
            'neural_consistency_weight': 0.0
        }
    ]
    
    print(f"\nBase Training Configuration:")
    print(f"  Pre-training epochs: {base_training_params['pre_epochs']}")
    print(f"  Main training epochs: {base_training_params['epochs']}")
    print(f"  Post-training epochs: {base_training_params['post_epochs']}")
    print(f"  Batch size: {base_training_params['batch_size']}")
    print(f"  Training points: {base_training_params['num_training']}")
    print(f"  Neural network: {base_training_params['neural_params']['hidden_size']}x{base_training_params['neural_params']['num_layers']}")
    print(f"  Initial Du: {base_training_params['params']['initial_Du']}, Dv: {base_training_params['params']['initial_Dv']}")
    
    # Run experiments
    for i, experiment in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/2: {experiment['description']}")
        print(f"{'='*80}")
        
        # Create experiment-specific parameters
        training_params = base_training_params.copy()
        training_params['loss_weights'] = base_training_params['loss_weights'].copy()
        training_params['loss_weights']['neural_consistency_weight'] = experiment['neural_consistency_weight']
        training_params['save_dir'] = os.path.join(base_save_dir, experiment['name'])
        
        print(f"\nConfiguration:")
        print(f"  Neural consistency weight: {experiment['neural_consistency_weight']}")
        print(f"  Save directory: {training_params['save_dir']}")
        
        # Run training
        try:
            result = hybrid_train(data_dir, **training_params)
            
            if len(result) == 4:
                physical_model, neural_model, model_save_path, loss_history = result
                if model_save_path:
                    print(f"\n✓ Models saved to: {model_save_path}")
            else:
                physical_model, neural_model, loss_history = result
            
            if not physical_model or not neural_model:
                print(f"\n✗ ERROR: Training failed for {experiment['name']}!")
                continue
            
            print(f"\n✓ Training completed successfully for {experiment['name']}!")
            final_params = physical_model.get_trainable_parameters()
            print(f"  Final physical parameters: Du={final_params['Du']:.6f}, Dv={final_params['Dv']:.6f}")
            
        except Exception as e:
            print(f"\n✗ ERROR during training for {experiment['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =============================================================================
    # COMPLETION
    # =============================================================================
    print_banner("TRAINING COMPLETED SUCCESSFULLY")
    print(f"\nResults saved in: {base_save_dir}")
    print(f"  - consistency_0.0/")
    print(f"  - consistency_1.0/")
    print(f"\nTo visualize results, run PLOT.py")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

