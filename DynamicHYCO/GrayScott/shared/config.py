"""
Configuration file for Gray-Scott PINN training parameters
"""

# Training parameters configuration
TRAINING_CONFIG = {
    # Data parameters
    'num_training_points': 50000,      # Number of data tuples to use for training
    'num_physics_points': 10000,      # Number of physics points for PDE loss
    
    # Network architecture
    'num_hidden': 128,                # Hidden layer size
    'num_layers': 4,                  # Number of hidden layers
    
    # Training parameters
    'epochs': 3000,                   # Number of training epochs
    'learning_rate': 1e-4,            # Learning rate for network weights
    'diffusion_learning_rate': 1e-4,  # Learning rate for diffusion parameters (Du, Dv)
    'weight_decay': 0,             # Weight decay for regularization
    'lambda_physics': 1.0,            # Weight for physics loss
    'lambda_ic': 100.0,               # Weight for initial condition loss

    # Parameter learning
    'learn_diffusion': False,         # Whether to learn diffusion parameters (Du, Dv only) - SET TO FALSE FOR FIXED PARAMS
    'init_Du': 0.10,                   # (Optional) initial Du if learn_diffusion True (overrides data params)
    'init_Dv': 0.05,                  # (Optional) initial Dv if learn_diffusion True
    'diffusion_warmup_epochs': 500,     # Number of epochs to freeze diffusion params before learning
    # PDE residual computed purely from autograd (FD blending removed)
    'lbfgs_lr': 1.0,                  # L-BFGS learning rate
    'lbfgs_mid_epochs': 200,            # L-BFGS epochs right after warmup unfreeze
    'lbfgs_epochs': 200,                # Final L-BFGS epochs after Adam phase
    'unfreeze_diffusion_after_warmup': True,   # If False, keep Du/Dv frozen for whole training
    'lbfgs_mid_keep_diffusion_frozen': True,  # If True and mid-LBFGS >0, run it with diffusion still frozen
    
    # Error tracking
    'error_check_interval': 50,       # How often to compute solution error (every N epochs)
    
    # Hardware
    'use_gpu': True                   # Use GPU if available

    
}

# Data and output paths
DATA_DIR = "data"                     # Directory containing training data
EXPERIMENT_NAME = "gray_scott_pinn_inverse"   # Name for this experiment
RESULTS_DIR = "results"               # Base directory for saving results

# Alternative configurations for different experiments
QUICK_TEST_CONFIG = {
    **TRAINING_CONFIG,
    'epochs': 500,
    'num_training_points': 1000,
    'num_physics_points': 2000,
    # Example override:
    # 'learn_diffusion': True,
    # 'init_Du': 0.18,
    # 'init_Dv': 0.07,
}

LARGE_SCALE_CONFIG = {
    **TRAINING_CONFIG,
    'epochs': 5000,
    'num_training_points': 20000,
    'num_physics_points': 50000,
    'num_hidden': 256,
    'num_layers': 6,
    # 'learn_diffusion': True,
    # 'init_Du': 0.2,
    # 'init_Dv': 0.08,
}
