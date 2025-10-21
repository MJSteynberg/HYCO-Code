# HYCO: Hybrid Consistency Models for Parameter Estimation in PDEs

A comprehensive framework for learning physical parameters and solutions of partial differential equations (PDEs) using hybrid physics-neural network models. This repository implements both **static** (steady-state) and **dynamic** (time-dependent) PDE solvers with various training strategies including Physics-Informed Neural Networks (PINNs), Finite Element Method (FEM), and novel hybrid consistency approaches.

## 🔬 Overview

HYCO (Hybrid Consistency) combines physics-based models with neural networks to solve inverse problems in PDEs. The key innovation is the **hybrid consistency loss** that enforces agreement between physics-based and data-driven components, enabling robust parameter estimation even with sparse or noisy data.

### Key Features

- **Multiple PDE Systems**: Darcy flow, Helmholtz equation, heat equation, Gray-Scott reaction-diffusion
- **Hybrid Training Strategies**: Seamlessly blend physics constraints with data-driven learning
- **Parameter Learning**: Automatically infer unknown physical parameters (diffusion coefficients, reaction rates, etc.)
- **Flexible Architectures**: JAX/Flax for static problems, PyTorch for dynamic systems
- **Comprehensive Benchmarking**: Compare PINN, FEM, and hybrid approaches

## 📁 Repository Structure

```
HYCO-Code/
├── StaticHYCO/              # Static (steady-state) PDE problems
│   └── src/
│       ├── darcy.py         # Darcy flow inverse problem
│       ├── helmholtz.py     # Helmholtz equation with Gaussian coefficients
│       ├── models/          # Model architectures
│       │   ├── physical_model.py    # FEM-based physics solver
│       │   ├── synthetic_model.py   # Neural network architectures
│       │   └── other_models.py      # PINN implementations
│       ├── tools/           # Utilities
│       │   ├── training.py          # Training loops (Hybrid, FEM, PINN)
│       │   ├── finite_element_method.py  # FEM solver
│       │   ├── plotting.py          # Visualization tools
│       │   └── experiment_utils.py  # Helper functions
│       ├── files/           # Saved results and intermediate data
│       ├── results/         # Generated plots and figures
│       └── cache/           # Cached solutions for reproducibility
│
├── DynamicHYCO/             # Dynamic (time-dependent) PDE problems
│   ├── GrayScott/           # Gray-Scott reaction-diffusion system
│   │   ├── MAIN.py          # Main training script for hybrid models
│   │   ├── PINN.py          # PINN with learnable diffusion coefficients
│   │   ├── PLOT.py          # Visualization script
│   │   ├── shared/          # Shared modules
│   │   │   ├── model.py             # Gray-Scott physics model
│   │   │   ├── pinn.py              # PINN architecture and trainer
│   │   │   ├── training.py          # Hybrid training loop
│   │   │   ├── data_generator.py    # Synthetic data generation
│   │   │   └── visualization.py     # Plotting utilities
│   │   ├── data/            # Training data
│   │   └── results/         # Trained models and metrics
│   │
│   └── Heat/                # Heat equation parameter estimation
│       ├── MAIN.py          # Main training script
│       ├── PINN.py          # PINN implementation for heat equation
│       ├── PLOT_COEFFICIENTS.py  # Plot learned coefficients
│       ├── PLOT_ERRORS.py        # Error analysis
│       ├── models/          # Model implementations
│       │   ├── FEM.py              # FEM solver for heat equation
│       │   ├── NN.py               # Neural network architectures
│       │   ├── training.py         # Training utilities
│       │   └── generate_data.py    # Data generation
│       ├── data/            # Training datasets
│       └── parameters/      # Learned parameters and errors
│
└── .gitignore               # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

The project uses different frameworks for static and dynamic problems:

**For StaticHYCO (JAX-based):**
```bash
pip install jax jaxlib flax optax numpy scipy matplotlib
```

**For DynamicHYCO (PyTorch-based):**
```bash
pip install torch numpy scipy matplotlib pandas
```

**Optional (for GPU acceleration):**
- JAX: Follow [JAX GPU installation guide](https://github.com/google/jax#installation)
- PyTorch: Install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/)

### Quick Start

#### 1. Static Problems (Darcy Flow / Helmholtz)

Run the Darcy flow inverse problem with different noise levels:

```bash
cd StaticHYCO/src
python darcy.py
```

Run the Helmholtz equation experiment:

```bash
cd StaticHYCO/src
python helmholtz.py
```

**Key Parameters to Modify:**
- `noise_level`: Amount of observation noise (0.0, 0.05, 0.1, etc.)
- `config.epochs`: Number of training epochs
- `config.n_train`: Number of training observations
- `config.subdomain`: Region where observations are available

#### 2. Dynamic Problems (Gray-Scott)

Train hybrid models with different consistency weights:

```bash
cd DynamicHYCO/GrayScott
python MAIN.py
```

Train PINN with learnable diffusion coefficients:

```bash
cd DynamicHYCO/GrayScott
python PINN.py
```

Visualize results:

```bash
python PLOT.py
```

#### 3. Heat Equation Parameter Estimation

Train on different data fractions (full, half, quarter):

```bash
cd DynamicHYCO/Heat
python MAIN.py
```

## 🧪 Experiments

### StaticHYCO: Steady-State Inverse Problems

**Darcy Flow Problem:**
- **PDE**: `-∇·(κ(x,y)∇u) + η(x,y)u = f(x,y)`
- **Goal**: Learn spatially-varying diffusion `κ` and reaction `η` coefficients
- **Parameters**: 10 coefficients controlling Fourier series expansion
- **Methods**: Hybrid (physics + neural), FEM (physics-only), PINN (neural + physics loss)

**Helmholtz Equation:**
- **PDE**: `-∇·(κ(x,y)∇u) + η(x,y)u = f(x,y)`
- **Coefficients**: Gaussian-shaped `κ(x,y)` and `η(x,y)` 
- **Parameters**: 6 parameters (amplitude, centers for each coefficient)
- **Special Feature**: Subdomain observations (only 25% of domain observed)

### DynamicHYCO: Time-Dependent Systems

**Gray-Scott Reaction-Diffusion:**
- **PDEs**: 
  - `∂u/∂t = Du∇²u - uv² + f(1-u)`
  - `∂v/∂t = Dv∇²v + uv² - (f+k)v`
- **Goal**: Learn diffusion coefficients `Du` and `Dv`
- **Features**: 
  - Pattern formation dynamics
  - Consistency-weighted hybrid training
  - Comparison of pure data-driven vs. physics-constrained models

**Heat Equation:**
- **PDE**: `∂u/∂t = ∇·(κ(x,y)∇u)`
- **Goal**: Learn spatially-varying thermal conductivity `κ(x,y)`
- **Experiments**: Training with full, half, and quarter datasets

## 📊 Results and Outputs

### Generated Files

- **`results/`**: Training metrics, loss curves, parameter evolution
- **`figures/`**: Comparison plots, solution visualizations, error heatmaps
- **`data/`**: Generated training data, cached solutions
- **`models/` or `parameters/`**: Saved model weights and learned parameters

### Typical Outputs

1. **Solution Comparisons**: Ground truth vs. learned solutions
2. **Parameter Evolution**: How learned parameters converge during training
3. **Error Analysis**: L2 errors, pointwise errors, parameter errors
4. **Consistency Plots**: Agreement between physics and neural components

## 🔧 Configuration

### StaticHYCO Configuration

Edit configuration classes in `darcy.py` or `helmholtz.py`:

```python
class Config:
    epochs = 2500              # Training epochs
    n_train = 50              # Number of training points
    syn_lr = 5e-3             # Neural network learning rate
    phys_lr = 5e-3            # Physics model learning rate
    hidden_dims = (256, 256)  # Neural network architecture
    noise_level = 0.0         # Observation noise level
```

### DynamicHYCO Configuration

Edit training parameters in scripts or `shared/config.py`:

```python
training_params = {
    'pre_epochs': 200,         # Pre-training epochs
    'epochs': 200,             # Main training epochs
    'post_epochs': 200,        # Fine-tuning epochs
    'batch_size': 1000,
    'physical_lr': 0.008,      # Physics parameters learning rate
    'neural_lr': 0.001,        # Neural network learning rate
    'physical_consistency_weight': 1.0,  # Consistency loss weight
}
```

## 📖 Key Concepts

### Hybrid Consistency Training

The hybrid model combines:
1. **Physics Model**: FEM or differential equation solver with learnable parameters
2. **Neural Model**: Flexible neural network for residual/correction
3. **Consistency Loss**: Enforces agreement between physics and neural predictions

**Loss Function:**
```
L = λ_data * L_data + λ_physics * L_physics + λ_consistency * L_consistency
```

Where:
- `L_data`: Fit to observations
- `L_physics`: Satisfy PDE constraints (for PINN)
- `L_consistency`: Agreement between physics and neural models

### Training Strategies

1. **FEM-Only**: Pure physics-based optimization (no neural network)
2. **PINN**: Neural network with physics-informed loss
3. **Hybrid**: Combined physics + neural with consistency enforcement

### Advantages of Hybrid Approach

- **Robustness**: Better generalization with limited data
- **Interpretability**: Learned physical parameters have clear meaning
- **Efficiency**: Physics constraints guide neural network training
- **Flexibility**: Can handle model misspecification via neural correction

## 📚 Publications

This code implements methods from research on hybrid physics-neural models for inverse problems. Key concepts include:

- Physics-informed machine learning
- Parameter estimation in PDEs
- Hybrid modeling strategies
- Finite element methods with neural networks

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional PDE systems
- New neural architectures
- Optimization strategies
- Documentation and examples

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

This project uses:
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [Flax](https://github.com/google/flax) - Neural network library for JAX
- [PyTorch](https://pytorch.org/) - Deep learning framework
- Finite element methods for physics-based solvers

## 📧 Contact

[Add your contact information here]

---

**Note**: This is research code. While we strive for correctness and reproducibility, some experimental features may require tuning for your specific use case.
