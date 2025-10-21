import matplotlib.pyplot as plt
import numpy as np


def plot_results_separate(u_real, u_train, u_test, dim=2, plot_type='Simulate'):
    # Initialize figure and axes based on the dimension
    fig = plt.figure(figsize=(12, 5), dpi=100)
    if dim == 2:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    elif dim == 3:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    z_min, z_max = 0, 1  # For 3D plots

    # Plot each trajectory in the training and testing datasets on the first subplot
    for i in range(u_train.shape[1]):
        if dim == 2:
            ax1.plot(u_train[:, i, 0], u_train[:, i, 1], 'r', linewidth=1.0)
        elif dim == 3:
            ax1.plot(u_train[:, i, 0], u_train[:, i, 1], u_train[:, i, 2], 'r', linewidth=1.0)

    # Legend for training dataset
    ax1.plot([], [], 'r', linewidth=1.0, label='Training Dataset')

    for i in range(u_test.shape[1]):
        if dim == 2:
            ax1.plot(u_test[:, i, 0], u_test[:, i, 1], '#32cd32', linewidth=1.0)
        elif dim == 3:
            ax1.plot(u_test[:, i, 0], u_test[:, i, 1], u_test[:, i, 2], '#32cd32', linewidth=1.0)

    # Legend for testing dataset
    ax1.plot([], [], '#32cd32', linewidth=1.0, label='Testing Dataset')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    if dim == 3:
        ax1.set_zlabel('z')
        ax1.set_zlim(z_min, z_max)
    ax1.set_title('SA-NODEs')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right')

    # Plot each trajectory in the exact dataset on the second subplot
    for i in range(u_real.shape[1]):
        if dim == 2:
            ax2.plot(u_real[:, i, 0], u_real[:, i, 1], 'b', linewidth=1.0)
        elif dim == 3:
            ax2.plot(u_real[:, i, 0], u_real[:, i, 1], u_real[:, i, 2], 'b', linewidth=1.0)

    # Plot a single point for the legend in exact solution
    if dim == 2:
        ax2.plot(0, 0, marker='o', markersize=1.0, color='b')
    elif dim == 3:
        ax2.plot([0], [0], [0], marker='o', markersize=1.0, color='b')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    if dim == 3:
        ax2.set_zlabel('z')
        ax2.set_zlim(z_min, z_max)
    ax2.set_title('Exact')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    # Save the plot
    plt.savefig(f'./figures/ODE_{plot_type}_{dim}D.png', bbox_inches='tight')
    plt.show()
