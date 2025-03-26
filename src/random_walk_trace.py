import matplotlib.pyplot as plt
import numpy as np

def random_walk_2d(steps):
    """
    Generate a 2D random walk trajectory
    :param steps: Number of steps in the random walk
    :return: Tuple of x and y coordinates
    """
    x_step = np.random.choice([-1, 1], steps)
    y_step = np.random.choice([-1, 1], steps)
    return x_step.cumsum(), y_step.cumsum()

def plot_single_walk(path):
    """
    Plot a single random walk trajectory
    :param path: Tuple of x and y coordinates
    """
    x_coords, y_coords = path
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker='.', markersize=4, linestyle='-', color='b', label='Random Walk')
    plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')  # Start point
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End')   # End point
    plt.title('Single Random Walk Trajectory (1000 Steps)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_walks():
    """
    Plot four different random walk trajectories
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(4):
        path = random_walk_2d(1000)
        x_coords, y_coords = path
        
        axes[i].plot(x_coords, y_coords, marker='.', markersize=4, linestyle='-', color='b')
        axes[i].scatter(x_coords[0], y_coords[0], color='green', s=100)  # Start point
        axes[i].scatter(x_coords[-1], y_coords[-1], color='red', s=100)  # End point
        axes[i].set_title(f'Trajectory {i + 1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].axis('equal')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Task 1: Single random walk trajectory
    path = random_walk_2d(1000)
    plot_single_walk(path)
    
    # Task 2: Four different random walk trajectories
    plot_multiple_walks()
