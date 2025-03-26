import numpy as np
import matplotlib.pyplot as plt

def random_walk_finals(num_steps=1000, num_walks=1000):
    """
    Generate final positions of multiple 2D random walks
    
    Simulate multiple random walks, each with a specified number of steps,
    and calculate the final positions in both x and y directions.

    Parameters:
        num_steps (int, optional): Number of steps in each random walk. Default is 1000.
        num_walks (int, optional): Number of random walks to simulate. Default is 1000.
        
    Returns:
        tuple: A tuple containing two numpy arrays (x_finals, y_finals)
            - x_finals: Array of final x positions for all random walks
            - y_finals: Array of final y positions for all random walks
    """
    x_finals = np.zeros(num_walks)
    y_finals = np.zeros(num_walks)
    for i in range(num_walks):
        x_finals[i] = np.sum(np.random.choice([-1, 1], num_steps))
        y_finals[i] = np.sum(np.random.choice([-1, 1], num_steps))
    return x_finals, y_finals

def calculate_mean_square_displacement():
    """
    Calculate the mean square displacement for different numbers of steps
    
    For a predefined sequence of step numbers [1000, 2000, 3000, 4000],
    simulate multiple random walks and calculate the mean square displacement
    for each number of steps. Default number of walks per simulation is 1000.
    
    Returns:
        tuple: A tuple containing two numpy arrays (steps, msd)
            - steps: Array of step numbers [1000, 2000, 3000, 4000]
            - msd: Corresponding array of mean square displacements
    """
    steps = np.array([1000, 2000, 3000, 4000])
    msd = []
    
    for i in steps:
        x_finals, y_finals = random_walk_finals(num_steps=i)
        ds = x_finals**2 + y_finals**2
        msd.append(np.mean(ds))
    
    return steps, np.array(msd)

def analyze_step_dependence():
    """
    Analyze the relationship between mean square displacement and number of steps,
    and perform a least-squares fit
    
    Returns:
        tuple: (steps, msd, k)
            - steps: Array of step numbers
            - msd: Corresponding array of mean square displacements
            - k: Fitted proportionality constant
    """
    steps, msd = calculate_mean_square_displacement()
    
    # Least-squares fit (forcing through the origin)
    # Theoretically, msd = k * steps, where k should be close to 2
    k = np.sum(steps * msd) / np.sum(steps**2)
    
    return steps, msd, k

if __name__ == "__main__":
    steps, msd, k = analyze_step_dependence()
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, msd, 'ro', ms=10, label='Experimental Data')
    plt.plot(steps, k * steps, 'g--', label=f'Fitted: $r^2={k:.2f}N$', lw=2)
    plt.plot(steps, 2 * steps, 'b-', label='Theory: $r^2=2N$', lw=2)
    
    plt.xlabel('Number of Steps $N$', fontsize=14)
    plt.ylabel('Mean Square Displacement $\\langle r^2 \\rangle$', fontsize=14)
    plt.title('Relationship between Steps and Mean Square Displacement', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    print("步数和对应的均方位移：")
    for n, m in zip(steps, msd):
        print(f"步数: {n:5d}, 均方位移: {m:.2f}")
    
    print(f"\n拟合结果：r² = {k:.4f}N")
    print(f"与理论值k=2的相对误差: {abs(k - 2) / 2 * 100:.2f}%")
    
    plt.show()
