import numpy as np
import matplotlib.pyplot as plt

def random_walk_2d(steps):
    """
    Generate a 2D random walk trajectory
    :param steps: Number of steps in the random walk
    :return: Tuple of x and y coordinates
    """
    x_step = np.random.choice([-1, 1], steps)
    y_step = np.random.choice([-1, 1], steps)
    x_coords = x_step.cumsum()
    y_coords = y_step.cumsum()
    return x_coords, y_coords

def calculate_msd(steps, num_walks):
    """
    Calculate the mean square displacement for a given number of steps
    :param steps: Number of steps in the random walk
    :param num_walks: Number of random walks to average over
    :return: Mean square displacement
    """
    msd_values = []
    for _ in range(num_walks):
        x_coords, y_coords = random_walk_2d(steps)
        r_squared = x_coords[-1]**2 + y_coords[-1]**2
        msd_values.append(r_squared)
    return np.mean(msd_values)

def analyze_msd():
    """
    Analyze the relationship between mean square displacement and number of steps
    """
    steps_list = [1000, 2000, 3000, 4000]
    msd_list = []
    num_walks = 1000

    for steps in steps_list:
        msd = calculate_msd(steps, num_walks)
        msd_list.append(msd)
        print(f"Steps: {steps}, Mean Square Displacement: {msd}")

    # Plotting the relationship
    plt.figure(figsize=(8, 6))
    plt.plot(steps_list, msd_list, marker='o', linestyle='-', color='b')
    plt.title('Mean Square Displacement vs. Number of Steps')
    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Square Displacement')
    plt.grid(True)
    plt.show()

    # Fitting the data to a linear relationship
    coefficients = np.polyfit(steps_list, msd_list, 1)
    slope, intercept = coefficients
    print(f"Fitted linear relationship: MSD = {slope:.2f} * Steps + {intercept:.2f}")

if __name__ == "__main__":
    analyze_msd()
    # 3. 设置图形属性
    # 4. 打印数据分析结果
    pass
