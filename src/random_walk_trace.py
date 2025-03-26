import numpy as np
import matplotlib.pyplot as plt

def random_walk(steps):
    """
    生成二维对角随机行走轨迹
    :param steps: 步数
    :return: x坐标列表, y坐标列表
    """
    x = [0]
    y = [0]
    
    for _ in range(steps):
        direction = np.random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        x.append(x[-1] + direction[0])
        y.append(y[-1] + direction[1])
    
    return x, y

def plot_single_walk(x, y):
    """
    绘制单轨迹图
    """
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o', markersize=4, linestyle='-', color='b', label='Random Walk')
    plt.scatter(x[0], y[0], color='g', s=100, label='Start')  # 起点
    plt.scatter(x[-1], y[-1], color='r', s=100, label='End')  # 终点
    plt.legend()
    plt.title('2D Random Walk (1000 Steps)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_multiple_walks(num_walks, steps):
    """
    绘制多轨迹图
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Multiple 2D Random Walks (1000 Steps)', fontsize=16)
    
    for i in range(num_walks):
        x, y = random_walk(steps)
        ax = axs[i // 2, i % 2]
        ax.plot(x, y, marker='o', markersize=4, linestyle='-', color='b')
        ax.scatter(x[0], y[0], color='g', s=100)
        ax.scatter(x[-1], y[-1], color='r', s=100)
        ax.set_title(f'Walk {i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 单轨迹模拟
    x, y = random_walk(1000)
    plot_single_walk(x, y)
    
    # 多轨迹对比
    plot_multiple_walks(4, 1000)
