import numpy as np
import matplotlib.pyplot as plt
import random

def random_walk(steps):
    """
    模拟二维对角随机行走
    
    参数:
        steps (int): 行走步数
        
    返回:
        x, y (numpy arrays): 行走轨迹的x和y坐标
    """
    # 初始化位置数组
    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    
    # 从原点(0,0)出发
    x[0], y[0] = 0, 0
    
    # 模拟每一步移动
    for i in range(1, steps + 1):
        # 随机选择对角移动方向
        direction = random.choice([1, 2, 3, 4])
        
        if direction == 1:    # 右上
            x[i] = x[i-1] + 1
            y[i] = y[i-1] + 1
        elif direction == 2:  # 右下
            x[i] = x[i-1] + 1
            y[i] = y[i-1] - 1
        elif direction == 3:  # 左上
            x[i] = x[i-1] - 1
            y[i] = y[i-1] + 1
        elif direction == 4:  # 左下
            x[i] = x[i-1] - 1
            y[i] = y[i-1] - 1
    
    return x, y

def plot_single_walk(steps=1000):
    """
    绘制单条随机行走轨迹
    
    参数:
        steps (int): 行走步数
    """
    x, y = random_walk(steps)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, alpha=0.6, lw=1)
    
    # 标记起点和终点
    plt.scatter(x[0], y[0], c='green', s=100, label='Start (0,0)')
    plt.scatter(x[-1], y[-1], c='red', s=100, label=f'End ({x[-1]}, {y[-1]})')
    
    plt.title(f'2D Random Walk - {steps} Steps')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # 确保x和y轴比例相同
    
    # 保存图像
    plt.savefig('results/single_random_walk.png')
    plt.show()

def plot_multiple_walks(steps=1000, num_walks=4):
    """
    绘制多条随机行走轨迹
    
    参数:
        steps (int): 每条轨迹的步数
        num_walks (int): 轨迹数量
    """
    plt.figure(figsize=(10, 10))
    
    for i in range(num_walks):
        x, y = random_walk(steps)
        
        plt.subplot(2, 2, i+1)
        plt.plot(x, y, alpha=0.6, lw=1)
        plt.scatter(x[0], y[0], c='green', s=50)
        plt.scatter(x[-1], y[-1], c='red', s=50)
        plt.title(f'Random Walk {i+1}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('results/multiple_random_walks.png')
    plt.show()

if __name__ == "__main__":
    # 创建结果目录
    import os
    os.makedirs('results', exist_ok=True)
    
    print("Simulating single random walk...")
    plot_single_walk(1000)
    
    print("\nSimulating multiple random walks...")
    plot_multiple_walks(1000, 4)
