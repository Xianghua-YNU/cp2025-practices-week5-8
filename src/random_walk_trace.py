import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class RandomWalk2D:
    def __init__(self, steps=1000):
        """
        初始化二维随机行走模拟器
        
        参数:
            steps (int): 行走步数，默认为1000
        """
        self.steps = steps
        self.results_dir = Path("results")
        self._setup_directories()
        
    def _setup_directories(self):
        """创建结果目录"""
        self.results_dir.mkdir(exist_ok=True)
    
    def generate_walk(self):
        """
        生成二维随机行走轨迹
        
        返回:
            tuple: (x坐标数组, y坐标数组)
        """
        # 生成每一步的移动方向（对角移动）
        directions = np.random.randint(1, 5, size=self.steps)
        
        # 初始化位置数组
        x = np.zeros(self.steps + 1)
        y = np.zeros(self.steps + 1)
        
        # 模拟每一步移动
        for i in range(1, self.steps + 1):
            if directions[i-1] == 1:    # 右上
                x[i] = x[i-1] + 1
                y[i] = y[i-1] + 1
            elif directions[i-1] == 2:  # 右下
                x[i] = x[i-1] + 1
                y[i] = y[i-1] - 1
            elif directions[i-1] == 3:  # 左上
                x[i] = x[i-1] - 1
                y[i] = y[i-1] + 1
            elif directions[i-1] == 4:  # 左下
                x[i] = x[i-1] - 1
                y[i] = y[i-1] - 1
                
        return x, y
    
    def plot_single_walk(self, save_fig=True):
        """
        绘制单条随机行走轨迹
        
        参数:
            save_fig (bool): 是否保存图像，默认为True
        """
        x, y = self.generate_walk()
        
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, alpha=0.7, lw=1, label='Path')
        plt.scatter(x[0], y[0], c='green', s=150, label='Start (0,0)')
        plt.scatter(x[-1], y[-1], c='red', s=150, label=f'End ({x[-1]:.0f}, {y[-1]:.0f})')
        
        plt.title(f'2D Random Walk - {self.steps} Steps', fontsize=14)
        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_fig:
            save_path = self.results_dir / 'single_random_walk.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_multiple_walks(self, num_walks=4, save_fig=True):
        """
        绘制多条随机行走轨迹
        
        参数:
            num_walks (int): 轨迹数量，默认为4
            save_fig (bool): 是否保存图像，默认为True
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.ravel()
        
        for i in range(num_walks):
            x, y = self.generate_walk()
            
            axes[i].plot(x, y, alpha=0.7, lw=1)
            axes[i].scatter(x[0], y[0], c='green', s=100, label='Start')
            axes[i].scatter(x[-1], y[-1], c='red', s=100, label=f'End ({x[-1]:.0f}, {y[-1]:.0f})')
            
            axes[i].set_title(f'Random Walk {i+1}', fontsize=12)
            axes[i].set_xlabel('X Position', fontsize=10)
            axes[i].set_ylabel('Y Position', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].axis('equal')
            axes[i].legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = self.results_dir / 'multiple_random_walks.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 初始化随机行走模拟器
    rw_simulator = RandomWalk2D(steps=1000)
    
    print("Simulating single random walk...")
    rw_simulator.plot_single_walk()
    
    print("\nSimulating multiple random walks...")
    rw_simulator.plot_multiple_walks(num_walks=4)

    
    print("\nSimulating multiple random walks...")
    plot_multiple_walks(1000, 4)
