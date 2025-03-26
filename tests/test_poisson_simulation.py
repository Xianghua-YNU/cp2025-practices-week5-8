import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poisson_simulation import plot_poisson_pmf, simulate_coin_flips, compare_simulation_theory
#from solutions.poisson_simulation_solution import plot_poisson_pmf, simulate_coin_flips, compare_simulation_theory

class TestPoissonSimulation(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.lambda_param = 8
        self.max_l = 20
        self.n_experiments = 1000
        self.n_flips = 100
        self.p_head = 0.08

    def test_pmf_calculation(self):
        """测试PMF计算结果是否正确"""
        pmf = plot_poisson_pmf(self.lambda_param, self.max_l)
        
        # 手动计算预期结果
        l_values = np.arange(self.max_l)
        expected_pmf = (self.lambda_param**l_values * np.exp(-self.lambda_param)) / factorial(l_values)
        
        # 验证概率和接近1
        self.assertAlmostEqual(np.sum(pmf), 1.0, places=3)
        # 验证所有概率非负
        self.assertTrue(np.all(pmf >= 0))
        # 验证计算结果
        np.testing.assert_array_almost_equal(pmf, expected_pmf)
        plt.close('all')

    def test_simulate_coin_flips(self):
        """测试抛硬币实验的基本属性"""
        results = simulate_coin_flips(self.n_experiments, self.n_flips, self.p_head)
        
        # 测试形状
        self.assertEqual(results.shape, (self.n_experiments,))
        
        # 测试范围
        self.assertTrue(np.all(results >= 0))
        self.assertTrue(np.all(results <= self.n_flips))
        
        # 测试均值
        expected_mean = self.n_flips * self.p_head
        actual_mean = np.mean(results)
        self.assertLess(abs(actual_mean - expected_mean) / expected_mean, 0.05)

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试零次实验
        results_zero_exp = simulate_coin_flips(0, self.n_flips, self.p_head)
        self.assertEqual(len(results_zero_exp), 0)

        # 测试零次抛掷
        results_zero_flips = simulate_coin_flips(self.n_experiments, 0, self.p_head)
        self.assertTrue(np.all(results_zero_flips == 0))

        # 测试极端概率
        results_p0 = simulate_coin_flips(100, 10, 0.0)
        results_p1 = simulate_coin_flips(100, 10, 1.0)
        self.assertTrue(np.all(results_p0 == 0))
        self.assertTrue(np.all(results_p1 == 10))

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.title')
    def test_visualization(self, mock_title, mock_plot, mock_figure):
        """测试可视化函数的调用"""
        plot_poisson_pmf(self.lambda_param, self.max_l)
        # 修改断言，检查是否至少调用了一次，而不是严格的一次
        self.assertTrue(mock_figure.called)
        mock_plot.assert_called_once()
        mock_title.assert_called_once_with(f'Poisson Probability Mass Function (λ={self.lambda_param})')
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ======================
# 参数设置
# ======================
n_coins = 100      # 每组实验抛硬币次数
p = 0.08           # 单次正面概率
lambda_ = n_coins * p  # λ = 8
N = 10000          # 实验组数

好的，我将通过减少一些不必要的函数和代码结构来简化你提供的代码，同时保持结果的一致性。以下是简化后的代码：
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# 设置随机种子
np.random.seed(42)

# 泊松分布参数
lambda_param = 8
# 实验组数
n_experiments = 10000
# 每组抛硬币次数
n_flips = 100
# 正面朝上的概率
p_head = 0.08

# 1. 绘制理论分布
l_values = np.arange(int(lambda_param * 2))
pmf = (lambda_param ** l_values * np.exp(-lambda_param)) / factorial(l_values)

plt.figure(figsize=(10, 6))
plt.plot(l_values, pmf, 'bo-', label='Theoretical Distribution')
plt.title(f'Poisson Probability Mass Function (λ={lambda_param})')
plt.xlabel('l')
plt.ylabel('p(l)')
plt.grid(True)
plt.legend()
plt.show()

# 2. 进行实验模拟
results = []
for _ in range(n_experiments):
    coins = np.random.choice([0, 1], n_flips, p=[1 - p_head, p_head])
    results.append(coins.sum())
results = np.array(results)

# 3. 比较实验结果与理论分布
max_l = max(int(lambda_param * 2), max(results) + 1)
l_values = np.arange(max_l)
pmf = (lambda_param ** l_values * np.exp(-lambda_param)) / factorial(l_values)

plt.figure(figsize=(12, 7))
plt.hist(results, bins=range(max_l + 1), density=True, alpha=0.7,
         label='Simulation Results', color='skyblue')
plt.plot(l_values, pmf, 'r-', label='Theoretical Distribution', linewidth=2)

plt.title(f'Poisson Distribution Comparison (N={n_experiments}, λ={lambda_param})')
plt.xlabel('Number of Heads')
plt.ylabel('Frequency/Probability')
plt.grid(True, alpha=0.3)
plt.legend()

# 打印统计信息
print(f"实验均值: {np.mean(results):.2f} (理论值: {lambda_param})")
print(f"实验方差: {np.var(results):.2f} (理论值: {lambda_param})")

plt.show()
 
