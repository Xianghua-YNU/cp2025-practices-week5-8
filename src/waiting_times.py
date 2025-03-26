import numpy as np
import matplotlib.pyplot as plt

def generate_coin_sequence(n_flips, p_head=0.08):
    """生成硬币序列，1表示正面，0表示反面
    
    这个函数模拟抛硬币实验，生成一个由0和1组成的随机序列。
    
    参数:
        n_flips (int): 抛硬币的总次数
        p_head (float): 硬币正面朝上的概率，默认为0.08
        
    返回:
        ndarray: 一个长度为n_flips的一维数组，其中1表示正面，0表示反面
    """
    # 待实现: 使用np.random.choice生成随机序列
    pass
    return np.random.choice([0, 1], size=n_flips, p=[1-p_head, p_head])
def calculate_waiting_times(coin_sequence):
    """计算两次正面之间的等待时间（反面次数）
    
    这个函数计算硬币序列中连续两次正面之间出现的反面次数。
    
    参数:
        coin_sequence (ndarray): 硬币序列，1表示正面，0表示反面
        
    返回:
        ndarray: 一个数组，包含所有等待时间（即连续两次正面之间的反面次数）
    
    示例:
        >>> sequence = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        >>> waiting_times = calculate_waiting_times(sequence)
        >>> print(waiting_times)  # 输出: [3 1]
        # 解释: 第一个1和第二个1之间有3个0，第二个1和第三个1之间有1个0
    """
    # 待实现:
    # 1. 使用np.nonzero找到所有正面（值为1）的位置索引
    # 2. 使用np.diff计算连续两个正面之间的间隔
    # 3. 减1得到中间的反面数量
    pass
     head_indices = np.nonzero(coin_sequence)[0]
    waiting_times = np.diff(head_indices) - 1
    return waiting_times

def plot_waiting_time_histogram(waiting_times, log_scale=False, n_flips=None):
    """绘制等待时间直方图
    
    这个函数绘制等待时间的频率分布直方图，可选择使用对数坐标。
    
    参数:
        waiting_times (ndarray): 等待时间数组
        log_scale (bool): 是否使用对数坐标，默认为False
        n_flips (int, optional): 抛硬币总次数，用于标题显示
        
    返回:
        None: 函数直接显示图形，不返回值
    """
    # 待实现:
    # 1. 创建图形
    # 2. 确定合适的bin数量
    # 3. 绘制直方图
    # 4. 设置坐标轴标签和标题
    # 5. 如果log_scale为True，设置y轴为对数刻度
    # 6. 显示图形
    pass
    plt.figure(figsize=(10, 6))
    
    # 确定合适的bin数量
    max_wait = max(waiting_times) if len(waiting_times) > 0 else 0
    bins = np.arange(0, max_wait + 2) - 0.5  # 确保每个整数值有一个bin
    
    # 绘制直方图
    plt.hist(waiting_times, bins=bins, density=True, alpha=0.7)
    plt.xlabel('Waiting Time (Number of Tails)')
    plt.ylabel('Frequency' if not log_scale else 'Frequency (Log Scale)')
    
    # 设置y轴为对数刻度（如果需要）
    if log_scale:
        plt.yscale('log')
        title = 'Waiting Time Distribution (Semi-log Scale)'
    else:
        title = 'Waiting Time Distribution'
    
    # 在标题中添加抛硬币次数
    if n_flips is not None:
        title += f' - {n_flips:,} Coin Flips'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
def analyze_waiting_time(waiting_times):
    """分析等待时间的统计特性
    
    这个函数计算等待时间的均值、标准差，并与理论值进行比较。
    
    参数:
        waiting_times (ndarray): 等待时间数组
        
    返回:
        dict: 包含以下键值对的字典:
            - "mean": 实验平均等待时间
            - "std": 实验等待时间标准差
            - "theoretical_mean": 理论平均等待时间（几何分布）
            - "exponential_mean": 理论平均等待时间（指数分布）
    
    注意:
        几何分布的均值为(1-p)/p，指数分布的均值为1/p，其中p是正面概率。
    """
    # 待实现:
    # 1. 计算实验平均等待时间和标准差
    # 2. 计算理论平均等待时间（几何分布和指数分布）
    # 3. 返回包含这些统计量的字典
    pass
    if len(waiting_times) == 0:
        return {"mean": None, "std": None, "theoretical_mean": None}
    
    mean_wait = np.mean(waiting_times)
    std_wait = np.std(waiting_times)
    
    # 理论上，等待时间应该服从几何分布，均值为(1-p)/p
    p_head = 0.08  # 硬币正面概率
    theoretical_mean = (1 - p_head) / p_head
    
    # 如果考虑指数分布模型（泊松过程中的等待时间）
    exponential_mean = 1 / p_head  # 1/λ = 1/0.08 = 12.5
    
    return {
        "mean": mean_wait,
        "std": std_wait,
        "theoretical_mean": theoretical_mean,
        "exponential_mean": exponential_mean
    }
def run_experiment(n_flips, title):
    """运行一次等待时间实验
    
    这个函数执行完整的等待时间实验流程，包括生成序列、计算等待时间、
    分析统计特性和绘制直方图。
    
    参数:
        n_flips (int): 抛硬币的总次数
        title (str): 实验标题，用于打印和图表显示
        
    返回:
        tuple: (waiting_times, stats)，其中:
            - waiting_times是等待时间数组
            - stats是统计分析结果字典
    """
    # 待实现:
    # 1. 打印实验标题
    # 2. 生成硬币序列并计算等待时间
    # 3. 分析等待时间并打印结果
    # 4. 绘制普通直方图和半对数直方图
    # 5. 返回等待时间数组和统计结果
    pass
     print(f"===== {title} =====")
    
    # 生成硬币序列并计算等待时间
    coin_sequence = generate_coin_sequence(n_flips)
    waiting_times = calculate_waiting_times(coin_sequence)
    
    # 分析等待时间
    stats = analyze_waiting_time(waiting_times)
    print(f"Average waiting time: {stats['mean']:.2f}")
    print(f"Theoretical average waiting time (Geometric): {stats['theoretical_mean']:.2f}")
    print(f"Theoretical average waiting time (Exponential): {stats['exponential_mean']:.2f}")
    
    # 绘制直方图
    plot_waiting_time_histogram(waiting_times, log_scale=False, n_flips=n_flips)
    plot_waiting_time_histogram(waiting_times, log_scale=True, n_flips=n_flips)
    
    return waiting_times, stats
if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 任务一：1000次抛掷
    waiting_times_1k, stats_1k = run_experiment(1000, "Task 1: 1000 Coin Flips")
    
    # 任务二：1000000次抛掷
    print("\n")
    waiting_times_1m, stats_1m = run_experiment(1000000, "Task 2: 1,000,000 Coin Flips")
