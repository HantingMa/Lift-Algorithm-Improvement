import numpy as np
import pandas as pd
from scipy.stats import poisson

# 设定电梯运行参数
total_floors = 18  # 总楼层数
requests_per_day = 40  # 每日请求的平均数量

# 模拟特定用户的电梯使用时间
times = {
    'Andy': [(8.5, 13, 1), (19, 1, 13)],  # 出发上班和回家
    'Bob': [(8, 4, 1), (20, 4, 1)],  # 早晚遛狗
    'Candy': [(14, 10, 1), (15, 10, 1), (16, 10, 1), (17, 10, 1)],  # 下午拿外卖
    'David': [(21, 2, 1), (6, 1, 2)],  # 晚上出发和凌晨回家
    'Elly': [(10, 13, 1), (11, 13, 1), (12, 13, 1), (13, 13, 1), (14, 13, 1), (15, 13, 1)]  # 顾客访问
}

# 创建模拟数据
simulation_data = []

# 按用户的固定时间添加数据
for user, trips in times.items():
    for time, start_floor, end_floor in trips:
        if np.random.rand() <= 0.8:  # 80%的概率按计划进行
            simulation_data.append((start_floor, end_floor, time))

# 随机添加其他请求
for _ in range(poisson.rvs(requests_per_day)):  # 按泊松分布随机生成的请求次数
    start_floor = np.random.randint(1, total_floors + 1)
    if np.random.rand() <= 0.4:  # 40% 概率到地面层
        end_floor = 1
    elif np.random.rand() <= 0.8:  # 继续40% 概率从地面层到楼层
        end_floor = np.random.randint(2, total_floors + 1)
    else:  # 剩余20% 随机楼层到楼层
        end_floor = np.random.randint(1, total_floors + 1)
    request_time = np.random.uniform(0, 24)  # 一天中的任意时间
    simulation_data.append((start_floor, end_floor, request_time))

# 创建DataFrame
df = pd.DataFrame(simulation_data, columns=['Start Floor', 'End Floor', 'Request Time'])

# 输出模拟结果
df_sorted = df.sort_values(by='Request Time')
df_sorted['Request Time'] = (df_sorted['Request Time'] * 60).apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
print(df_sorted)

# Adjusting request times from decimal hours to 24-hour time format

