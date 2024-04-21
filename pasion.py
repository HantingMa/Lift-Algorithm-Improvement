import numpy as np
import pandas as pd
from scipy.stats import poisson

# 设定电梯运行参数
total_floors = 18  # 总楼层数
requests_per_day = 40  # 每日请求的平均数量

# 模拟特定用户的电梯使用时间
times = {
    'Andy': [('08:30', 13, 1), ('19:00', 1, 13)],  # 出发上班和回家
    'Bob': [('08:00', 4, 1), ('20:00', 4, 1)],  # 早晚遛狗
    'Candy': [('14:00', 10, 1), ('15:00', 10, 1), ('16:00', 10, 1), ('17:00', 10, 1)],  # 下午拿外卖
    'David': [('21:00', 2, 1), ('06:00', 1, 2)],  # 晚上出发和凌晨回家
    'Elly': [('10:00', 13, 1), ('11:00', 13, 1), ('12:00', 13, 1), ('13:00', 13, 1), ('14:00', 13, 1), ('15:00', 13, 1)]  # 顾客访问
}

# 创建模拟数据
simulation_data = []

# 按用户的固定时间添加数据
for user, trips in times.items():
    for time, start_floor, end_floor in trips:
        if np.random.rand() <= 0.8:  # 80%的概率按计划进行
            simulation_data.append((time, start_floor, end_floor))

# 随机添加其他请求
for _ in range(poisson.rvs(requests_per_day)):  # 按泊松分布随机生成的请求次数
    start_floor = np.random.randint(1, total_floors + 1)
    end_floor = np.random.choice([1, np.random.randint(2, total_floors + 1)], p=[0.4, 0.6])
    request_time = np.random.uniform(0, 24)  # 一天中的任意时间
    formatted_time = f"{int(request_time):02d}:{int((request_time % 1) * 60):02d}"
    simulation_data.append((formatted_time, start_floor, end_floor))

# 创建DataFrame
df = pd.DataFrame(simulation_data, columns=['Request Time', 'Start Floor', 'End Floor'])

# 输出模拟结果
df_sorted = df.sort_values(by='Request Time')
formatted_output = [(row['Request Time'], row['End Floor'], row['Start Floor']) for index, row in df_sorted.iterrows()]

# Show the formatted output
# print(df_sorted)

print(formatted_output)