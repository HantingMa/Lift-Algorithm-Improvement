import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime 

# 数据准备
requests = [
('00:02', 16, 14), ('00:06', 11, 2), ('01:51', 1, 17), ('02:32', 1, 12), 
    ('02:35', 11, 6), ('02:52', 1, 4), ('02:55', 1, 11), ('03:19', 3, 7), 
    ('04:02', 1, 12), ('04:16', 1, 4), ('04:40', 2, 13), ('04:48', 5, 3), 
    ('05:31', 14, 15), ('06:00', 2, 1), ('06:54', 8, 4), ('07:26', 4, 18), 
    ('07:37', 4, 3), ('07:44', 7, 12), ('07:52', 1, 8), ('08:29', 1, 10), 
    ('08:30', 1, 13), ('10:00', 1, 13), ('10:16', 5, 6), ('10:52', 17, 13), 
    ('11:00', 1, 13), ('11:04', 18, 9), ('11:17', 11, 1), ('12:00', 1, 13), 
    ('12:52', 5, 16), ('13:00', 1, 13), ('13:51', 1, 5), ('14:00', 1, 13), 
    ('14:30', 1, 3), ('15:00', 1, 10), ('15:00', 1, 13), ('15:20', 17, 12), 
    ('16:00', 1, 10), ('16:52', 5, 17), ('17:00', 1, 10), ('17:50', 10, 2), 
    ('17:50', 4, 16), ('17:53', 1, 17), ('18:43', 5, 14), ('19:00', 13, 1), 
    ('19:00', 8, 10), ('20:00', 1, 4), ('20:15', 18, 17), ('20:51', 1, 7), 
    ('20:53', 16, 7), ('21:00', 1, 2), ('21:37', 17, 12), ('22:02', 1, 10), 
    ('22:04', 3, 7), ('22:39', 15, 4), ('22:41', 15, 14), ('23:13', 3, 3), 
    ('23:56', 14, 8),('00:55', 1, 7), ('01:17', 11, 6), ('01:51', 1, 5), 
    ('02:36', 11, 7), ('02:46', 1, 6), ('05:21', 1, 10), ('06:00', 2, 1), 
    ('07:05', 14, 9), ('07:17', 1, 7), ('07:40', 8, 4), ('07:50', 1, 11), 
    ('08:00', 1, 4), ('08:30', 1, 13), ('08:35', 14, 12), ('08:49', 17, 3), 
    ('08:53', 14, 16), ('10:19', 4, 10), ('11:00', 1, 13), ('11:06', 13, 13), 
    ('11:34', 1, 16), ('11:40', 5, 14), ('12:49', 1, 17), ('13:00', 1, 13), 
    ('14:00', 1, 10), ('14:00', 1, 13), ('14:26', 16, 6), ('14:43', 1, 3), ('15:00', 1, 13), 
    ('15:28', 12, 4), ('16:00', 1, 10), ('16:18', 1, 6), ('16:36', 12, 6), ('16:39', 1, 4), 
    ('17:13', 10, 3), ('17:49', 15, 3), ('18:15', 1, 15), ('18:49', 4, 2), ('20:00', 1, 4), 
    ('20:05', 5, 7), ('20:19', 4, 13), ('20:22', 1, 10), ('20:53', 10, 3), ('21:00', 1, 2), 
    ('21:27', 1, 6), ('21:54', 2, 1), ('22:06', 6, 15), ('22:38', 6, 10), ('22:57', 1, 10)
]

# 转换为DataFrame
df = pd.DataFrame(requests, columns=['time', 'start_floor', 'end_floor'])

# 添加小时列
df['hour'] = df['time'].apply(lambda x: datetime.strptime(x, "%H:%M").hour)

# 计算距离
df['distance'] = abs(df['start_floor'] - df['end_floor'])

# 定义特征和目标
X = df[['hour', 'start_floor', 'end_floor', 'distance']]
y = df['distance']

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测和模型评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 初始化楼层数
floors = range(1, 18)

# 模拟每小时的最佳初始楼层
best_floors = {}
for hour in df['hour'].unique():
    group = df[df['hour'] == hour]
    predicted_times = []
    for floor in floors:
        # 计算每个楼层的预测时间
        times = [model.predict([[hour, floor, end_floor, abs(floor - end_floor)]])[0] for end_floor in group['end_floor']]
        total_time = sum(times)
        predicted_times.append(total_time)
    best_floors[hour] = floors[np.argmin(predicted_times)]

# 输出每小时的最佳初始楼层
print("Best initial floors per hour:")
for hour in sorted(best_floors.keys()):
    print(f"Hour {hour}: Best initial floor is {best_floors[hour]}")