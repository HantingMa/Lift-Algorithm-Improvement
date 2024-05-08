import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
requests = [('00:40', 11, 5), ('00:48', 5, 7), ('03:19', 10, 1), ('04:17', 11, 1), ('06:00', 1, 2), ('07:02', 14, 3), ('07:12', 6, 14), ('07:15', 15, 1), ('07:16', 2, 1), ('07:21', 9, 10), ('07:22', 16, 12), ('07:28', 2, 1), ('07:34', 17, 1), ('07:48', 15, 1), ('07:49', 16, 1), ('07:56', 3, 1), ('08:06', 9, 1), ('08:09', 10, 17), ('08:09', 14, 1), ('08:13', 12, 1), ('08:15', 1, 1), ('08:21', 12, 1), ('08:28', 15, 1), ('08:29', 6, 1), ('08:30', 7, 6), ('08:30', 13, 1), ('08:32', 16, 1), ('08:32', 5, 11), ('08:32', 13, 1), ('08:35', 17, 1), ('09:45', 10, 15), ('09:46', 18, 1), ('09:47', 4, 1), ('10:00', 1, 13), ('10:27', 18, 1), ('11:06', 5, 1), ('11:12', 12, 1), ('11:14', 7, 1), ('11:21', 1, 1), ('11:25', 2, 1), ('11:38', 13, 8), ('11:38', 14, 1), ('11:48', 7, 1), ('12:00', 1, 13), ('12:00', 1, 4), ('12:05', 6, 6), ('12:09', 6, 1), ('12:15', 7, 1), ('12:15', 1, 1), ('12:17', 13, 1), ('12:20', 4, 1), ('12:22', 10, 6), ('12:29', 7, 10), ('12:31', 18, 13), ('12:33', 17, 1), ('12:44', 5, 1), ('12:48', 5, 3), ('12:48', 1, 3), ('12:48', 13, 1), ('12:51', 17, 1), ('12:57', 5, 9), ('13:00', 4, 1), ('13:06', 2, 18), ('13:52', 2, 1), ('13:57', 1, 1), ('14:00', 10, 1), ('14:40', 7, 13), ('14:54', 2, 1), ('15:00', 1, 10), ('15:00', 13, 1), ('16:00', 10, 1), ('17:00', 1, 10), ('17:04', 15, 7), ('17:07', 14, 1), ('17:07', 5, 1), ('17:08', 3, 2), ('17:08', 3, 1), ('17:15', 18, 1), ('17:18', 15, 1), ('17:25', 7, 6), ('17:25', 16, 6), ('17:36', 14, 12), ('17:44', 7, 1), ('17:50', 6, 1), ('17:51', 14, 7), ('17:51', 17, 1), ('17:58', 11, 1), ('18:05', 18, 11), ('18:06', 16, 15), ('18:14', 13, 1), ('18:23', 12, 1), ('18:29', 5, 1), ('18:29', 7, 6), ('18:31', 5, 1), ('18:33', 15, 1), ('18:33', 2, 1), ('18:34', 10, 1), ('18:36', 10, 1), ('18:49', 7, 12), ('18:52', 3, 1), ('19:03', 17, 1), ('19:47', 14, 1), ('19:58', 2, 1), ('20:00', 1, 4), ('20:11', 11, 12), ('22:46', 14, 1), ('23:00', 2, 1), ('23:10', 13, 12), ('23:35', 3, 14), ('23:45', 8, 10)]
# Convert to DataFrame
df = pd.DataFrame(requests, columns=['time', 'start_floor', 'end_floor'])

# Convert time to hour of the day
df['hour'] = df['time'].apply(lambda x: datetime.strptime(x, "%H:%M").hour)

# Calculate the distance
df['distance'] = abs(df['start_floor'] - df['end_floor'])

# Define features and target
X = df[['hour', 'start_floor', 'end_floor', 'distance']]
y = df['distance']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Simulate the best initial floor for each hour
floors = range(1, 18)
best_floors = {}
for hour in df['hour'].unique():
    group = df[df['hour'] == hour]
    predicted_times = []
    for floor in floors:
        prediction_data = pd.DataFrame({
            'hour': [hour] * len(group),
            'start_floor': [floor] * len(group),
            'end_floor': group['end_floor'],
            'distance': abs(floor - group['end_floor'])
        })
        times = model.predict(prediction_data)
        total_time = sum(times)
        predicted_times.append(total_time)
    best_floors[hour] = floors[np.argmin(predicted_times)]

print(best_floors)

# Plot the best initial floor for each hour
plt.figure(figsize=(10, 6))
hours = sorted(best_floors.keys())
best_floors_list = [best_floors[hour] for hour in hours]
sns.lineplot(x=hours, y=best_floors_list, marker='o')
plt.title('Best Initial Floors for Each Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Best Initial Floor')
plt.xticks(hours)  # Ensure all hours are shown
plt.grid(True)
plt.show()