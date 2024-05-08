import numpy as np
import pandas as pd
from scipy.stats import poisson

# Set elevator operation parameters
total_floors = 18  # Total number of floors
requests_per_day = 100  # Average number of requests per day
peak_hours = [(7, 9), (11, 13), (17, 19)]  # Peak periods

# Simulate specific user elevator usage times
times = {
    'Andy': [('08:30', 13, 1), ('19:00', 1, 13)],  # Depart for work and return home
    'Bob': [('08:00', 4, 1), ('12:00', 1, 4), ('13:00', 4, 1), ('20:00', 1, 4)],  # Morning and evening dog walk, lunch break at home
    'Candy': [('14:00', 10, 1), ('15:00', 1, 10), ('16:00', 10, 1), ('17:00', 1, 10)],  # Afternoon food delivery
    'David': [('23:00', 2, 1), ('06:00', 1, 2)],  # Late night departure and early morning return home
    'Elly': [('10:00', 1, 13), ('11:00', 13, 1), ('12:00', 1, 13), ('13:00', 13, 1), ('14:00', 1, 13), ('15:00', 13, 1)]  # Customer visits
}

# Create simulation data
simulation_data = []

# Add data based on the fixed times of users
for user, trips in times.items():
    for time, start_floor, end_floor in trips:
        if np.random.rand() <= 0.8:  # 80% probability of following the planned schedule
            simulation_data.append((time, start_floor, end_floor))

# Randomly add other requests
for _ in range(poisson.rvs(requests_per_day)):
    start_floor = np.random.choice(range(1, total_floors + 1), p=[0.05] + [0.95 / (total_floors - 1)] * (total_floors - 1))  # Higher probability starting from the 1st floor
    end_floor = np.random.choice([1, np.random.randint(2, total_floors + 1)], p=[0.6, 0.4])  # Higher probability going to the 1st floor
    
    # Adjust the request time probability distribution based on peak periods
    if np.random.rand() <= 0.7:  # 70% of requests occur during peak hours
        peak_start, peak_end = peak_hours[np.random.randint(len(peak_hours))]
        request_time = np.random.uniform(peak_start, peak_end)
    else:
        request_time = np.random.uniform(0, 24)
        
    formatted_time = f"{int(request_time):02d}:{int((request_time % 1) * 60):02d}"
    simulation_data.append((formatted_time, start_floor, end_floor))

# Create DataFrame
df = pd.DataFrame(simulation_data, columns=['Request Time', 'Start Floor', 'End Floor'])

# Output simulation results
df_sorted = df.sort_values(by='Request Time')
formatted_output = [(row['Request Time'], row['Start Floor'], row['End Floor']) for index, row in df_sorted.iterrows()]

print(formatted_output)
