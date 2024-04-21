import numpy as np
import pandas as pd
from scipy.stats import poisson

def time_to_decimal(timestr):
    """Convert time string 'HH:MM' to decimal hours."""
    hours, minutes = map(int, timestr.split(':'))
    return hours + minutes / 60

class ElevatorSimulation:
    def __init__(self, floors, requests, move_time, wait_time, initial_floor):
        # Convert 'HH:MM' time format to decimal hours in requests
        self.requests = [(time_to_decimal(r[0]), r[1], r[2], False) for r in requests]
        self.floors = floors
        self.move_time = move_time
        self.wait_time = wait_time
        self.initial_floor = initial_floor
        self.elevator_position_updates = []  # Record elevator position updates
        self.wait_times = []  # Record waiting times for each request

    def simulate_elevator(self, algorithm):
        self.elevator_position_updates.clear()
        self.wait_times.clear()

        current_floor = self.initial_floor
        current_time = 0
        self.log_elevator_position(current_time, current_floor)

        if algorithm == 'fcfs':
            self.fcfs(current_floor, current_time)
        elif algorithm == 'disk':
            self.disk(current_floor, current_time)
        elif algorithm == 'smart':
            self.smart(current_floor, current_time)

        self.report_results()

    def update_wait_times_and_log(self, request_time, current_time, start_floor, end_floor):
        wait_time = current_time - request_time
        self.wait_times.append(wait_time)
        print(f"Moving from floor {start_floor} to {end_floor}. Wait time for this request: {wait_time:.2f}")

    def fcfs(self, current_floor, current_time):
        for i, (request_time, start_floor, end_floor, processed) in enumerate(self.requests):
            if not processed:
                current_time = max(current_time, request_time) + self.move_time * abs(current_floor - start_floor)
                self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                current_floor = end_floor
                current_time += self.move_time * abs(start_floor - end_floor) + self.wait_time
                self.log_elevator_position(current_time, current_floor)
                self.requests[i] = (request_time, start_floor, end_floor, True)

    def smart(self, current_floor, current_time):
        pass

    def disk(self, current_floor, current_time):
        while not all(r[3] for r in self.requests):  # Check if all requests are processed
            for direction in [1, -1]:  # Up and Down
                sorted_requests = sorted((r for r in self.requests if not r[3]), key=lambda x: x[1], reverse=(direction == -1))
                for i, (request_time, start_floor, end_floor, processed) in enumerate(sorted_requests):
                    if (direction == 1 and start_floor >= current_floor) or (direction == -1 and start_floor <= current_floor):
                        current_time = max(current_time, request_time) + self.move_time * abs(current_floor - start_floor)
                        self.log_elevator_position(current_time, start_floor)
                        self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                        current_floor = end_floor
                        current_time += self.move_time * abs(start_floor - end_floor) + self.wait_time
                        self.log_elevator_position(current_time, current_floor)
                        index = self.requests.index((request_time, start_floor, end_floor, processed))
                        self.requests[index] = (request_time, start_floor, end_floor, True)

    def log_elevator_position(self, current_time, current_floor):
        self.elevator_position_updates.append((current_time, current_floor))
        print(f"At time {current_time:.2f}, elevator is at floor {current_floor}.")

    def report_results(self):
        average_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        max_wait_time = max(self.wait_times) if self.wait_times else 0
        print(f"\nAverage wait time: {average_wait_time:.2f}")
        print(f"Maximum wait time: {max_wait_time:.2f}")
        # Create a DataFrame from the position updates for visualization
        df_positions = pd.DataFrame(self.elevator_position_updates, columns=['Time', 'Floor'])
        print("\nElevator Movements:")
        print(df_positions)

# Example requests, time format is 'HH:MM'
requests = [('00:02', 16, 14), ('00:06', 11, 2), ('01:51', 1, 17), ('02:32', 1, 12), ('02:35', 11, 6), ('02:52', 1, 4), ('02:55', 1, 11), ('03:19', 3, 7), ('04:02', 1, 12), ('04:16', 1, 4), ('04:40', 2, 13), ('04:48', 5, 3), ('05:31', 14, 15), ('06:00', 2, 1), ('06:54', 8, 4), ('07:26', 4, 18), ('07:37', 4, 3), ('07:44', 7, 12), ('07:52', 1, 8), ('08:29', 1, 10), ('08:30', 1, 13), ('10:00', 1, 13), ('10:16', 5, 6), ('10:52', 17, 13), ('11:00', 1, 13), ('11:04', 18, 9), ('11:17', 11, 1), ('12:00', 1, 13), ('12:52', 5, 16), ('13:00', 1, 13), ('13:51', 1, 5), ('14:00', 1, 13), ('14:30', 1, 3), ('15:00', 1, 10), ('15:00', 1, 13), ('15:20', 17, 12), ('16:00', 1, 10), ('16:52', 5, 17), ('17:00', 1, 10), ('17:50', 10, 2), ('17:50', 4, 16), ('17:53', 1, 17), ('18:43', 5, 14), ('19:00', 13, 1), ('19:00', 8, 10), ('20:00', 1, 4), ('20:15', 18, 17), ('20:51', 1, 7), ('20:53', 16, 7), ('21:00', 1, 2), ('21:37', 17, 12), ('22:02', 1, 10), ('22:04', 3, 7), ('22:39', 15, 4), ('22:41', 15, 14), ('23:13', 3, 3), ('23:56', 14, 8)]

# Initialize simulation
simulation = ElevatorSimulation(20, requests, 1, 0.5, 1)  # Modified to include more floors
simulation.simulate_elevator('disk')
simulation.simulate_elevator('fcfs')
