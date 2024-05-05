import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def time_to_decimal(timestr):
    hours, minutes = map(int, timestr.split(':'))
    return hours + minutes / 60

class ElevatorSimulation:
    def __init__(self, floors, requests, move_time, wait_time, initial_floor, elevator_count):
        self.requests = [(time_to_decimal(r[0]), r[1], r[2], False) for r in requests]
        self.floors = floors
        self.move_time = move_time
        self.wait_time = wait_time
        self.initial_floor = initial_floor
        self.elevator_count = elevator_count
        self.elevator_position_updates = []
        self.wait_times = []
        self.processed_requests = []

    def simulate_elevator(self, algorithm, secondary_algorithm=None):
        self.elevator_position_updates.clear()
        self.wait_times.clear()
        self.processed_requests.clear()

        current_floor = self.initial_floor
        current_time = 0
        self.log_elevator_position(current_time, current_floor)

        if algorithm == 'smart':
            current_floor = self.smart(current_time)
            current_time += self.move_time * abs(self.initial_floor - current_floor)
            self.log_elevator_position(current_time, current_floor)
            if secondary_algorithm:
                if secondary_algorithm == 'fcfs':
                    self.fcfs(current_floor, current_time)
                elif secondary_algorithm == 'disk':
                    self.disk(current_floor, current_time)
        else:
            if algorithm == 'fcfs':
                self.fcfs(current_floor, current_time)
            elif algorithm == 'disk':
                self.disk(current_floor, current_time)

        self.report_results()

    def elevator_allocation(self):
        elevators = [0] * (self.elevator_count + 1)

        def add_floor(e):
            best = float('inf')
            elevator_number = 0
            for i in range(1, len(e)):
                floors_serviced = e[i] - e[i-1] + 1
                cir_time = self.move_time * e[i] * 2
                cir_time += self.wait_time * floors_serviced
                avg_carry = cir_time * floors_serviced  # Simplified for this example
                if cir_time + ((cir_time / 100) * avg_carry) < best:
                    elevator_number = i
                    best = cir_time + ((cir_time / 100) * avg_carry)
            for i in range(elevator_number, len(e)):
                e[i] += 1
            return e

        for _ in range(1, self.floors):
            elevators = add_floor(elevators)
        return elevators[1:]  # Return only the elevator stopping floors

    def fcfs(self, current_floor, current_time):
        # Example of modified FCFS to use elevator allocation
        elevator_zones = self.elevator_allocation()
        for zone_start, zone_end in zip([0] + elevator_zones[:-1], elevator_zones):
            for i, (request_time, start_floor, end_floor, processed) in enumerate(self.requests):
                if not processed and zone_start <= start_floor <= zone_end:
                    wait_time = max(current_time, request_time) - request_time
                    current_time += self.move_time * abs(current_floor - start_floor)
                    self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                    current_floor = end_floor
                    current_time += self.move_time * abs(start_floor - end_floor) + self.wait_time
                    self.log_elevator_position(current_time, current_floor)
                    index = self.requests.index((request_time, start_floor, end_floor, processed))
                    self.requests[index] = (request_time, start_floor, end_floor, True)
    def disk(self, current_floor, current_time):
        # Disk scheduling-like approach, with modifications for elevator allocation
        elevator_zones = self.elevator_allocation()
        for zone_start, zone_end in zip([0] + elevator_zones[:-1], elevator_zones):
            while not all(r[3] for r in self.requests if zone_start <= r[1] <= zone_end):
                for direction in [1, -1]:  # Up and Down
                    sorted_requests = sorted((r for r in self.requests if not r[3] and zone_start <= r[1] <= zone_end), key=lambda x: x[1], reverse=(direction == -1))
                    for request_time, start_floor, end_floor, processed in sorted_requests:
                        if (direction == 1 and start_floor >= current_floor) or (direction == -1 and start_floor <= current_floor):
                            current_time = max(current_time, request_time) + self.move_time * abs(current_floor - start_floor)
                            self.log_elevator_position(current_time, start_floor)
                            self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                            current_floor = end_floor
                            current_time += self.move_time * abs(start_floor - end_floor) + self.wait_time
                            self.log_elevator_position(current_time, current_floor)
                            index = self.requests.index((request_time, start_floor, end_floor, processed))
                            self.requests[index] = (request_time, start_floor, end_floor, True)

    def smart(self, current_time):
        hourly_floors = {0: 2, 2: 13, 3: 4, 4: 12, 6: 3, 7: 14, 8: 6, 9: 6, 11: 6, 12: 7, 13: 11, 14: 6, 15: 11, 16: 10, 17: 4, 18: 15, 19: 1, 20: 7, 21: 5, 22: 13, 23: 6}
        current_hour = int(current_time)
        
        # 根据当前时间找到最佳楼层
        best_floor = hourly_floors.get(current_hour, self.initial_floor)
        
        # 如果当前时间的请求中有更高频率的楼层,则选择该楼层
        hour_requests = [r for r in self.requests if int(r[0]) == current_hour and not r[3]]
        if hour_requests:
            floor_counts = {}
            for _, start_floor, _, _ in hour_requests:
                floor_counts[start_floor] = floor_counts.get(start_floor, 0) + 1
            most_frequent_floor = max(floor_counts, key=floor_counts.get)
            if floor_counts[most_frequent_floor] > 1:
                best_floor = most_frequent_floor
        return best_floor
    
    def update_wait_times_and_log(self, request_time, current_time, start_floor, end_floor):
        wait_time = current_time - request_time
        self.wait_times.append(wait_time)
        print(f"Moving from floor {start_floor} to {end_floor}. Wait time for this request: {wait_time:.2f}")

    def log_elevator_position(self, current_time, current_floor):
        self.elevator_position_updates.append((current_time, current_floor))

    def report_results(self):
        average_wait_time = np.mean(self.wait_times) if self.wait_times else 0
        max_wait_time = max(self.wait_times, default=0)
        print(f"\nAverage wait time: {average_wait_time:.2f}")
        print(f"Maximum wait time: {max_wait_time:.2f}")
        self.visualize_elevator_movements()
        self.visualize_wait_times()
    def visualize_elevator_movements(self):
        times, floors = zip(*self.elevator_position_updates)
        plt.figure(figsize=(10, 6))
        plt.step(times, floors, where='post', label='Elevator Position')
        plt.ylabel('Floor')
        plt.xlabel('Time (minutes)')
        plt.title('Elevator Movement Over Time')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_wait_times(self):
        request_times, wait_times = zip(*[(req[0], wt) for req, wt in zip(self.requests, self.wait_times)])
        plt.figure(figsize=(10, 6))
        plt.scatter(request_times, wait_times, color='red')
        plt.xlabel('Request Time (hours)')
        plt.ylabel('Wait Time (minutes)')
        plt.title('Wait Times for Each Request')
        plt.grid(True)
        plt.show()

# Example requests (modified for simplification)
requests = [('00:12', 1, 11), ('00:47', 17, 10), ('00:56', 7, 13), ('01:13', 1, 15), ('01:47', 3, 1), ('02:12', 5, 10), ('04:41', 2, 17), ('05:54', 1, 5), ('06:00', 2, 1), ('06:05', 1, 15), ('06:08', 10, 6), ('06:31', 1, 9), ('06:50', 1, 3), ('06:51', 2, 5), ('07:24', 1, 1), ('07:30', 1, 12), ('07:34', 1, 10), ('07:42', 9, 9), ('08:00', 1, 4), ('08:07', 12, 7), ('08:30', 1, 13), ('09:01', 1, 5), ('10:00', 1, 13), ('11:00', 1, 13), ('11:05', 18, 17), ('12:00', 1, 13), ('12:43', 1, 16), ('12:56', 1, 4), ('13:00', 1, 13), ('13:54', 1, 2), ('14:00', 1, 13), ('14:00', 1, 10), ('14:08', 14, 14), ('14:39', 8, 14), ('14:56', 14, 16), ('15:00', 1, 13), ('15:00', 1, 10), ('15:04', 18, 6), ('15:19', 7, 18), ('15:28', 12, 6), ('15:38', 16, 7), ('15:56', 5, 9), ('16:00', 1, 10), ('17:00', 1, 10), ('17:06', 1, 3), ('17:34', 12, 8), ('17:49', 15, 17), ('17:52', 1, 3), ('18:03', 2, 12), ('20:00', 1, 4), ('21:55', 1, 15), ('22:04', 15, 8), ('22:05', 7, 13), ('22:14', 1, 6), ('22:34', 4, 1), ('22:37', 13, 15), ('23:07', 1, 3)]

simulation = ElevatorSimulation(18, requests, 0.5, 0.1, 2, 2)
simulation.simulate_elevator('smart', 'disk')
simulation2 = ElevatorSimulation(18, requests, 0.5, 0.1, 2, 2)
simulation2.simulate_elevator('disk')
#simulation.simulate_elevator('disk')
#simulation.simulate_elevator('smart', 'disk')