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
            target_floor = self.smart(current_time)
            while current_floor != target_floor:
                direction = 1 if target_floor > current_floor else -1
                next_floor = current_floor + direction
                
                # Process requests while moving towards the target floor
                self.process_requests(current_floor, next_floor, current_time, secondary_algorithm)
                
                current_floor = next_floor
                current_time += self.move_time
                self.log_elevator_position(current_time, current_floor)
            
            # Process requests at the target floor
            self.process_requests(current_floor, current_floor, current_time, secondary_algorithm)
        else:
            if algorithm == 'sstf':
                self.sstf(current_floor, current_time)
            elif algorithm == 'scan':
                self.scan(current_floor, current_time)

        self.report_results()

    def process_requests(self, start_floor, end_floor, current_time, algorithm):
        for i, (request_time, floor, _, processed) in enumerate(self.requests):
            if start_floor <= floor <= end_floor and not processed and current_time >= request_time:
                arrival_time = current_time + self.move_time * abs(start_floor - floor)
                self.update_wait_times_and_log(request_time, arrival_time, floor, floor)
                self.requests[i] = (request_time, floor, _, True)
                
                if algorithm == 'sstf':
                    self.sstf(floor, arrival_time)
                elif algorithm == 'scan':
                    self.scan(floor, arrival_time)
                break

    def elevator_allocation(self):
        request_counts = [0] * (self.floors + 1)
        for _, start_floor, _, _ in self.requests:
            request_counts[start_floor] += 1

        total_requests = sum(request_counts)
        elevator_zones = []
        requests_per_elevator = total_requests // self.elevator_count

        current_count = 0
        start_floor = 1
        for i in range(1, self.floors + 1):
            current_count += request_counts[i]
            if current_count >= requests_per_elevator:
                elevator_zones.append((start_floor, i))
                start_floor = i + 1
                current_count = 0

        if start_floor <= self.floors:
            elevator_zones.append((start_floor, self.floors))

        return elevator_zones

    def sstf(self, current_floor, current_time):
        elevator_zones = self.elevator_allocation()
        for zone_start, zone_end in elevator_zones:
            while not all(r[3] for r in self.requests if zone_start <= r[1] <= zone_end):
                unprocessed_requests = [(i, r) for i, r in enumerate(self.requests) if not r[3] and zone_start <= r[1] <= zone_end]
                if unprocessed_requests:
                    closest_request_index, closest_request = min(unprocessed_requests, key=lambda x: abs(x[1][1] - current_floor))
                    request_time, start_floor, end_floor, _ = closest_request
                    # Calculate arrival time when elevator actually starts moving towards the request
                    arrival_time = max(current_time, request_time) + self.move_time * abs(current_floor - start_floor)
                    self.log_elevator_position(arrival_time, start_floor)
                    # Update current time to when the elevator reaches the destination floor
                    current_time = arrival_time + self.wait_time + self.move_time * abs(start_floor - end_floor)
                    self.log_elevator_position(current_time, end_floor)
                    self.update_wait_times_and_log(request_time, arrival_time, start_floor, end_floor)
                    current_floor = end_floor
                    self.requests[closest_request_index] = (request_time, start_floor, end_floor, True)

    def scan(self, current_floor, current_time):
        elevator_zones = self.elevator_allocation()
        for zone_start, zone_end in elevator_zones:
            direction = 1
            while not all(r[3] for r in self.requests if zone_start <= r[1] <= zone_end):
                unprocessed_requests = [(i, r) for i, r in enumerate(self.requests) if not r[3] and zone_start <= r[1] <= zone_end]
                if unprocessed_requests:
                    if direction == 1:
                        sorted_requests = sorted(unprocessed_requests, key=lambda x: x[1][1])
                    else:
                        sorted_requests = sorted(unprocessed_requests, key=lambda x: x[1][1], reverse=True)
                    for request_index, request in sorted_requests:
                        request_time, start_floor, end_floor, _ = request
                        if (direction == 1 and start_floor >= current_floor) or (direction == -1 and start_floor <= current_floor):
                            arrival_time = max(current_time, request_time) + self.move_time * abs(current_floor - start_floor)
                            self.log_elevator_position(arrival_time, start_floor)
                            self.update_wait_times_and_log(request_time, arrival_time, start_floor, end_floor)
                            current_floor = end_floor
                            current_time = arrival_time + self.move_time * abs(start_floor - end_floor) + self.wait_time
                            self.log_elevator_position(current_time, current_floor)
                            self.requests[request_index] = (request_time, start_floor, end_floor, True)
                direction *= -1


    def smart(self, current_time):
        hourly_floors = {0: 2, 2: 13, 3: 4, 4: 12, 6: 3, 7: 14, 8: 6, 9: 6, 11: 6, 12: 7, 13: 11, 14: 6, 15: 11, 16: 10, 17: 4, 18: 15, 19: 1, 20: 7, 21: 5, 22: 13, 23: 6}
        current_hour = int(current_time)
        
        best_floor = hourly_floors.get(current_hour, self.initial_floor)
        
        hour_requests = [r for r in self.requests if int(r[0]) == current_hour and not r[3]]
        if hour_requests:
            floor_weights = {}
            for request_time, start_floor, _, _ in hour_requests:
                wait_time = current_time - request_time
                floor_weights[start_floor] = floor_weights.get(start_floor, 0) + 1 + (wait_time / 10)
            best_floor = max(floor_weights, key=floor_weights.get)
        return best_floor

    def update_wait_times_and_log(self, request_time, arrival_time, start_floor, end_floor):
        wait_time = arrival_time - request_time
        self.wait_times.append(wait_time)
        print(f"Request at {request_time:.2f}, Moving from floor {start_floor} to {end_floor}. Wait time for this request: {wait_time:.2f}")

    def report_results(self):
        average_wait_time = np.mean(self.wait_times) if self.wait_times else 0
        max_wait_time = max(self.wait_times, default=0)
        total_travel_distance = sum(abs(self.elevator_position_updates[i][1] - self.elevator_position_updates[i-1][1]) for i in range(1, len(self.elevator_position_updates)))
        
        print(f"\nAverage wait time: {average_wait_time:.2f}")
        print(f"Maximum wait time: {max_wait_time:.2f}")
        print(f"Total travel distance: {total_travel_distance:.2f}")
        self.visualize_elevator_movements()
        self.visualize_wait_times()

    def log_elevator_position(self, current_time, current_floor):
        self.elevator_position_updates.append((current_time, current_floor))
    

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

requests = [('00:15', 1, 13), ('00:24', 1, 1), ('00:40', 1, 7), ('00:42', 13, 13), ('02:07', 1, 1), ('02:21', 5, 4), ('02:25', 1, 17), ('02:49', 8, 14), ('03:27', 17, 5), ('03:52', 12, 5), ('03:59', 9, 16), ('04:13', 1, 12), ('04:47', 16, 4), ('05:10', 1, 10), ('05:58', 1, 1), ('06:00', 2, 1), ('06:05', 12, 15), ('06:10', 9, 2), ('06:46', 1, 10), ('07:26', 5, 18), ('07:41', 1, 11), ('08:00', 1, 4), ('08:30', 1, 13), ('08:51', 9, 3), ('10:00', 1, 13), ('10:17', 17, 14), ('11:00', 1, 13), ('11:02', 2, 8), ('11:05', 1, 9), ('11:47', 1, 11), ('11:59', 1, 6), ('12:00', 1, 13), ('12:04', 12, 15), ('12:10', 1, 4), ('12:11', 2, 9), ('12:28', 8, 13), ('12:38', 1, 6), ('12:47', 11, 10), ('13:33', 5, 12), ('13:56', 5, 11), ('14:00', 1, 10), ('14:01', 17, 17), ('14:22', 1, 2), ('14:54', 3, 18), ('15:00', 1, 10), ('15:00', 1, 13), ('15:52', 6, 15), ('16:00', 1, 10), ('17:20', 8, 6), ('17:29', 1, 8), ('18:54', 1, 14), ('19:00', 13, 1), ('19:09', 1, 1), ('19:33', 6, 16), ('20:00', 1, 4), ('21:00', 1, 2), ('21:01', 1, 5), ('22:05', 10, 5), ('22:47', 16, 11), ('23:41', 18, 8)]
simulation = ElevatorSimulation(18, requests, 0.5, 0.1, 2, 2)
simulation.simulate_elevator('smart', 'sstf')
simulation2 = ElevatorSimulation(18, requests, 0.5, 0.1, 2, 2)
simulation2.simulate_elevator('sstf')