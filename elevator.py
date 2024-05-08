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
        hourly_floors = {0: 6, 3: 1, 4: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 3, 11: 1, 12: 2, 13: 1, 14: 1, 15: 8, 16: 1, 17: 1, 18: 1, 19: 1, 20: 5, 22: 1, 23: 11}
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
        # self.visualize_elevator_movements() 
        # The results show insignificant
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
        plt.ylabel('Wait Time (second)')
        plt.title('Wait Times for Each Request')
        plt.grid(True)
        plt.show()

requests = [('00:10', 10, 1), ('00:33', 13, 1), ('01:27', 7, 2), ('02:50', 3, 1), ('02:55', 9, 16), ('03:15', 8, 1), ('05:15', 11, 1), ('05:49', 13, 1), ('06:00', 1, 2), ('06:12', 3, 13), ('06:12', 9, 7), ('07:01', 2, 1), ('07:03', 7, 1), ('07:09', 2, 4), ('07:15', 15, 1), ('07:16', 11, 15), ('07:20', 13, 7), ('07:27', 7, 1), ('07:31', 13, 1), ('07:38', 6, 1), ('07:38', 2, 1), ('07:41', 4, 1), ('07:41', 16, 1), ('07:42', 18, 1), ('07:48', 5, 1), ('07:51', 7, 1), ('07:51', 14, 1), ('08:00', 4, 1), ('08:04', 12, 2), ('08:06', 2, 13), ('08:06', 6, 1), ('08:07', 3, 1), ('08:09', 14, 1), ('08:10', 17, 1), ('08:21', 10, 1), ('08:24', 3, 1), ('08:30', 13, 1), ('08:41', 2, 1), ('08:44', 13, 1), ('09:00', 13, 1), ('09:16', 9, 7), ('09:51', 13, 1), ('10:00', 1, 13), ('10:55', 16, 1), ('11:00', 13, 1), ('11:01', 4, 2), ('11:02', 1, 1), ('11:15', 9, 3), ('11:24', 18, 1), ('11:25', 1, 3), ('11:25', 12, 7), ('11:31', 17, 9), ('11:31', 11, 1), ('11:43', 8, 3), ('11:43', 14, 1), ('11:52', 17, 8), ('11:53', 4, 1), ('11:54', 12, 1), ('11:54', 11, 9), ('11:59', 4, 16), ('11:59', 13, 1), ('12:00', 1, 13), ('12:00', 1, 4), ('12:09', 12, 1), ('12:15', 2, 15), ('12:18', 13, 1), ('12:27', 12, 1), ('12:33', 18, 17), ('12:40', 11, 10), ('12:41', 14, 1), ('12:44', 10, 1), ('12:44', 16, 12), ('12:47', 3, 6), ('12:52', 9, 7), ('13:00', 13, 1), ('13:00', 4, 1), ('13:04', 10, 15), ('14:00', 1, 13), ('14:00', 10, 1), ('14:06', 17, 13), ('14:08', 15, 5), ('14:46', 18, 1), ('15:00', 13, 1), ('15:00', 1, 10), ('16:00', 10, 1), ('16:07', 15, 1), ('16:40', 8, 13), ('17:00', 7, 8), ('17:00', 1, 10), ('17:06', 3, 6), ('17:25', 2, 1), ('17:33', 9, 17), ('17:44', 7, 1), ('17:47', 10, 1), ('17:48', 18, 15), ('17:53', 17, 11), ('17:54', 14, 1), ('18:20', 7, 1), ('18:27', 8, 1), ('18:40', 2, 1), ('18:45', 10, 1), ('18:47', 12, 18), ('18:47', 12, 1), ('18:48', 18, 17), ('18:53', 15, 1), ('18:55', 12, 1), ('18:56', 12, 1), ('18:56', 5, 2), ('18:58', 17, 1), ('18:59', 14, 1), ('19:00', 1, 13), ('20:00', 1, 4), ('21:05', 7, 1), ('21:27', 6, 10), ('21:52', 15, 12), ('22:42', 9, 1), ('23:00', 2, 1), ('23:08', 9, 5), ('23:53', 17, 11), ('23:58', 14, 1)]
simulation = ElevatorSimulation(18, requests, 0.05, 0.1, 2, 1)
simulation.simulate_elevator('smart', 'scan')
simulation2 = ElevatorSimulation(18, requests, 0.05, 0.1, 2, 1)
simulation2.simulate_elevator('scan')
simulation3 = ElevatorSimulation(18, requests, 0.05, 0.1, 2, 1)
simulation3.simulate_elevator('smart', 'sstf')
simulation4 = ElevatorSimulation(18, requests, 0.05, 0.1, 2, 1)
simulation4.simulate_elevator('sstf')