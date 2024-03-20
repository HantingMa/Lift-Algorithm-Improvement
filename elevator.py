class ElevatorSimulation:
    def __init__(self, floors, requests, move_time, wait_time, initial_floor):
        self.floors = floors
        self.requests = [(r[0], r[1], r[2], False) for r in requests]
        self.move_time = move_time
        self.wait_time = wait_time
        self.initial_floor = initial_floor
        self.elevator_position_updates = []  # 记录电梯位置更新
        self.wait_times = []  # 记录每个请求的等待时间

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

        self.report_results()

    def update_wait_times_and_log(self, request_time, current_time, start_floor, end_floor):
        wait_time = current_time - request_time
        self.wait_times.append(wait_time)
        print(f"Moving from floor {start_floor} to {end_floor}. Wait time for this request: {wait_time:.2f}")

    def fcfs(self, current_floor, current_time):
        for i, (request_time, start_floor, end_floor, processed) in enumerate(self.requests):
            if not processed:
                current_time = max(current_time, request_time) + abs(current_floor - start_floor) * self.move_time
                self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                current_floor = end_floor
                current_time += abs(start_floor - end_floor) * self.move_time + 2 * self.wait_time
                self.log_elevator_position(current_time, current_floor)
                self.requests[i] = (request_time, start_floor, end_floor, True)

    def disk(self, current_floor, current_time):
        directions = [1, -1]
        requests = sorted(self.requests, key=lambda x: x[1])
        for direction in directions:
            for i, (request_time, start_floor, end_floor, processed) in enumerate(requests if direction == 1 else reversed(requests)):
                if ((direction == 1 and start_floor >= current_floor) or (direction == -1 and start_floor <= current_floor)) and not processed:
                    current_time = max(current_time, request_time) + abs(current_floor - start_floor) * self.move_time
                    self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor)
                    current_floor = end_floor
                    current_time += abs(start_floor - end_floor) * self.move_time + 2 * self.wait_time
                    self.log_elevator_position(current_time, current_floor)
                    if direction == 1:
                        self.requests[i] = (request_time, start_floor, end_floor, True)
                    else:
                        self.requests[len(self.requests) - 1 - i] = (request_time, start_floor, end_floor, True)

    def log_elevator_position(self, current_time, current_floor):
        self.elevator_position_updates.append((current_time, current_floor))
        print(f"At time {current_time:.2f}, elevator is at floor {current_floor}.")

    def report_results(self):
        average_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        max_wait_time = max(self.wait_times) if self.wait_times else 0
        print(f"\nAverage wait time: {average_wait_time:.2f}")
        print(f"Maximum wait time: {max_wait_time:.2f}")

# 示例
requests = [(0, 3, 5), (2, 1, 6), (4, 7, 2)]
simulation = ElevatorSimulation(10, requests, 1, 0.5, 1)
simulation.simulate_elevator('fcfs')

