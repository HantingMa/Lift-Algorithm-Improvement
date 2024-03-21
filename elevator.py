import random
import matplotlib.pyplot as plt

class ElevatorSimulation:
    def __init__(self, floors, requests, move_time, wait_time, initial_floor):
        self.floors = floors
        self.requests = sorted(requests, key=lambda x: x[0])  # 根据请求时间排序
        self.move_time = move_time
        self.wait_time = wait_time
        self.initial_floor = initial_floor
        self.wait_times = []  # 记录每个请求的等待时间
        self.elevator_trajectory = []  # 用于记录电梯的运行轨迹

    def simulate_elevator(self, algorithm):
        self.elevator_trajectory.clear()
        self.wait_times.clear()

        current_floor = self.initial_floor
        current_time = 0
        self.log_elevator_position(current_time, current_floor)

        if algorithm == 'fcfs':
            self.fcfs(current_floor, current_time)
        elif algorithm == 'disk':
            self.disk(current_floor, current_time)

        self.visualize_trajectory()
        self.report_results()

    def update_wait_times_and_log(self, request_time, current_time, start_floor, end_floor, current_floor):
        # 等待时间的计算现在正确地包含了从当前位置到请求起始位置的移动时间
        wait_time = (current_time + abs(current_floor - start_floor) * self.move_time) - request_time
        self.wait_times.append(wait_time)
        print(f"Moving from floor {start_floor} to {end_floor}. Wait time for this request: {wait_time:.2f}")

    def fcfs(self, current_floor, current_time):
        for (request_time, start_floor, end_floor) in self.requests:
            current_time = max(current_time, request_time)
            self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor, current_floor)
            current_floor = end_floor  # 更新当前楼层
            current_time += abs(start_floor - end_floor) * self.move_time + self.wait_time  # 更新当前时间
            self.log_elevator_position(current_time, current_floor)

    def disk(self, current_floor, current_time):
        directions = [1, -1]  # 先上后下
        requests = sorted(self.requests, key=lambda x: x[1])  # 根据起始楼层排序

        for direction in directions:
            for (request_time, start_floor, end_floor) in requests if direction == 1 else reversed(requests):
                # 检查方向和请求楼层关系
                if direction == 1 and start_floor >= current_floor or direction == -1 and start_floor <= current_floor:
                    current_time = max(current_time, request_time)
                    self.update_wait_times_and_log(request_time, current_time, start_floor, end_floor, current_floor)
                    current_floor = end_floor
                    current_time += abs(start_floor - end_floor) * self.move_time + self.wait_time
                    self.log_elevator_position(current_time, current_floor)

    def log_elevator_position(self, current_time, current_floor):
        self.elevator_trajectory.append((current_time, current_floor))
        print(f"At time {current_time:.2f}, elevator is at floor {current_floor}.")

    def visualize_trajectory(self):
        times, floors = zip(*self.elevator_trajectory)
        plt.figure()
        plt.plot(times, floors, marker='o')
        plt.yticks(range(1, self.floors + 1))
        plt.xlabel('Time')
        plt.ylabel('Floor')
        plt.title('Elevator Trajectory')
        plt.grid(True)
        plt.show()

    def report_results(self):
        average_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        max_wait_time = max(self.wait_times) if self.wait_times else 0
        print(f"\nAverage wait time: {average_wait_time:.2f}")
        print(f"Maximum wait time: {max_wait_time:.2f}")

# 生成60个随机请求
floors = 10
num_requests = 60
random_requests = [(random.randint(0, 30), random.randint(1, floors), random.randint(1, floors)) for _ in range(num_requests)]

# 实例化并运行模拟
simulation = ElevatorSimulation(floors, random_requests, 1, 0.5, 1)
simulation.simulate_elevator('disk')  # 现在尝试 disk 算法
simulation.simulate_elevator('fcfs')
