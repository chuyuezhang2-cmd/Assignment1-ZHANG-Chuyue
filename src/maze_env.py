import numpy as np
import pygame
import random
from collections import deque  # 用于 BFS

class MazeEnv:
    def __init__(self, grid_size=10, num_traps=2, num_coins=6):  # 初始陷阱减小，硬币增加
        self.grid_size = grid_size
        self.num_traps = num_traps
        self.num_coins = num_coins
        self.reset()

    def reset(self):
        # 生成迷宫
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                if random.random() < 0.2:
                    self.grid[i, j] = 1

        self.agent_pos = [1, 1]
        while self.grid[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]

        self.goal_pos = [self.grid_size-2, self.grid_size-2]
        while self.goal_pos == self.agent_pos or self.grid[self.goal_pos[0], self.goal_pos[1]] == 1:
            self.goal_pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]

        # 检查连通性：确保代理能到终点
        while not self.is_connected():
            # 不通，重生成内部墙壁
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    self.grid[i, j] = 1 if random.random() < 0.2 else 0
            # 重置代理和终点
            self.agent_pos = [1, 1]
            while self.grid[self.agent_pos[0], self.agent_pos[1]] == 1:
                self.agent_pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]

            self.goal_pos = [self.grid_size-2, self.grid_size-2]
            while self.goal_pos == self.agent_pos or self.grid[self.goal_pos[0], self.goal_pos[1]] == 1:
                self.goal_pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]

        # 生成陷阱
        self.trap_positions = []
        for _ in range(self.num_traps):
            pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]
            while pos in [self.agent_pos, self.goal_pos] or self.grid[pos[0], pos[1]] == 1:
                pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]
            self.trap_positions.append(pos)

        # 生成coins
        self.coin_positions = []
        for _ in range(self.num_coins):
            pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]
            while pos in [self.agent_pos, self.goal_pos] + self.trap_positions or self.grid[pos[0], pos[1]] == 1:
                pos = [random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)]
            self.coin_positions.append(pos)

        self.done = False
        self.steps = 0
        return self.get_state()

    def is_connected(self):
        # BFS 检查代理到终点是否连通
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = deque([tuple(self.agent_pos)])
        visited[self.agent_pos[0], self.agent_pos[1]] = True

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if [x, y] == self.goal_pos:
                return True
            for dx, dy in deltas:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and not visited[nx, ny] and self.grid[nx, ny] != 1:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
        return False

    def step(self, action):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        dx, dy = deltas[action]
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        reward = -0.02  # 步罚减小

        collision = False
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and self.grid[new_x, new_y] != 1:
            prev_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            self.agent_pos = [new_x, new_y]
            new_dist = abs(new_x - self.goal_pos[0]) + abs(new_y - self.goal_pos[1])
            if new_dist < prev_dist:
                reward += 2.0  # 靠近奖励加大
            elif new_dist > prev_dist:
                reward -= 0.5  # 远离罚减小

            if self.agent_pos in self.coin_positions:
                reward += 30  # coins奖励加大
                self.coin_positions.remove(self.agent_pos)

            if self.agent_pos in self.trap_positions:
                reward -= 30  # 罚减小，但仍结束
                collision = True
                self.done = True
        else:
            reward -= 2.0  # 墙罚
            collision = True

        trap_penalty = 0
        for trap in self.trap_positions:
            dist_to_trap = abs(self.agent_pos[0] - trap[0]) + abs(self.agent_pos[1] - trap[1])
            if dist_to_trap < 3:
                trap_penalty -= 1 / (dist_to_trap + 0.1)  # 附近罚减小
        reward += trap_penalty

        if self.agent_pos == self.goal_pos:
            reward += 300  # 终点加大
            self.done = True

        # 障碍移动：加概率，减慢速度
        for i in range(len(self.trap_positions)):
            if random.random() < 0.2:  # 40% 几率移动，调低让更慢
                trap_action = random.randint(0, 4)
                tx, ty = deltas[trap_action]
                new_tx, new_ty = self.trap_positions[i][0] + tx, self.trap_positions[i][1] + ty
                if 0 < new_tx < self.grid_size-1 and 0 < new_ty < self.grid_size-1 and self.grid[new_tx, new_ty] != 1:
                    self.trap_positions[i] = [new_tx, new_ty]

        self.steps += 1
        if self.steps > 500:  # 超时延长
            self.done = True
            reward -= 10  # 超时罚减小

        return self.get_state(), reward, self.done

    def get_state(self):
        state = self.grid.flatten().copy()
        agent_idx = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        goal_idx = self.goal_pos[0] * self.grid_size + self.goal_pos[1]
        state[agent_idx] = 2
        state[goal_idx] = 3
        for trap in self.trap_positions:
            state[trap[0] * self.grid_size + trap[1]] = 4
        for coin in self.coin_positions:
            state[coin[0] * self.grid_size + coin[1]] = 5
        return state.astype(np.float32)

    def render(self, screen, cell_size=50, show_q_values=None, collision=False):
        screen.fill((255, 255, 255))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                if self.grid[i, j] == 1:
                    pygame.draw.rect(screen, (50, 50, 50), rect)  # 深灰墙
                else:
                    pygame.draw.rect(screen, (220, 220, 220), rect, 1)  # 浅灰网格

        # Agent: 蓝圈，碰撞闪红
        agent_color = (255, 0, 0) if collision else (0, 0, 255)
        pygame.draw.circle(screen, agent_color,
                           ((self.agent_pos[1] + 0.5) * cell_size, (self.agent_pos[0] + 0.5) * cell_size),
                           cell_size // 3)

        # Goal: 绿方块，脉动动画
        goal_size = cell_size if self.steps % 20 < 10 else cell_size * 0.9
        goal_rect = pygame.Rect(self.goal_pos[1] * cell_size + (cell_size - goal_size)/2,
                                self.goal_pos[0] * cell_size + (cell_size - goal_size)/2, goal_size, goal_size)
        pygame.draw.rect(screen, (0, 255, 0), goal_rect)

        # Traps: 红三角，旋转动画
        for trap in self.trap_positions:
            points = [
                ((trap[1] + 0.5) * cell_size, trap[0] * cell_size),
                ((trap[1]) * cell_size, (trap[0] + 1) * cell_size),
                ((trap[1] + 1) * cell_size, (trap[0] + 1) * cell_size)
            ]
            pygame.draw.polygon(screen, (255, 0, 0), points)

        # Coins: 黄圈，闪烁
        for coin in self.coin_positions:
            coin_radius = cell_size // 4 if self.steps % 10 < 5 else cell_size // 5
            pygame.draw.circle(screen, (255, 255, 0),
                               ((coin[1] + 0.5) * cell_size, (coin[0] + 0.5) * cell_size),
                               coin_radius)

        # Q值热图 (如果show_q_values)
        if show_q_values is not None and len(show_q_values) > 0:
            q_array = np.array(show_q_values)  # 确保是 numpy 数组
            max_q = np.max(q_array) if q_array.size > 0 else 1.0
            min_q = np.min(q_array) if q_array.size > 0 else 0.0
            q_range = max_q - min_q if max_q > min_q else 1.0

            for a in range(len(q_array)):
                norm = (q_array[a] - min_q) / q_range
                green = int(255 * norm)
                green = max(0, min(255, green))  # 强制范围
                color = (0, green, 0)
                pygame.draw.rect(screen, color, (10 + a*50, 460, 40, 20))  # 底部热图条

        pygame.display.flip()