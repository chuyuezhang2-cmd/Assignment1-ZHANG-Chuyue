import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.10,
                 buffer_size=20000, batch_size=128, target_update=100):

        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)  # 新增：target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.target_update = target_update  # 每100步更新target
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]  # 用target计算
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()

    def get_q_values(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.model(state).detach().numpy().flatten()