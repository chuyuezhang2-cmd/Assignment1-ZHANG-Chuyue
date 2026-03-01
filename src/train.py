from maze_env import MazeEnv
from dqn_agent import DQNAgent
import numpy as np

def train(episodes=800):
    env = MazeEnv()
    state_size = env.grid_size ** 2
    action_size = 5

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.10,
        buffer_size=20000,
        batch_size=128
    )

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            steps += 1

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {steps}")

        if (e + 1) % 100 == 0:
            agent.save(f"trained_model_ep{e+1}.pth")
            print(f"  → Model saved at episode {e+1}")

    agent.save("trained_model_final.pth")
    print("Training completed. Final model saved as trained_model_final.pth")

if __name__ == "__main__":
    train()