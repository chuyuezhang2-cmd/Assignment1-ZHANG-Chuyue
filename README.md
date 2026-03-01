# Dynamic Maze Explorer: A Maze Navigation Game Based on Deep Q-Learning
  
**Author**: ZHANG Chuyue  
**YouTube**:https://youtu.be/0_h6cyYHTCM

## AI Mode Demo (Dynamic Gameplay)

<div align="center">
  <img src="demo.gif" alt="AI Mode Gameplay" width="500">
  <p><em>AI agent navigating the maze in real-time (looping)</em></p>
</div>

## Project Overview

Dynamic Maze Explorer is a maze navigation game where an agent learns to reach the green goal while collecting yellow coins and avoiding moving red traps using Deep Q-Network (DQN), an advanced form of Q-Learning.

- **Objective**: Navigate the 10x10 maze to the goal, maximizing rewards.
- **Modes**: AI (autonomous DQN agent) and Manual (player control via arrow keys + Space to stay).
- **Controls**: Press 'M' to switch modes, 'P' to pause, 'R' to reset.
- **Lives**: Start with 6, lose one on trap collision or timeout.

## Technology Stack

- Python 3.9.7
- PyTorch (DQN model and training)
- Pygame (game rendering and interaction)
- NumPy (grid and state handling)

## How to Run

### 1. Install Dependencies

```bash
pip install pygame torch numpy


