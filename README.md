# Dynamic Maze Explorer

CDS524 Assignment 1 - Reinforcement Learning Game Design  
**Author**: ZHANG Chuyue  

## Project Overview

Dynamic Maze Explorer is a maze navigation game where an agent learns to reach the green goal while collecting yellow coins and avoiding moving red traps using Deep Q-Network (DQN), an advanced form of Q-Learning.

- **Objective**: Navigate the 10x10 maze to the goal, maximizing rewards.
- **Modes**: AI (autonomous DQN agent) and Manual (player control via arrow keys + Space to stay).
- **Controls**: Press 'M' to switch modes, 'P' to pause, 'R' to reset.
- **Lives**: Start with 6, lose one on trap collision or timeout.

## Technology Stack

- Python 3.9+
- PyTorch (DQN model and training)
- Pygame (game rendering and interaction)
- NumPy (grid and state handling)

## How to Run

### 1. Install Dependencies

```bash
pip install pygame torch numpy

## AI Mode Demo (Dynamic Gameplay)

<image-card alt="AI Mode Gameplay" src="demo.gif" ></image-card>

*AI agent navigating the maze, collecting yellow coins, avoiding red traps, and reaching the green goal.*
