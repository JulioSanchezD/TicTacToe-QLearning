# Tic-Tac-Toe Reinforcement Learning

This project implements a Tic-Tac-Toe environment and trains an AI agent using Q-learning. The agent learns optimal strategies through self-play and can compete against a human player.

## Features
- A `TicTacToeEnv` class to simulate the game.
- A Q-learning training script to optimize the AI agent.
- A play script allowing human users to compete against the trained agent.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies.

```bash
pip install numpy pandas tqdm
```

## Usage
### Train the Model
Run the `train.py` script to train the AI using Q-learning.

```bash
python train.py
```
This will generate a `q_table.csv` file storing the learned Q-values.

### Play Against the AI
After training, you can compete against the AI using `play.py`.

```bash
python play.py
```
You'll be prompted to enter your moves, and the AI will respond based on its learned strategy.

## Code Overview
### `environment.py`
Defines the `TicTacToeEnv` class, which provides:
- Board management
- Move validation
- Game state updates
- Random opponent moves during training

### `train.py`
- Implements Q-learning for training the agent.
- Saves the trained Q-table as `q_table.csv`.

### `play.py`
- Loads the trained Q-table.
- Allows human players to compete against the AI.

## Future Improvements
- Improve the AI’s exploration strategy.

## License
This project is released under the MIT License.

