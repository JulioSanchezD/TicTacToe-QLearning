import ast
import pandas as pd
from environment import TicTacToeEnv
from train import greedy_policy

env = TicTacToeEnv(train=False)
q_table = pd.read_csv('q_table.csv', index_col=0)


# Ask user who starts
turn = input("Do you want to start? (y/n): ").strip().lower()
if turn == "y":
    turn = "X"
else:
    turn = "0"

finished = False
state = env.get_state()
while not finished:
    if turn == "X":  # User's turn
        x, y = map(int, input("Enter your move (row, col): ").split())
        action = (x, y)
    else:  # Agent's turn
        action = ast.literal_eval(greedy_policy(q_table, state))  # Select best move from Q-table

    # Play the action
    state, score, finished = env.step(turn, action)
    if score == 2:
        print("It's a draw!")
    elif score == 10:
        if turn == "X":
            print("You won!")
        else:
            print("You lost!")

    # Swap turn
    turn = "0" if turn == "X" else "X"

print("Game Over!")