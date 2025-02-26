import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from environment import TicTacToeEnv

def add_row(df, idx):
    df = pd.concat(
        [
            df,
            pd.DataFrame(np.zeros((1, len(df.columns))), columns=df.columns, index=[idx])
        ]
    )
    return df

def greedy_policy(q_table, state):
    return q_table.loc[state].idxmax()

def epsilon_greedy_policy(q_table, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(q_table, state)
    return random.sample(env.actions, 1)[0]

if __name__ == '__main__':
    env = TicTacToeEnv(train=True)
    q_table = pd.DataFrame(columns=env.actions, dtype=float)
    q_table = add_row(q_table, env.get_state())
    
    # Training parameters
    n_training_episodes = 200000  # Total training episodes
    learning_rate = 0.2  # Learning rate

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.5  # Minimum exploration probability
    decay_rate = 0.0001  # Exponential decay rate for exploration prob
    gamma = 0.95  # Discounting rate

    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reset the environment
        state = env.reset()
        terminated = False
        turn = random.choice(['X', '0'])

        # Random first move is turn == 'X', simulating user is first
        if turn == 'X':
            state = env.random_play(turn)
            if state not in q_table.index:
                q_table = add_row(q_table, state)
            turn = '0'

        # repeat
        while not terminated:
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(q_table, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated = env.step(turn, action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state not in q_table.index:
                q_table = add_row(q_table, new_state)
            q_table.at[state, action] = q_table.loc[state][action] + learning_rate * (
                reward + gamma * np.max(q_table.loc[new_state]) - q_table.loc[state][action]
            )

            # Our next state is the new state
            state = new_state

    # Save trained Q-Table to a csv file
    q_table.to_csv('q_table.csv')
    print(f"{q_table.shape[0]} Combinations explored!")