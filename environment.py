import random
import numpy as np


class TicTacToeEnv:
    def __init__(self, train: bool):
        self.train = train
        self.reset()
        self.actions = [(row, col) for row in range(3) for col in range(3)]

    def reset(self):
        self.grid = np.full((3, 3), ' ')
        self.turn = None
        return self.get_state()

    def print_grid(self):
        if not self.train:
            for row in range(3):
                if row > 0:
                    print("---------")
                print(" | ".join(self.grid[row]))
            print()

    def get_state(self) -> str:
        return '|'.join([''.join(self.grid[row]) for row in range(3)])

    def check_winner(self):
        """Returns True if the given player has won."""
        for i in range(3):
            if np.all(self.grid[i, :] == self.turn) or np.all(self.grid[:, i] == self.turn):
                return True
        if np.all(np.diag(self.grid) == self.turn) or np.all(np.diag(np.fliplr(self.grid)) == self.turn):
            return True
        return False

    def validate_move(self, turn: str, row: int, col: int):
        if (
                turn == self.turn or
                not (0 <= row <= 2 and 0 <= col <= 2) or
                self.grid[row, col] != ' '
        ):
            return False
        return True


    def random_play(self, turn: str) -> str:
        actions = self.actions.copy()
        while True:
            row, col = random.sample(actions, 1)[0]
            if self.validate_move(turn, row, col):
                break
            else:
                actions.remove((row, col))
        self.turn = turn
        self.grid[row, col] = turn
        return self.get_state()

    def is_finished(self) -> tuple:
        # Check for win
        if self.check_winner():
            return 10, True

        # Check for draw
        if np.count_nonzero(self.grid != ' ') == 9:
            return 2, True

        # Continue game
        return 0, False

    def step(self, turn: str, action: tuple) -> tuple:
        row, col = action

        # Invalid move
        if not self.validate_move(turn, row, col):
            return self.get_state(), -10, True

        # Apply move
        self.turn = turn
        self.grid[row, col] = turn
        new_state = self.get_state()
        self.print_grid()

        # If terminated finish game
        score, terminated = self.is_finished()
        if terminated:
            return new_state, score, terminated

        if self.train:
            # Random move
            new_state = self.random_play('0' if turn == 'X' else 'X')
            self.print_grid()

            # Return score and status
            score, terminated = self.is_finished()
            return new_state, -score, terminated
        else:
            return new_state, score, terminated



if __name__ == "__main__":
    env = TicTacToeEnv(train=False)
    env.step('X', (0, 0))
    env.step('0', (0, 1))
    env.step('X', (1, 1))
    env.step('0', (2, 2))
    env.step('X', (0, 2))
    env.step('0', (1, 2))
    env.step('X', (2, 0)) # Win
    # env.step('X', (2, 1)) # Draw
    # env.step('0', (2, 0)) # Draw
    # env.step('X', (1, 0)) # Draw
