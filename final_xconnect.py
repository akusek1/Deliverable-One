import numpy as np
from kaggle_environments import make
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Neural Network for board evaluation
class BoardEvaluator(nn.Module):
    def __init__(self, rows, cols):
        super(BoardEvaluator, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output is between -1 and 1 for evaluation
        )

    def forward(self, x):
        return self.network(x)


# Replay buffer for storing experiences
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, value):
        self.buffer.append((state, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, values = zip(*batch)
        return torch.FloatTensor(states), torch.FloatTensor(values)

    def __len__(self):
        return len(self.buffer)


# Self-play game for generating training data
def self_play_game(evaluator, config):
    env = make("connectx")
    board = np.zeros((config.rows, config.columns))

    while not is_terminal_node(board, config):
        valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
        if not valid_moves:
            break

        # Use minimax with evaluator for move selection
        best_move = None
        best_value = float('-inf')

        for move in valid_moves:
            next_board = drop_piece(board.copy(), move, 1, config)
            value = minimax_with_evaluator(next_board, 3, float('-inf'), float('inf'),
                                           False, 1, config, evaluator)
            if value > best_value:
                best_value = value
                best_move = move

        board = drop_piece(board, best_move, 1, config)

        if is_terminal_node(board, config):
            return board, 1.0  # Win

        # Opponent's turn
        valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
        if not valid_moves:
            return board, 0.0  # Draw

        opp_move = random.choice(valid_moves)
        board = drop_piece(board, opp_move, 2, config)

        if is_terminal_node(board, config):
            return board, -1.0  # Loss

    return board, 0.0  # Draw


# Train the board evaluator using self-play data
def train_evaluator(config, n_episodes=1000):
    evaluator = BoardEvaluator(config.rows, config.columns)
    optimizer = optim.Adam(evaluator.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()

    for episode in range(n_episodes):
        final_board, outcome = self_play_game(evaluator, config)
        replay_buffer.push(final_board, outcome)

        if len(replay_buffer) >= 32:
            states, values = replay_buffer.sample(32)
            optimizer.zero_grad()
            predictions = evaluator(states)
            loss = nn.MSELoss()(predictions.squeeze(), values)
            loss.backward()
            optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Buffer size: {len(replay_buffer)}")

    return evaluator


# Minimax algorithm with evaluator and alpha-beta pruning
def minimax_with_evaluator(node, depth, alpha, beta, maximizingPlayer, mark, config, evaluator):
    if depth == 0 or is_terminal_node(node, config):
        with torch.no_grad():
            return evaluator(torch.FloatTensor(node).unsqueeze(0)).item()

    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

    if maximizingPlayer:
        value = float('-inf')
        for col in valid_moves:
            child = drop_piece(node.copy(), col, mark, config)
            value = max(value, minimax_with_evaluator(child, depth - 1, alpha, beta,
                                                      False, mark, config, evaluator))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for col in valid_moves:
            child = drop_piece(node.copy(), col, mark % 2 + 1, config)
            value = min(value, minimax_with_evaluator(child, depth - 1, alpha, beta,
                                                      True, mark, config, evaluator))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


# Helper functions
def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = piece
            break
    return next_grid


def is_terminal_node(grid, config):
    # Check for draw
    if list(grid[0, :]).count(0) == 0:
        return True

    # Check for win
    for check_func in [check_horizontal, check_vertical, check_diagonal]:
        if check_func(grid, config):
            return True
    return False


def check_horizontal(grid, config):
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_terminal_window(window, config):
                return True
    return False


def check_vertical(grid, config):
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_terminal_window(window, config):
                return True
    return False


def check_diagonal(grid, config):
    # Positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True

    # Negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False


def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow


# Score a move using minimax
def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    return minimax_alpha_beta(next_grid, nsteps - 1, -np.Inf, np.Inf, False, mark, config)


# Minimax with alpha-beta pruning
def minimax_alpha_beta(node, depth, alpha, beta, maximizingPlayer, mark, config):
    if depth == 0 or is_terminal_node(node, config):
        return get_heuristic(node, mark, config)

    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    ordered_moves = order_moves(node, valid_moves, mark if maximizingPlayer else mark % 2 + 1, config)

    if maximizingPlayer:
        value = -np.Inf
        for col in ordered_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax_alpha_beta(child, depth - 1, alpha, beta, False, mark, config))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.Inf
        for col in ordered_moves:
            child = drop_piece(node, col, mark % 2 + 1, config)
            value = min(value, minimax_alpha_beta(child, depth - 1, alpha, beta, True, mark, config))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


# Heuristic for board evaluation
def get_heuristic(grid, mark, config):
    num_fours = count_windows(grid, 4, mark, config)
    num_threes = count_windows(grid, 3, mark, config)
    num_twos = count_windows(grid, 2, mark, config)

    num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
    num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)

    center_array = grid[:, config.columns // 2]
    center_count = sum(center_array == mark)
    center_count_opp = sum(center_array == (mark % 2 + 1))

    return (num_fours * 1e6 - num_fours_opp * 1e5 +
            num_threes * 1e3 - num_threes_opp * 1e4 +
            num_twos * 10 - num_twos_opp * 8 +
            center_count * 50 - center_count_opp * 40)


def count_windows(grid, num_discs, piece, config):
    num_windows = 0

    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1

    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1

    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1

    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1

    return num_windows


def check_window(window, num_discs, piece, config):
    return window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs


def order_moves(grid, valid_moves, mark, config):
    move_scores = []

    for col in valid_moves:
        next_grid = drop_piece(grid, col, mark, config)
        score = get_heuristic(next_grid, mark, config)
        move_scores.append((-abs(col - config.columns // 2), score, col))

    return [col for _, _, col in sorted(move_scores, reverse=True)]

# Agent function for the game
def agent(obs, config):
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    n_steps = 4 if len(valid_moves) <= 6 else 3

    scores = {col: score_move(grid, col, obs.mark, config, n_steps) for col in valid_moves}

    max_score = max(scores.values())
    best_moves = [key for key in scores.keys() if scores[key] == max_score]

    center_moves = [move for move in best_moves if abs(move - config.columns // 2) <= 1]
    if center_moves:
        return random.choice(center_moves)

    return random.choice(best_moves)


# Run the ConnectX game environment
env = make("connectx")
env.run([agent, "random"])
env.render(mode="ipython")