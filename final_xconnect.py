import numpy as np
from kaggle_environments import make
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class BoardEvaluator(nn.Module):
    def __init__(self, rows, cols):
        super(BoardEvaluator, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1 for board evaluation
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

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

def self_play_game(evaluator, config):
    env = make("connectx")
    board = np.zeros((config.rows, config.columns))
    
    while not is_terminal_node(board, config):
        valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
        if not valid_moves:
            break
            
        # Use n-step lookahead with learned evaluation
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
        
        # Store position and outcome for training
        if is_terminal_node(board, config):
            return board, 1.0  # Win
        
        # Opponent's turn (simplified)
        valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
        if not valid_moves:
            return board, 0.0  # Draw
            
        opp_move = random.choice(valid_moves)
        board = drop_piece(board, opp_move, 2, config)
        
        if is_terminal_node(board, config):
            return board, -1.0  # Loss
    
    return board, 0.0  # Draw

def train_evaluator(config, n_episodes=1000):
    evaluator = BoardEvaluator(config.rows, config.columns)
    optimizer = optim.Adam(evaluator.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()
    
    for episode in range(n_episodes):
        final_board, outcome = self_play_game(evaluator, config)
        
        # Store experience
        replay_buffer.push(final_board, outcome)
        
        # Train on batch
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

def ml_agent(obs, config):
    # Load or train evaluator if not already done
    global trained_evaluator
    if not hasattr(ml_agent, 'trained_evaluator'):
        ml_agent.trained_evaluator = train_evaluator(config)
    
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    
    # Use minimax with learned evaluation function
    best_move = None
    best_value = float('-inf')
    
    for move in valid_moves:
        next_grid = drop_piece(grid.copy(), move, obs.mark, config)
        value = minimax_with_evaluator(next_grid, 3, float('-inf'), float('inf'), 
                                     False, obs.mark, config, ml_agent.trained_evaluator)
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_move

# Advanced heuristic function
def get_advanced_heuristic(grid, mark, config):
    def count_patterns(grid, num_discs, piece, config):
        return count_windows(grid, num_discs, piece, config)

    # Pattern counts for the current player and opponent
    my_twos = count_patterns(grid, 2, mark, config)
    my_threes = count_patterns(grid, 3, mark, config)
    my_fours = count_patterns(grid, 4, mark, config)
    opp_twos = count_patterns(grid, 2, mark % 2 + 1, config)
    opp_threes = count_patterns(grid, 3, mark % 2 + 1, config)
    opp_fours = count_patterns(grid, 4, mark % 2 + 1, config)

    # Center column preference
    center_array = grid[:, config.columns // 2]
    center_count = np.count_nonzero(center_array == mark)
    opp_center_count = np.count_nonzero(center_array == (mark % 2 + 1))

    # Fork detection
    my_forks = my_threes * 2 if my_threes > 1 else 0
    opp_forks = opp_threes * 2 if opp_threes > 1 else 0

    # Position-based scoring
    position_score = sum(
        [(1.0 - 0.2 * abs(c - config.columns // 2)) * (config.rows - r)
         for r in range(config.rows)
         for c in range(config.columns)
         if grid[r][c] == mark]
    )

    # Dynamic weights based on the game phase
    empty_cells = np.count_nonzero(grid == 0)
    early_game_weight = max(0, (config.rows * config.columns - empty_cells) / (config.rows * config.columns))
    late_game_weight = 1 - early_game_weight

    # Combine factors with dynamic weights
    heuristic_score = (
        position_score * 10 +
        center_count * 50 -
        opp_center_count * 40 +
        my_twos * 10 -
        opp_twos * 8 +
        my_threes * 1e3 * early_game_weight -
        opp_threes * 1e4 * late_game_weight +
        my_forks * 2e3 -
        opp_forks * 2e3 +
        my_fours * 1e6 -
        opp_fours * 1e5
    )

    return heuristic_score

# Helper functions for pattern detection
def check_window(window, num_discs, piece, config):
    return window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs

def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # Horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # Vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # Positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = [grid[row + i][col + i] for i in range(config.inarow)]
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # Negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = [grid[row - i][col + i] for i in range(config.inarow)]
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# Drop a piece into the grid
def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = piece
            break
    return next_grid

# Check for terminal states
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

def is_terminal_node(grid, config):
    if list(grid[0, :]).count(0) == 0:
        return True  # Draw
    for check_func in [check_horizontal, check_vertical, check_diagonal]:
        if check_func(grid, config):
            return True  # Win
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
            window = [grid[row + i][col + i] for i in range(config.inarow)]
            if is_terminal_window(window, config):
                return True
    # Negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = [grid[row - i][col + i] for i in range(config.inarow)]
            if is_terminal_window(window, config):
                return True
    return False

# Alpha-beta pruning with enhancements
def minimax_alpha_beta(node, depth, alpha, beta, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

    if depth == 0 or is_terminal:
        return get_advanced_heuristic(node, mark, config)

    if maximizingPlayer:
        value = float('-inf')
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax_alpha_beta(child, depth - 1, alpha, beta, False, mark, config))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for col in valid_moves:
            child = drop_piece(node, col, mark % 2 + 1, config)
            value = min(value, minimax_alpha_beta(child, depth - 1, alpha, beta, True, mark, config))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

# Agent function
def agent(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    best_move = None
    best_value = float('-inf')

    for move in valid_moves:
        next_grid = drop_piece(grid, move, obs.mark, config)
        value = minimax_alpha_beta(next_grid, 4, float('-inf'), float('inf'), False, obs.mark, config)
        if value > best_value:
            best_value = value
            best_move = move

    return best_move

# Create the game environment
env = make("connectx")

# Two agents play one game round (one using the adjacent strategy, the other random)
env.run([agent, "random"])

# Show the game
env.render(mode="ipython")