# Authors: Alex Kusek, Athziri Garcia, Shayla Salamon Ruiz, Ayushi Maurya
# Date: 24 October 2024
# Description: This code implements a bot that chooses an adjacent column to place the disk in.
# The code used was from Alex Kusek's nstep-heuristic.py file which included code taken from 
# Alexis Cook's One-Step Lookahead and N-Step Lookahead tutorials

from kaggle_environments import make, evaluate
import random
import numpy as np

# Global variable to store the previous move made by the agent
previous_move = None

# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# Helper function for minimax_alpha_beta: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
    num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
    score = num_threes - 1e2 * num_threes_opp - 1e4 * num_fours_opp + 1e6 * num_fours
    return score

# Drop a piece in a given column
def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = piece
            break
    return next_grid

# Helper function for minimax_alpha_beta: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

# Helper function for minimax_alpha_beta: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False

# Minimax_alpha_beta Implementation
def minimax_alpha_beta(node, depth, alpha, beta, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)
    
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax_alpha_beta(child, depth - 1, alpha, beta, False, mark, config))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cut-off
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark % 2 + 1, config)
            value = min(value, minimax_alpha_beta(child, depth - 1, alpha, beta, True, mark, config))
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cut-off
        return value

# Uses minimax_alpha_beta to calculate value of dropping piece in selected column
def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax_alpha_beta(next_grid, nsteps - 1, -np.Inf, np.Inf, False, mark, config)
    return score

N_STEPS = 3

def agent(obs, config):
    global previous_move
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    
    # If this is the first move, pick a random valid column
    if previous_move is None:
        move = random.choice(valid_moves)
    else:
        # Try to place the piece in an adjacent column if possible
        left_move = previous_move - 1 if previous_move - 1 in valid_moves else None
        right_move = previous_move + 1 if previous_move + 1 in valid_moves else None
        
        # Prioritize left or right move if available
        if left_move is not None:
            move = left_move
        elif right_move is not None:
            move = right_move
        else:
            # If no adjacent moves are valid, choose the move with the highest heuristic score
            scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
            max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
            move = random.choice(max_cols)
    
    # Update previous move with the current move
    previous_move = move
    
    return move

# Create the game environment
env = make("connectx")

# Two agents play one game round (one using the adjacent strategy, the other random)
env.run([agent, "random"])

# Show the game
env.render(mode="ipython")
