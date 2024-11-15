# Agent to select the best valid column
def alternative_agent(observation, configuration):
    import numpy as np

    # Constants
    HASH_MOD = int(1e9) + 7
    DEPTH_LIMIT = 4  # Adjust depth for reasonable computation time

    # Function to generate a hash for the board state
    def board_hash(state):
        idx = 0
        hash_value = 0
        for r in range(configuration.rows):
            for c in range(configuration.columns):
                if state[r][c] == 1:
                    hash_value = (hash_value + pow(5, idx, HASH_MOD)) % HASH_MOD
                elif state[r][c] == 2:
                    hash_value = (hash_value + pow(13, idx, HASH_MOD)) % HASH_MOD
                idx += 1
        return hash_value

    # Function to verify if a move is allowed
    def move_allowed(state, col):
        return state[0][col] == 0
    
    # Function to apply a move and return the resulting board state
    def apply_move(state, col, player):
        updated_state = state.copy()
        for r in range(configuration.rows - 1, -1, -1):
            if updated_state[r][col] == 0:
                updated_state[r][col] = player
                break
        return updated_state

    # Function to check if the game has reached a terminal state
    def game_ended(state):
        return not any(state[0][col] == 0 for col in range(configuration.columns))

    # Function to evaluate the board from the agent's perspective
    def board_score(state, player):
        return np.sum(state == player) - np.sum(state == (3 - player))

    # Minimax algorithm with depth limit
    def run_minimax(state, depth, player, maximizing):
        if depth == 0 or game_ended(state):
            return board_score(state, player)
        
        if maximizing:
            max_eval = -np.inf
            for col in range(configuration.columns):
                if move_allowed(state, col):
                    next_state = apply_move(state, col, player)
                    eval_score = run_minimax(next_state, depth - 1, player, False)
                    max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = np.inf
            for col in range(configuration.columns):
                if move_allowed(state, col):
                    next_state = apply_move(state, col, 3 - player)
                    eval_score = run_minimax(next_state, depth - 1, player, True)
                    min_eval = min(min_eval, eval_score)
            return min_eval

    # Determine the optimal move for the agent
    def optimal_move(state, player):
        top_score = -np.inf
        selected_column = -1
        for col in range(configuration.columns):
            if move_allowed(state, col):
                simulated_state = apply_move(state, col, player)
                score = run_minimax(simulated_state, DEPTH_LIMIT, player, False)
                if score > top_score:
                    top_score = score
                    selected_column = col
        return selected_column
    
    # Prepare the board and select the best move
    game_state = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    return optimal_move(game_state, observation.mark)
