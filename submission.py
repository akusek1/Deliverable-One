from kaggle_environments import make
import random

# Create a custom Connect X environment with specified rows, columns, and win condition
def connect_x_env(rows, columns, inarow):
    # Custom configuration for the Connect X game
    config = {
        "rows": rows,        # Number of rows on the board
        "columns": columns,  # Number of columns on the board
        "inarow": inarow,    # Number of consecutive tokens needed to win
    }
    
    # Instantiating the game environment with the custom configuration
    env = make("connectx", configuration=config, debug=True)
    return env

# Selects a random valid column for the agent
def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)

# Specify the board dimensions and the 'connect x' value
rows = 8    # Set number of rows (Y)
columns = 11 # Set number of columns (X)
inarow = 5  # Connect N (how many in a row to win)

# Create the custom Connect N environment
env = connect_x_env(rows, columns, inarow)

# Run the game between two random agents
env.reset()
env.run([agent_random, agent_random])

# Render the game in Jupyter Notebook
env.render(mode="ipython")
