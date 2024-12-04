import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from kaggle_environments import make, evaluate

class BoardEvaluator(nn.Module):
    def __init__(self):
        super(BoardEvaluator, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 64)  # Input layer (6 rows * 7 columns)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: predicted score (value of the board state)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to convert the board to a tensor
def board_to_tensor(board, mark):
    tensor = torch.tensor([(1 if cell == mark else -1 if cell != 0 else 0) for cell in board], dtype=torch.float)
    return tensor

# Self-play to generate training data
def self_play(agent, games=100):
    data = []
    env = make("connectx")
    
    for _ in range(games):
        env.reset()
        obs = env.reset()[0]["observation"]
        done = False
        board_states = []
        rewards = []
        
        while not done:
            # Current player's move
            mark = obs["mark"]
            board_state = board_to_tensor(obs["board"], mark)
            board_states.append(board_state)
            
            action = agent(obs, env.configuration)
            action_opponent = random.choice([c for c in range(env.configuration.columns) if obs["board"][c] == 0])

            # Step with both agents' actions (only unpack two values)
            obs, reward = env.step([action, action_opponent])

            # If reward is structured, extract the numeric value
            if isinstance(reward, dict):
                reward = reward['reward']  # Adjust this line based on the actual structure of `reward`
                
            rewards.append(reward)
            
            # Determine if the game is done (by checking if the board is full or if there's a winner)
            done = obs["status"] != "RUNNING"  # Use the status from the observation to check for game end
        
        # Final reward for the outcome of the game
        final_reward = rewards[-1] if rewards else 0
        for state in board_states:
            data.append((state, final_reward))
            
    return data

# Training the evaluation function
def train_evaluator(data, epochs=10, batch_size=32):
    model = BoardEvaluator()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            states, rewards = zip(*batch)
            states = torch.stack(states)
            rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, rewards)
            loss.backward()
            optimizer.step()
            
    return model

# Use learned evaluation function in the minimax
def learned_minimax(grid, col, model, depth, alpha, beta, maximizingPlayer, mark, config):
    if depth == 0 or is_terminal_node(grid, config):
        board_tensor = board_to_tensor(grid.flatten(), mark)
        return model(board_tensor).item()
    
    valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(grid, col, mark, config)
            value = max(value, learned_minimax(child, col, model, depth - 1, alpha, beta, False, mark, config))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(grid, col, mark % 2 + 1, config)
            value = min(value, learned_minimax(child, col, model, depth - 1, alpha, beta, True, mark, config))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

# Updated agent function to use the learned evaluator
def agent_learned(obs, config, model):
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    n_steps = 3
    
    scores = {
        col: learned_minimax(grid, col, model, n_steps - 1, -np.Inf, np.Inf, True, obs.mark, config)
        for col in valid_moves
    }
    
    max_score = max(scores.values())
    best_moves = [col for col, score in scores.items() if score == max_score]
    return random.choice(best_moves)

# Collect data and train the model
data = self_play(agent)
model = train_evaluator(data)

# Use the trained model in the agent
env = make("connectx")
env.run([lambda obs, config: agent_learned(obs, config, model), "random"])
env.render(mode="ipython")
