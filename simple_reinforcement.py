import numpy as np
from kaggle_environments import make
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 possible columns
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 6 * 7)
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

class ConnectFourRL:
    def __init__(self):
        self.model = SimpleNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=1000)
        
    def get_move(self, board, valid_moves):
        state = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.model(state).squeeze()
        
        # Filter only valid moves
        valid_probs = action_probs[valid_moves]
        move = valid_moves[torch.argmax(valid_probs).item()]
        return move
    
    def train_step(self, state, action, reward):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        self.optimizer.zero_grad()
        action_probs = self.model(state).squeeze()
        loss = -torch.log(action_probs[action]) * reward
        loss.backward()
        self.optimizer.step()

def create_board():
    return np.zeros((6, 7))

def get_valid_moves(board):
    return [col for col in range(7) if board[0][col] == 0]

def make_move(board, col, player):
    for row in range(5, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return board
    return board

def check_win(board, player):
    # Horizontal
    for row in range(6):
        for col in range(4):
            if all(board[row][col + i] == player for i in range(4)):
                return True
    
    # Vertical
    for row in range(3):
        for col in range(7):
            if all(board[row + i][col] == player for i in range(4)):
                return True
    
    # Diagonal (positive slope)
    for row in range(3):
        for col in range(4):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True
    
    # Diagonal (negative slope)
    for row in range(3, 6):
        for col in range(4):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True
    
    return False

def train_agent(episodes=1000):
    agent = ConnectFourRL()
    
    for episode in range(episodes):
        board = create_board()
        game_over = False
        
        while not game_over:
            # Agent's turn
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                break
                
            state_before = board.copy()
            move = agent.get_move(board, valid_moves)
            board = make_move(board, move, 1)
            
            # Check if agent won
            if check_win(board, 1):
                agent.train_step(state_before, move, 1.0)  # Positive reward for winning
                game_over = True
                continue
                
            # Random opponent's turn
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                break
                
            opp_move = random.choice(valid_moves)
            board = make_move(board, opp_move, 2)
            
            # Check if opponent won
            if check_win(board, 2):
                agent.train_step(state_before, move, -1.0)  # Negative reward for losing
                game_over = True
            else:
                agent.train_step(state_before, move, 0.1)  # Small positive reward for neutral moves
        
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    return agent

def create_competition_agent(trained_agent):
    def agent(obs, config):
        board = np.array(obs['board']).reshape(6, 7)
        valid_moves = [col for col in range(7) if obs['board'][col] == 0]
        if not valid_moves:
            return 0
        return trained_agent.get_move(board, valid_moves)
    return agent

# Train the agent
print("Starting training...")
trained_agent = train_agent(episodes=1000)
print("Training completed!")

# Create the competition agent
competition_agent = create_competition_agent(trained_agent)

# Test a single game
env = make("connectx")
env.run([competition_agent, "random"])
env.render(mode="ipython")