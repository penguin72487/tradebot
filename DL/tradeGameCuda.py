import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TradingEnvironment(gym.Env):
    def __init__(self, price_data, initial_equity=1000.0, max_position=1.0):
        super(TradingEnvironment, self).__init__()
        self.price_data = price_data
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-max_position, high=max_position, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(price_data[0]),), dtype=np.float32)
        self.position = 0.0  # The number of units held, based on current equity and price.
        self.equity = initial_equity
        self.initial_equity = initial_equity

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.equity = self.initial_equity
        return self.price_data[self.current_step]

    def step(self, action):
        prev_price = self.price_data[self.current_step][0]
        self.current_step += 1

        if self.current_step >= len(self.price_data):
            done = True
            reward = self.equity
            return None, reward, done, {}
        else:
            done = False
            curr_price = self.price_data[self.current_step][0]

            # Update position based on action, calculate new position based on percentage of equity.
            action_percentage = np.clip(action[0], -1.0, 1.0)  # Ensure action is within [-1, 1]
            target_position_value = action_percentage * self.equity  # Target position in dollar value
            target_position_units = target_position_value / curr_price  # Convert to number of units

            # Calculate profit or loss
            profit = target_position_units * (curr_price - prev_price)
            self.equity += profit

            # Update the actual position to the target position
            self.position = target_position_units

            # Reward is based on the updated equity
            reward = self.equity

        return self.price_data[self.current_step], reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Equity: {self.equity}, Position (units): {self.position}, HoleEquity: {self.price_data[self.current_step][0] * 1000 / self.price_data[0][0]}')

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_dim)
        self.fc_v = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x
    
    def policy(self, x):
        x = self.forward(x)
        action_probs = torch.tanh(self.fc_pi(x))  # Output in range [-1, 1] representing percent of equity.
        return action_probs
    
    def value(self, x):
        x = self.forward(x)
        value = self.fc_v(x)
        return value

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, hidden_size=128, epochs=10):
        self.actor_critic = ActorCritic(input_dim, action_dim, hidden_size=hidden_size).to(device)
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_probs = self.actor_critic.policy(state)
        return action_probs.detach().cpu().numpy()[0]
    
    def train(self, trajectory):
        states, actions, rewards, next_states, dones = zip(*trajectory)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        
        # Calculate values and TD targets
        values = self.actor_critic.value(states).view(-1)  # Adjust shape by flattening the tensor
        next_values = self.actor_critic.value(next_states).view(-1)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Calculate the difference between target and current value estimation
        advantages = (td_targets - values).detach()
        
        # Update policy and value networks
        for _ in range(self.epochs):  # multiple epochs
            action_probs = self.actor_critic.policy(states)
            old_action_probs = action_probs.detach()
            ratio = (action_probs / (old_action_probs + 1e-8))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss
            value_loss = nn.MSELoss()(values, td_targets.detach())

            # Total loss
            loss = policy_loss + value_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

if __name__ == '__main__':
    # Enable anomaly detection to find the operation that failed to compute its gradient
    torch.autograd.set_detect_anomaly(True)
    
    # Load configuration from config.json
    with open('C:\\gitproject\\tradebot\\ML\\btcLSTMmore\\configlite.json', 'r') as f:
        config = json.load(f)

    # Load environment parameters
    initial_equity = config['initial_equity']
    max_position = config['max_position']

    # Load agent parameters
    lr = config['agent']['lr']
    gamma = config['agent']['gamma']
    clip_ratio = config['agent']['clip_ratio']
    hidden_size = config['agent']['hidden_size']
    epochs = config['agent']['epochs']

    # Load training parameters
    num_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']

    # Load the dataset
    file_path = config['file_path']
    df = pd.read_csv(file_path)
    features = ['close']
    price_data = df[features].values

    # Initialize the environment
    env = TradingEnvironment(price_data, initial_equity=initial_equity, max_position=max_position)

    # Initialize the agent
    agent = PPOAgent(input_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], lr=lr, gamma=gamma, clip_ratio=clip_ratio, hidden_size=hidden_size, epochs=epochs)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        step_count = 0

        while not done and step_count < len(price_data) - 1:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step([action])
            
            if next_state is not None:
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
            
            step_count += 1
            if step_count % 100 == 0:
                env.render()

        # Train agent after each episode
        agent.train(trajectory)
