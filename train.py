import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from agent import DQNetwork



env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000

# Initialize replay memory
memory = deque(maxlen=memory_size)

# Create the Q-network and the optimizer
q_network = DQNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Explore: random action
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return np.argmax(q_network(state).numpy())  # Exploit: best action

def replay_experience():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.max(q_network(torch.FloatTensor(next_state)).detach().numpy())
        
        target_f = q_network(torch.FloatTensor(state))
        target_f = target_f.detach().numpy()
        target_f[action] = target

        q_network.train()
        optimizer.zero_grad()
        loss = loss_fn(q_network(torch.FloatTensor(state)), torch.FloatTensor(target_f))
        loss.backward()
        optimizer.step()

episodes = 500
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time in range(200):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Store experience in memory
        memory.append((state, action, reward, next_state, done))

        state = next_state

        # Train the agent
        replay_experience()

        if done:
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            break

    # Reduce epsilon (exploration) after each episode
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

