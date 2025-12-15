"""
Deep Reinforcement Learning with DQN (CartPole)

Goal:
- Learn control policies through interaction with an environment

What this explores:
- Deep Q-Networks with experience replay
- Epsilon-greedy exploration and target networks
- Training an agent to balance CartPole via reward maximization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
import random
from matplotlib import animation
from IPython.display import display, clear_output

env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.reset()

num_steps_to_viz = 200
step_count = 0
for i in range(num_steps_to_viz):
   step_count += 1

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

from collections import namedtuple, deque

Transition = namedtuple(
    "Transition",
    ('state_a', 'action', 'state_b', 'reward')
)

t = Transition([0,0,0,0], 1, [1,1], 0.5)

class TransitionMemoryStorage():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add_transition(self, t):
        self.buffer.append(t)

    def sample(self, num_samples):
        return random.sample(self.buffer, num_samples)

    def can_sample(self, num_samples):
        return len(self.buffer) >= num_samples

import math

class EpsilonGreedyStrategy():
    def __init__(self, max_epsilon, min_epsilon, decay):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def _get_explore_prob(self, current_step):
        # exponential decay
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
               math.exp(-self.decay * current_step)

    def should_explore(self, current_step):
        epsilon = self._get_explore_prob(current_step)
        return random.random() < epsilon

class Agent():
    def __init__(self, strategy):
        self.strategy = strategy
        self.current_step = 0

    def select_action(self, input_state, policy_dqn):
        self.current_step += 1

        if self.strategy.should_explore(self.current_step):
            return random.randrange(2)   # CartPole action space = {0, 1}

        with torch.no_grad():
            q_values = policy_dqn(input_state)
            action = torch.argmax(q_values, dim=1).item()
            return action

from itertools import count

e_greedy_strategy = EpsilonGreedyStrategy(max_epsilon=0.9, min_epsilon=0.05, decay=0.99)
agent = Agent(strategy=e_greedy_strategy)
memory = TransitionMemoryStorage(1000)
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)
loss_func = nn.SmoothL1Loss()

BATCH_SIZE = 128
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Target network soft update factor

def optimize_model():
    if not memory.can_sample(BATCH_SIZE):
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state_b)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.state_b if s is not None])
    state_batch = torch.cat(batch.state_a)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    predicted_q_values = policy_net(state_batch).gather(1, action_batch)

    next_state_q_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad(): # This is needed because we don't want to pass in "None" values to our target_net as it would crash
        next_state_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_q_values = reward_batch + (next_state_q_values * GAMMA)

    loss = loss_func(predicted_q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target(policy_net, target_net):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

    target_net.load_state_dict(target_net_state_dict)

num_episodes = 500
for i_episode in range(num_episodes):
    if(i_episode % 25 == 0): # Sanity check every 25 episodes
        print(f"On episode {i_episode}")

    current_state, _ = env.reset()
    current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

    done = False
    while not done:
        action = agent.select_action(current_state, policy_net)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        memory.add_transition(Transition(current_state, torch.tensor([action]).unsqueeze(0), next_state, torch.tensor([reward])))
        current_state = next_state
        optimize_model()
        update_target(policy_net, target_net)
        done = terminated or truncated

print('Complete')

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

num_trials = 10000
overall_max_reward = 0

done = False
for i in range(num_trials):
    episode_reward = 0
    for t in count():
        action = agent.select_action(state, policy_net)
        state,reward,terminated,truncated,_ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = terminated or truncated

        episode_reward+=1
        overall_max_reward = max(episode_reward, overall_max_reward)

        if done:
            break

print(f"Longest time alive across {num_trials} trials: {overall_max_reward}")

env.reset()

step_count = 0
for t in count():
    step_count += 1
    if(t % 10 == 0):
        plt.imshow(env.render())
        display(plt.gcf())
        clear_output(wait=True)

    action = agent.select_action(state, policy_net)
    state,reward,terminated,truncated,_ = env.step(action)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    done = terminated or truncated
    if done:
        print(f"Survived for {step_count} steps!")
        if(step_count <= 50):
            print("Grade: F")
        elif((step_count > 50) and (step_count <= 100)):
            print("Grade: D")
        elif((step_count > 100) and (step_count <= 195)):
            print("Grade: C")
        elif((step_count > 195) and (step_count <= 450)):
            print("Grade: B")
        elif((step_count > 450) and (step_count <= 500)):
            print("Grade: A")
        break

env.close()
print("Complete!")
