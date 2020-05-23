import gym
import random
import numpy as np
import tqdm

from lib.q_learning import QLeaning

env = gym.make("FrozenLake8x8-v0")

agent = QLeaning(action_size=env.action_space.n,
                 state_size=env.observation_space.n)

total_episodes = 100000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.0001            # Exponential decay rate for exploration prob

rewards = []

for episode in tqdm.tqdm(range(total_episodes)):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            # exploitation
            action = agent.get_action(state)
        else:
            # exploration
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        agent.update(state, action, new_state, reward, learning_rate, gamma)

        total_rewards += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

rewards = np.array(rewards)

np.save(open('tmp/rewards.npy', 'wb'), rewards)
