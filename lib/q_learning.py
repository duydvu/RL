import numpy as np
import random


from .rl_model import RLModel


class QLeaning(RLModel):
    def __init__(self, action_size: int, state_size: int):
        self.qtable = np.zeros((state_size, action_size))

    def get_action(self, state: int):
        return np.argmax(self.qtable[state, :])
    
    def update(self, state: int, action: int, new_state: int, reward: float, learning_rate: float, gamma: float):
        self.qtable[state, action] = self.qtable[state, action] + \
            learning_rate * (reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])
