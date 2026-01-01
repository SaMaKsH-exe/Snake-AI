import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGame, Direction, Point


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.game = 0
        self.memory= deque(maxlen=MAX_MEMORY)
        
        

    def get_state(self, game):
        pass


    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

def train():
    pass

if __name__ == '__main__':
    train()


