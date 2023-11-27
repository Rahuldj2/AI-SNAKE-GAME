import torch
import random
import numpy as np
from ytQRF import SnakeReinforce
from collections import deque
from ytQRF import Coordinate_obj
from ytQRF import Orientation
from model import Linear_QNet, QTrainer

from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)#state has11 values
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]#grab the head from the snake
        coordinate_point_l = Coordinate_obj(head.x - 30, head.y)
        coordinate_point_r = Coordinate_obj(head.x + 30, head.y)
        coordinate_point_u = Coordinate_obj(head.x, head.y - 30)
        coordinate_point_d = Coordinate_obj(head.x, head.y + 30)

        direction_l = game.orientation == Orientation.WEST#left
        direction_r = game.orientation == Orientation.EAST#right
        direction_u = game.orientation == Orientation.NORTH#up
        direction_d = game.orientation == Orientation.SOUTH#down

        #total 11 states
        state = [
            # Danger straight
            (direction_r and game.get_collision_bool(coordinate_point_r)) or
            (direction_l and game.get_collision_bool(coordinate_point_l)) or
            (direction_u and game.get_collision_bool(coordinate_point_u)) or
            (direction_d and game.get_collision_bool(coordinate_point_d)),

            # Danger right
            (direction_u and game.get_collision_bool(coordinate_point_r)) or
            (direction_d and game.get_collision_bool(coordinate_point_l)) or
            (direction_l and game.get_collision_bool(coordinate_point_u)) or
            (direction_r and game.get_collision_bool(coordinate_point_d)),

            # Danger left
            (direction_d and game.get_collision_bool(coordinate_point_r)) or
            (direction_u and game.get_collision_bool(coordinate_point_l)) or
            (direction_r and game.get_collision_bool(coordinate_point_u)) or
            (direction_l and game.get_collision_bool(coordinate_point_d)),

            # Move orientation
            direction_l,
            direction_r,
            direction_u,
            direction_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        #store the state, action, reward, next_state, done in memory
        self.memory.append((state, action, reward, next_state, done))   

    
    def train_long_memory(self):
        if (len(self.memory)) > BATCH_SIZE:
            #randomly sample from memory
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #     self.train_short_memory(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation  
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeReinforce()
    while True:
        state_old = agent.get_state(game)
        final_move=agent.get_action(state_old)

        # perform new move and get new state
        reward, done, score = game.snake_play(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
