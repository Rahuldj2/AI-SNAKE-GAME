import torch
import random
import numpy as np
from SnakeQlearning import SnakeReinforce
from collections import deque
from SnakeQlearning import Coordinate_obj
from SnakeQlearning import Orientation
from model import Reinforcement_QNetwork, QValueIterTrainer
import matplotlib.pyplot as plt


from helper import plot
MAX_ALLOWED_SIZE = 100_000
MEM_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.001 

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.discount_rate_gamma_factor = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_ALLOWED_SIZE)
        self.model = Reinforcement_QNetwork(11, 256, 3)#state has11 values
        self.trainer = QValueIterTrainer(self.model, lr=LEARNING_RATE, discount_rate_gamma_factor=self.discount_rate_gamma_factor)


    def retrieve_snake_state(self, game):
        head = game.snake[0]
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

    def memorize(self, state, action, reward, next_state, done):
        #store the state, action, reward, next_state, done in memory
        self.memory.append((state, action, reward, next_state, done))   

    
    def hone_retro_mind(self):
        if (len(self.memory)) > MEM_SAMPLE_SIZE:
            #randomly sample from memory
            snake_reward_memory = random.sample(self.memory, MEM_SAMPLE_SIZE)
        else:
            snake_reward_memory = self.memory
        
        states, actions, rewards, next_states, dones = zip(*snake_reward_memory)
        self.trainer.train_using_Q(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in snake_reward_memory:
        #     self.hone_antero_mind(state, action, reward, next_state, done)

    def hone_antero_mind(self, state, action, reward, next_state, done):
        self.trainer.train_using_Q(state, action, reward, next_state, done)


    def retrieve_snake_action(self, state):
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
    

def enhance_snake():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score_till_now = 0
    avg_scores_after_100 = [] 
    total_after_100=0
    agent = Agent()
    game = SnakeReinforce()
    while True:
        state_old = agent.retrieve_snake_state(game)
        final_move=agent.retrieve_snake_action(state_old)

        # perform new move and get new state
        reward, done, score = game.snake_play(final_move)
        state_new = agent.retrieve_snake_state(game)

        # train short memory
        agent.hone_antero_mind(state_old, final_move, reward, state_new, done)

        # memorize
        agent.memorize(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.hone_retro_mind()

            if score > high_score_till_now:
                high_score_till_now = score
                #agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', high_score_till_now)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            if agent.n_games >=100:
                total_after_100+=1
                avg_score_after_100 = total_score / total_after_100
                avg_scores_after_100.append(avg_score_after_100)
                total_score = 0
                # Plot new graph with fresh data after every 100 games
                plt.plot(avg_scores_after_100)
                plt.xlabel('Games')
                plt.ylabel('Average Score')
                plt.title('Average Score After Every 100 Games')
                if (agent.n_games==200):
                    plt.savefig(f'average_score_after_{agent.n_games}_games.png')
                
                # plt.show()
                plt.show()

if __name__ == '__main__':
    enhance_snake()
