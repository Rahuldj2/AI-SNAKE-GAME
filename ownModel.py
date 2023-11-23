import numpy as np
import random
from ytQRF import SnakeGameAI, Direction

class QLearning:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Exploration
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation

    def update_q_value(self, state, action, reward, next_state):
        current_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                       self.learning_rate * (reward + self.gamma * max_next_q_value)
        self.q_table[state, action] = new_q_value

    def train(self, num_episodes, env):
        for episode in range(num_episodes):
            state = tuple(env.get_state())
            total_reward = 0

            while True:
                action = self.choose_action(state)
                reward, done, _ = env.play_step(action)
                next_state = tuple(env.get_state())
                self.update_q_value(state, action, reward, next_state)

                total_reward += reward
                state = next_state

                if done:
                    break

            # Decay epsilon over time for exploration-exploitation trade-off
            self.epsilon = max(0.1, self.epsilon * 0.99)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Create Snake game environment
snake_game = SnakeGameAI()

# Create Q-learning agent
state_size = 11  # Number of elements in the state representation
action_size = 3  # Number of possible actions (left, straight, right)
agent = QLearning(state_size=state_size, action_size=action_size, learning_rate=0.1, gamma=0.9, epsilon=1.0)

# Train the agent on the Snake game environment
agent.train(num_episodes=100, env=snake_game)
