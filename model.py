
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
This file contains the implementation of a reinforcement learning model for snake using Q-learning.

Q-learning is a model-free reinforcement learning algorithm that learns an optimal policy by
estimating the value of each state-action pair, known as the Q-value. The Q-value represents
the expected cumulative reward that an agent will receive by taking a particular action in a
given state and following the optimal policy thereafter.

The Q-value iteration algorithm is used to update the Q-values based on the Bellman equation,
which states that the optimal Q-value for a state-action pair is equal to the immediate reward
plus the discounted maximum Q-value of the next state. The discount factor determines the
importance of future rewards compared to immediate rewards.

The loss function used in this implementation is the mean squared error (MSE) loss, which
measures the difference between the predicted Q-values and the target Q-values. The optimizer
used is Adam, which is a popular optimization algorithm for training neural networks.

The Reinforcement_QNetwork class represents the neural network model used for Q-learning. It
consists of an input layer, a hidden layer, and an output layer. The feed_forward method performs
a feed-forward pass through the neural network.

The QValueIterTrainer class is a trainer class for Q-value iteration. It takes the Q-network model,
learning rate, and discount rate gamma factor as input. The train_using_Q method trains the model
using Q-value iteration, updating the Q-values based on the Bellman equation and minimizing the
MSE loss using backpropagation.
"""


class Reinforcement_QNetwork(nn.Module):
    """
    A neural network model for reinforcement learning.

    Args:
        input_size (int): The size of the input layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.neuralNetLayer1 = nn.Linear(input_size, hidden_size)  # input layer
        self.neuralNetLayer2 = nn.Linear(hidden_size, output_size)  # output layer

    def forward(self, x):
        """
        Perform a feed-forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = F.relu(self.neuralNetLayer1(x))  # activation function
        x = self.neuralNetLayer2(x)
        return x

    def saveModel(self, file_name='model.pth'):
        """
        Save the model to a file.

        Args:
            file_name (str, optional): The name of the file to save the model to. Defaults to 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.saveModel(self.state_dict(), file_name)


class QValueIterTrainer:
    """
    A trainer class for Q-value iteration.

    Args:
        model (Reinforcement_QNetwork): The Q-network model.
        lr (float): The learning rate for the optimizer.
        discount_rate_gamma_factor (float): The discount rate gamma factor for calculating new Q values.
    """

    def __init__(self, model, lr, discount_rate_gamma_factor):
        self.lr = lr
        self.discount_rate_gamma_factor = discount_rate_gamma_factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # optimizer
        self.criterion = nn.MSELoss()  # loss function

    def train_using_Q(self, state, action, reward, next_state, game_over):
        """
        Train the model using Q-value iteration.

        Args:
            state (list or torch.Tensor): The current state.
            action (list or torch.Tensor): The action taken in the current state.
            reward (list or torch.Tensor): The reward received for taking the action.
            next_state (list or torch.Tensor): The next state.
            game_over (bool or list): Whether the game is over or not.

        Returns:
            None
        """
        state = torch.tensor(state, dtype=torch.float)  # convert to tensor
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)  # convert to tensor
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)  # tuple

        pred = self.model(state)

        target = pred.clone()  # clone the predicted value

        # value iteration algorithm for calculating new Q value
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.discount_rate_gamma_factor * torch.max(self.model(next_state[idx]))  # Q learning formula
            target[idx][torch.argmax(action).item()] = Q_new  # update the target

        self.optimizer.zero_grad()  # set the gradient to zero
        loss = self.criterion(target, pred)  # calculate the loss
        loss.backward()  # back propagation
        self.optimizer.step()  # update the weights
