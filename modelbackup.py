import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)#input layer
        self.linear2 = nn.Linear(hidden_size, output_size)#output layer

    def forward(self, x):
        x = F.relu(self.linear1(x))#activation function
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)#optimizer
        self.criterion = nn.MSELoss()#loss function

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)#convert to tensor
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)#convert to tensor
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:#if the shape is 1D
            #unsqueeze to make it 2D
            #(1,5) -> (1,1,5)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )#tuple

        #1: predicted Q values with current state
        pred = self.model(state)

        #2: Q_new = r + y * max(next_predicted Q value) -> only do this if not game over
        target = pred.clone()#clone the predicted value
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))#Q learning formula
            target[idx][torch.argmax(action).item()] = Q_new#update the target

        self.optimizer.zero_grad()#set the gradient to zero
        loss = self.criterion(target, pred)#calculate the loss
        loss.backward()#back propagation
        self.optimizer.step()#update the weights