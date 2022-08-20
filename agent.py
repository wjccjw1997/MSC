import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = input_shape[0]
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.Conv2d(512, num_actions, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.network(x)
        return x

class Agent:
    def __init__(self, algorithm, env, learning_rate, gamma, device):
        self.algorithm = algorithm
        self.env = env
        self.online_model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
        self.target_model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.best_reward = 0
        self.reward_tracker = []
        self.loss_tracker = []
        model_save_path = "saved/"
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        self.root = model_save_path

    def update(self, S, S_, R, done, A):
        if self.algorithm == "DQN":
            with torch.no_grad():
                Q = self.target_model(S_).max(1)[0].squeeze()
                target = R + self.gamma * Q * (1 - done)
        elif self.algorithm == "doubleDQN":
            with torch.no_grad():
                V = self.target_model(S_)
                A_d = self.online_model(S_).max(1)[1].squeeze()
                Q = V.squeeze().gather(1, A_d.unsqueeze(1)).squeeze()
                target = R + self.gamma * Q * (1 - done)
        y = self.online_model(S).squeeze().gather(1, A.unsqueeze(1)).squeeze()
        loss = (y - target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_action(self, state, epsilon):
        if np.random.uniform() > epsilon:
            with torch.no_grad():
                q_value = self.online_model(state)
                action = q_value.max(1)[1].data[0]
        else:
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def backup(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save(self):
        f_name = self.root + self.algorithm + "data.csv"
        with open(f_name, "w") as fp:
            fp.write(self.algorithm + ", rewards, " + str(self.reward_tracker[0]))
            for r in self.reward_tracker[1:]:
                fp.write(", " + str(r))
            fp.write("\n")
            fp.write(self.algorithm + ", loss, " + str(self.loss_tracker[0]))
            for l in self.loss_tracker[1:]:
                fp.write(", " + str(l))