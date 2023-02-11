import numpy as np
import torch
import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.action_space = np.arange(out_dim)

        # self.layer1 = nn.Linear(in_features=in_dim, out_features=64)
        # self.layer2 = nn.Linear(in_features=64, out_features=out_dim)

        self.layer1 = nn.Linear(in_features=in_dim, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        # self.layer3 = nn.Linear(in_features=32, out_features=32)
        # self.layer4 = nn.Linear(in_features=32, out_features=24)
        self.layer5 = nn.Linear(in_features=32, out_features=24)
        self.layer6 = nn.Linear(in_features=24, out_features=out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # activation1 = torch.relu(self.layer1(obs))
        # out = torch.softmax(self.layer2(activation1), dim=0)
        # return out

        activation1 = torch.relu(self.layer1(obs))
        activation2 = torch.relu(self.layer2(activation1))
        # activation3 = torch.relu(self.layer3(activation2))
        # activation4 = torch.relu(self.layer4(activation3))
        activation5 = torch.relu(self.layer5(activation2))
        output = torch.softmax(self.layer6(activation5), dim=0)
        return output

    def sample_action(self, obs):
        action_probs = self.forward(obs).detach().numpy()
        action = np.random.choice(self.action_space, p=action_probs)
        return action
