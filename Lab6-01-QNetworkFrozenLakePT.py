import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


def one_hot(x, n):
    base = np.zeros(n, dtype=int)
    base[x] = 1
    return base


env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.01


class Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, inputs):
        return self.linear(inputs)


hidden_sizd = 2 * input_size
net = Net(input_size, hidden_sizd, output_size)
optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

gamma = 0.99
num_episodes = 2000

# for i in range(num_episodes):
s = env.reset()
# e = 1.0 / ((i / 50) + 10)
rAll = 0
done = False
local_loss = []

x = torch.Tensor(one_hot(s, input_size)).float()
print(net(x))



    # while not done:
    #     x = torch.Tensor()
    #     Qs0 = net.eval()
