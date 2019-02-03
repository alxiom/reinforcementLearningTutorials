import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


def one_hot(hot, n):
    return np.identity(n)[hot:hot + 1]


env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1 * output_size  # since "mean" squared error, enhance rate to recover the division by mean


class QNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.uniform_(self.linear.weight, 0.0, 0.01)

    def forward(self, inputs):
        return self.linear(inputs)


q_net = QNet(input_size, output_size)
q_net.train()

optimizer = torch.optim.SGD(q_net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

gamma = 0.99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1.0 / ((i / 50) + 10)
    rAll = 0
    done = False

    while not done:
        x = torch.Tensor(one_hot(s, input_size)).float()
        Qs = q_net(x).data.numpy()

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = reward
        else:
            x1 = torch.Tensor(one_hot(s1, input_size)).float()
            Qs1 = q_net(x1).data.numpy()
            Qs[0, a] = reward + gamma * np.max(Qs1)

        loss = criterion(q_net(x), torch.Tensor(Qs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        s = s1

    rList.append(rAll)

print("Success Rate : {}".format(str(sum(rList) / num_episodes)))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
