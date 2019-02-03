import gym
import numpy as np
import torch
from torch import nn

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 1e-1 * output_size  # since "mean" squared error, enhance rate to recover the division by mean


class QNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)


q_net = QNet(input_size, output_size)
q_net.train()

optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

gamma = 0.99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1.0 / ((i / 10) + 1)
    rAll = 0
    step_count = 0
    done = False

    while not done:
        x = torch.Tensor([s]).float()
        Qs = q_net(x).data.numpy()

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100
        else:
            x1 = torch.Tensor([s1]).float()
            Qs1 = q_net(x1).data.numpy()
            Qs[0, a] = reward + gamma * np.max(Qs1)

        loss = criterion(q_net(x), torch.Tensor(Qs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        s = s1

    rList.append(rAll)
    print("episode = {}, step = {}".format(i, rAll))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

q_net.eval()
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = torch.Tensor([observation]).float()
    Qs = q_net(x).data.numpy()
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("total score = {}".format(reward_sum))
        break
