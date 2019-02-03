import gym
import numpy as np
import random
import torch
from torch import nn
from collections import deque

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
replay_memory = 50000
batch_size = 10
learning_rate = 1e-1 * output_size  # since "mean" squared error, enhance rate to recover the division by mean


class DQNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNet, self).__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, inputs):
        return self.linear(inputs)


def replay_train(net, train_batch):
    x_stack = np.empty(0).reshape(0, net.input_size)
    y_stack = np.empty(0).reshape(0, net.output_size)

    for state, action, reward, next_state, done in train_batch:
        x = torch.Tensor([state]).float()
        Qs = net(x).data.numpy()

        if done:
            Qs[0, action] = reward
        else:
            x1 = torch.Tensor([next_state]).float()
            Qs1 = net(x1).data.numpy()
            Qs[0, action] = reward + gamma * np.max(Qs1)

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Qs])

        loss = criterion(net(torch.Tensor(x_stack)), torch.Tensor(y_stack))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


hidden_size = 16
dq_net = DQNet(input_size, hidden_size, output_size)
dq_net.train()

optimizer = torch.optim.Adam(dq_net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

gamma = 0.99
num_episodes = 1000

replay_buffer = deque(maxlen=replay_memory)
rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1.0 / ((i / 10) + 1)
    rAll = 0
    step_count = 0
    done = False

    while not done:
        x = torch.Tensor([s]).float()
        Qs = dq_net(x).data.numpy()

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100

        replay_buffer.append((s, a, reward, s1, done))

        rAll += reward
        step_count += 1
        s = s1

        if rAll > 10000:  # good enough
            break

    print("episode = {}, step = {}, reward = {}".format(i, step_count, rAll))
    if step_count > 10000:
        pass

    if i % batch_size == 1:
        for _ in range(50):
            minibatch = random.sample(replay_buffer, batch_size)
            replay_train(dq_net, minibatch)

dq_net.eval()
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = torch.Tensor([observation]).float()
    Qs = dq_net(x).data.numpy()
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("total score = {}".format(reward_sum))
        break
