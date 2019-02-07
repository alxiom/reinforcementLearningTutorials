import gym
import numpy as np
import random
import torch
from torch import nn
from collections import deque

seed = 42
step_limit = 5000
env = gym.make('CartPole-v0')
env._max_episode_steps = step_limit
env.seed(seed)
torch.manual_seed(seed)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 64

batch_size = 64
training_interval = 10
replay_memory = 50000
learning_rate = 1e-1

gamma = 0.99
max_episodes = 5000
initial_epsilon = 1.0
epsilon_decay = 0.99

break_episodes = 4
break_reward = 4500


class DQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, inputs):
        return self.linear(inputs)


def replay_train(net, train_batch, criterion, optimizer):
    x_batch = []
    y_batch = []

    for state, action, reward, next_state, done in train_batch:
        x = torch.Tensor(state).float()
        q = net(x).data.numpy()

        if done:
            q[action] = reward
        else:
            next_x = torch.Tensor([next_state]).float()
            next_q = net(next_x).data.numpy()
            q[action] = reward + gamma * np.max(next_q)

        x_batch.append(state)
        y_batch.append(q)

    loss = criterion(net(torch.Tensor(x_batch).float()), torch.Tensor(y_batch).float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.numpy()


def simulate_bot(net):
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        x = torch.Tensor(state).float()
        q = net(x).data.numpy()
        action = np.argmax(q)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print(f"total score = {reward_sum}")
            break


def main():
    dqn = DQN(input_size, hidden_size, output_size)
    dqn.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=replay_memory)
    reward_history = deque(maxlen=break_episodes)

    for episode in range(max_episodes):
        state = env.reset()
        eps = initial_epsilon * np.power(epsilon_decay, episode)
        episode_reward = 0
        done = False

        while not done:
            if np.random.rand(1) < eps:
                action = env.action_space.sample()
            else:
                x = torch.Tensor(state).float()
                q = dqn(x).data.numpy()
                action = np.argmax(q)

            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100

            replay_buffer.append((state, action, reward, next_state, done))

            episode_reward += reward
            state = next_state

            if episode_reward >= break_reward:  # good enough
                break

        print(f"episode = {episode + 1}, rewards = {episode_reward}")

        reward_history.append(episode_reward)
        reward_avg = np.mean(reward_history)
        if reward_avg == break_reward:
            print(f"game cleared in {episode} episodes with avg step {reward_avg}")
            break

        if ((episode + 1) % training_interval == 0) & (len(replay_buffer) > batch_size * training_interval):
            for _ in range(training_interval):
                minibatch = random.sample(replay_buffer, batch_size)
                replay_train(dqn, minibatch, criterion, optimizer)

    dqn.eval()
    simulate_bot(dqn)


if __name__ == "__main__":
    main()
