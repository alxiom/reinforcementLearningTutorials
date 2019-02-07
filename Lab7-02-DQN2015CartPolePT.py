import gym
import numpy as np
import random
import torch
from torch import nn
from collections import deque

seed = 42

env = gym.make('CartPole-v0').unwrapped
env.seed(seed)

torch.manual_seed(seed)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 64

batch_size = 64
training_interval = 10
replay_memory = 50000
learning_rate = 1e-1
target_update_interval = 5

gamma = 0.99
max_episodes = 5000
half_eps_episode = max_episodes * 0.02

break_test_count = 4
break_test_avg = 5000
step_limit = 5000


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


def replay_train(main_net, target_net, train_batch, criterion, optimizer):
    x_batch = []
    y_batch = []

    for state, action, reward, next_state, done in train_batch:
        x = torch.Tensor(state).float()
        q = main_net(x).data.numpy()

        if done:
            q[action] = reward
        else:
            next_x = torch.Tensor([next_state]).float()
            next_q = target_net(next_x).data.numpy()
            q[action] = reward + gamma * np.max(next_q)

        x_batch.append(state)
        y_batch.append(q)

        loss = criterion(main_net(torch.Tensor(x_batch).float()), torch.Tensor(y_batch).float())
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
    main_dqn = DQN(input_size, hidden_size, output_size)
    main_dqn.train()
    target_dqn = DQN(input_size, hidden_size, output_size)
    target_dqn.eval()

    target_dqn.load_state_dict(main_dqn.state_dict())

    optimizer = torch.optim.Adam(main_dqn.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    replay_buffer = deque(maxlen=replay_memory)
    step_counts = deque(maxlen=break_test_count)

    for episode in range(max_episodes):
        state = env.reset()
        e = 1.0 / ((episode / half_eps_episode) + 1.0)
        rewards = 0
        step_count = 0
        done = False

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                x = torch.Tensor(state).float()
                q = main_dqn(x).data.numpy()
                action = np.argmax(q)

            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100

            replay_buffer.append((state, action, reward, next_state, done))

            rewards += reward
            step_count += 1
            state = next_state

            if step_count > step_limit:  # good enough
                break

        print(f"episode = {episode + 1}, rewards = {rewards}, steps = {step_count}")

        step_counts.append(step_count)
        step_avg = np.mean(step_counts)
        if step_avg > break_test_avg:
            print(f"game cleared in {episode} episodes with avg step {step_avg}")
            break

        if (episode + 1) % target_update_interval == 0:
            target_dqn.load_state_dict(main_dqn.state_dict())

        if ((episode + 1) % training_interval == 0) & (len(replay_buffer) > batch_size * training_interval):
                for _ in range(training_interval):
                    minibatch = random.sample(replay_buffer, batch_size)
                    replay_train(main_dqn, target_dqn, minibatch, criterion, optimizer)

    main_dqn.eval()
    simulate_bot(main_dqn)


if __name__ == "__main__":
    main()
