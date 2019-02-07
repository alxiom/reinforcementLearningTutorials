import gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque

seed = 42
step_limit = 600
env = gym.make('MountainCar-v0')
env._max_episode_steps = step_limit
env.seed(seed)
torch.manual_seed(seed)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 200

learning_rate = 1e-3

gamma = 0.99
max_episodes = 500

initial_epsilon = 0.3
epsilon_decay = 0.99

break_episodes = 30
break_reward = -300


class DQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, inputs):
        return self.linear(inputs)


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
    optimizer = torch.optim.SGD(dqn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    reward_history = deque(maxlen=break_episodes)

    for episode in range(max_episodes):
        state = env.reset()
        eps = initial_epsilon
        episode_reward = 0

        for step in range(step_limit):
            x = torch.Tensor(state).float()
            q = dqn(x).data.numpy()

            if np.random.rand(1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q)

            next_state, reward, done, _ = env.step(action)

            next_x = torch.Tensor(next_state).float()
            next_q = dqn(next_x).data.numpy()
            q[action] = reward + gamma * np.max(next_q)

            loss = criterion(dqn(x), torch.Tensor(q))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                if step < step_limit:
                    eps *= epsilon_decay
                    scheduler.step()
                break

            episode_reward += reward
            state = next_state

        print(f"episode = {episode + 1}, rewards = {episode_reward}")

        reward_history.append(episode_reward)
        reward_avg = np.mean(reward_history)
        if reward_avg > break_reward:
            print(f"game cleared in {episode} episodes with avg step {reward_avg}")
            break

    dqn.eval()
    simulate_bot(dqn)


if __name__ == "__main__":
    main()
