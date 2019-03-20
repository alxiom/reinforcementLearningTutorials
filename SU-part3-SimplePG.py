import gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

step_limit = 1000

torch.manual_seed(42)
env = gym.make("CartPole-v1")
env._max_episode_steps = step_limit
env.seed(42)

learning_rate = 1e-2
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, hidden_dim):
        super(Policy, self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim, bias=False),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim, bias=False),
            nn.Softmax(dim=-1)
        )

        self.gamma = gamma

        self.policy_history = torch.Tensor()
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        return self.model(x)


policy = Policy(hidden_dim=128)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state):
    state_tensor = torch.FloatTensor(state)
    action_tensor = policy(state_tensor)
    action_dist = Categorical(action_tensor)
    action = action_dist.sample()

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat((policy.policy_history, action_dist.log_prob(action).reshape(1)))
    else:
        policy.policy_history = (action_dist.log_prob(action))

    return action


def update_policy():
    reward = 0
    rewards = []

    for r in policy.reward_episode[::-1]:
        reward = r + policy.gamma * reward
        rewards.insert(0, reward)

    rewards_tensor = torch.FloatTensor(rewards)
    rewards_tensor_stdz = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + np.finfo(np.float32).eps)

    loss = (torch.sum(torch.mul(policy.policy_history, rewards_tensor_stdz).mul(-1), -1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor()
    policy.reward_episode = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        time = 0

        for time in range(step_limit):
            action = select_action(state)
            state, reward, done, _ = env.step(action.item())

            policy.reward_episode.append(reward)

            if done:
                break

        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy()

        if episode % 50 == 0:
            print(f"Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}")

        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is {running_reward} > reward threshold {env.spec.reward_threshold})")
            break


main(1000)
