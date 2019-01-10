import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np


# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
gamma = 0.99
use_e_greedy = False
use_add_noise = True
num_episodes = 2000

rList = []
for i in range(num_episodes):
    e = 1.0 / (i / 100 + 1)
    state = env.reset()
    rAll = 0
    done = False

    while not done:

        if use_e_greedy & (np.random.rand(1) < e):
            action = env.action_space.sample()
        else:
            if use_add_noise:
                action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
            else:
                action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + gamma * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Sucess Rate : {}".format(str(sum(rList) / num_episodes)))
print("Final Q-Table Values")
print("Left Down Right Up")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()
