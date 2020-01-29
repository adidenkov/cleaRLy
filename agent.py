import model as m
from visualize import update_viz


import torch.optim as optim
import gym
import statistics
import policy_gradient as pg


env = gym.make("CartPole-v1")

policy = m.DiscretePolicy(
    n_obs = env.observation_space.shape[0],
    n_acts = env.action_space.n,
    n_hidden = 64
)

policy_optimizer = optim.Adam(policy.parameters(), lr = 1e-3)

last_episode_rewards = []

for episode in range(100000):
    episode_reward = 0
    trajectories = []

    state = env.reset()

    if len(last_episode_rewards) == 100:
        avg_ep_reward = statistics.mean(last_episode_rewards)
        print(avg_ep_reward)
        update_viz(episode, avg_ep_reward)
        last_episode_rewards.clear()

    while len(trajectories) == 0 or not trajectories[-1]["done"]:
        state, episode_reward = pg.step_environment(state, policy, env, trajectories, episode_reward)

    last_episode_rewards.append(episode_reward)
    episode_reward = 0

    if episode % 10 == 0:
        pg.improve(policy, policy_optimizer, env, trajectories, episode_reward)