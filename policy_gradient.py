import torch
from collections import deque
import statistics
from visualize import update_viz





# %%
episode_reward = 0
last_episode_rewards = []


# %%
for episode in range(100000):
    trajectories = []

    state = env.reset()

    if len(last_episode_rewards) == 100:
        avg_ep_reward = statistics.mean(last_episode_rewards)
        print(avg_ep_reward)
        update_viz(episode, avg_ep_reward)
        last_episode_rewards.clear()

    while len(trajectories) == 0 or not trajectories[-1]["done"]:
        action = policy(torch.tensor(state, dtype=torch.float32))
        new_state, reward, done, _ = env.step(action.item())

        episode_reward += reward

        trajectories.append({
            "state": state,
            "action": action,
            "reward": torch.tensor([reward]),
            "done": done
        })

        state = new_state

    last_episode_rewards.append(episode_reward)
    episode_reward = 0

    if episode % 10 == 0:

        states = torch.tensor([trajectory["state"] for trajectory in trajectories], dtype=torch.float32)
        actions = torch.tensor([trajectory["action"] for trajectory in trajectories], dtype=torch.float32)
        dones = torch.tensor([torch.tensor([1.0]) if trajectory["done"] else torch.tensor([0.0]) for trajectory in trajectories], dtype=torch.float32)
        rewards = [trajectory["reward"] for trajectory in trajectories]

        returns = [0] * len(rewards)
        discounted_future = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + discounted_future
            
            discounted_future = returns[i] * 0.99

        returns = torch.tensor(returns)

        mean = returns.mean()
        std = returns.std() + 1e-6
        returns = (returns - mean)/std
        
        log_probs = policy.log_prob(states, actions)

        policy_loss = -(torch.dot(returns, log_probs)).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
