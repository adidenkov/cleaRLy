import torch
from collections import deque


def step_environment(state, policy, env, trajectories, episode_reward):
    action = policy(torch.tensor(state, dtype=torch.float32))
    new_state, reward, done, _ = env.step(action.item())

    episode_reward += reward

    trajectories.append({
        "state": state,
        "action": action,
        "reward": torch.tensor([reward]),
        "done": done
    })

    return new_state, episode_reward

def improve(policy, policy_optimizer, env, trajectories, episode_reward):
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

    policy_loss = -(returns * log_probs).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
