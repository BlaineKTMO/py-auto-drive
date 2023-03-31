import torch
import torch.optim as optim
import torch.nn as nn

from src.PolicyGradientNetwork import PolicyNetwork


class PolicyGradAgent:
    def __init__(self, laserScanSize, outputSize, hiddenSize, lr):
        self.policy_network = PolicyNetwork(laserScanSize, outputSize, hiddenSize)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def select_action(self, laserScan, heading):
        with torch.no_grad():
            action_probs = self.policy_network(torch.FloatTensor(laserScan), torch.FloatTensor(heading))
            action_probs = nn.Softmax(dim=-1)(action_probs)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        return action.item(), action_probs[action]

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + 0.99 * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
