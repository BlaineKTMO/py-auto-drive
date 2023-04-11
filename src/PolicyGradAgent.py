import torch
import torch.optim as optim
import torch.nn as nn

from src.PolicyGradientNetwork import PolicyNetwork

discount_factor = 0.99
eps = 0.000000001

class PolicyGradAgent:
    def __init__(self, inputSize, outputSize, hiddenSize, lr):
        self.policy_network = PolicyNetwork(inputSize, outputSize, hiddenSize)
        self.optimizer = optim.SGD(self.policy_network.parameters(), lr=lr)

    def select_action(self, input):
        # Picking from policy
        # with torch.no_grad():
        #     action = self.policy_network(torch.FloatTensor(input))
        #     action_dist = torch.distributions.Categorical(action_probs)
        #     action = action_dist.sample()
        # return action.item(), action_probs[action]
        
        action_mean, action_std = self.policy_network(torch.FloatTensor(input))
        action_mean = torch.tensor([action_mean]) * torch.ones(2)
        action_std = torch.tensor([action_std]) * torch.ones(2)
        action_dist = torch.distributions.MultivariateNormal(action_mean, torch.diag_embed(action_std.unsqueeze(0) + eps))
        action = action_dist.sample()
        return action.numpy()[0], action_dist.log_prob(action)

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0

        # Process rewards
        for reward in reversed(rewards):
            # Calculate discounted rewards
            cumulative_reward = reward + discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Optimization
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
