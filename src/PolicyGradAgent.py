import torch
import torch.optim as optim
import torch.nn as nn

from src.PolicyGradientNetwork import PolicyNetwork

discount_factor = 0.99
eps = 0.000000001

class PolicyGradAgent:
    def __init__(self, inputSize, outputSize, hiddenSize, lr):
        self.policy_network = PolicyNetwork(inputSize, outputSize, hiddenSize)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def select_action(self, input):
        # Picking from policy
        with torch.no_grad():
            # softplus = torch.nn.Softplus()

            input = input[181:]

            action_mean_l, action_mean_r = self.policy_network(torch.FloatTensor(input))

            action_dist_l = torch.distributions.Normal(action_mean_l, 15)
            action_dist_r = torch.distributions.Normal(action_mean_r, 15)

            action_l = action_dist_l.rsample()
            action_r = action_dist_r.rsample()

            action = (action_mean_l, action_mean_r)
            rand_action = (action_l, action_r)
            action_prob = action_dist_l.log_prob(action_l) * action_dist_r.log_prob(action_r)
            # print(action_prob)

            return action, rand_action, action_prob.unsqueeze(0)

    def update_policy(self, states, rewards, log_probs, actions):
        discounted_rewards = []
        cumulative_reward = 0

        # Process rewards
        for reward in reversed(rewards):
            # Calculate discounted rewards
            cumulative_reward = reward + discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # discounted_rewards = torch.nn.functional.normalize(discounted_rewards, p=3, dim=0)

        # Calculate policy loss
        self.optimizer.zero_grad()
        policy_loss = []
        for state, log_prob, reward, action in zip(states, log_probs, discounted_rewards, actions):
            # loss = -abs(-log_prob * reward)
            action_mean_l, action_mean_r = self.policy_network(torch.FloatTensor(state[181:]))
            print(action[0], action[1])
            print(reward)

            action_dist_l = torch.distributions.Normal(action_mean_l, 0.5)
            action_dist_r = torch.distributions.Normal(action_mean_r, 0.5)

            loss_l = -action_dist_l.log_prob(action[0]) * reward
            loss_r = -action_dist_r.log_prob(action[1]) * reward

            loss_l.backward(retain_graph=True)
            loss_r.backward()

            # Optimization
            self.optimizer.step()
