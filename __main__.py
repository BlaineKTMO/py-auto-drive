import torch
from src.PolicyGradAgent import PolicyGradAgent

agent = PolicyGradAgent(180, 2, 32, 0.01)

def compute_reward(robot_pos, goal_pos):
    distance = ((robot_pos - goal_pos) ** 2).sum().sqrt()
    return -distance

def run_episode(policyAgent, env, goal_position):
    laser_scan, heading = env.reset()
    done = False
    rewards = []
    log_probs = []

    while not done:
        action, action_prob = policyAgent.select_action(laser_scan, heading)
        next_laser_scan, next_heading, reward, done = env.step(action)
        log_probs.append(torch.log(action_prob))

        # Compute reward based on progress towards goal
        current_position = env.get_robot_position()
        reward = compute_reward(current_position, goal_position)
        rewards.append(reward)
        log_probs.append(torch.log(action_prob))

    # Update policy using accumulated rewards and log probabilities
    policyAgent.update_policy(rewards, log_probs)
