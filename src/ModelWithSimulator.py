import torch
import PyBotSim.src.simulator as Sim
import src.PolicyGradAgent as Agent


class Model:
    def __init__(self, inputSize, outputSize, hiddenSize, learningRate):
        self.sim = Sim.Simulator()
        self.agent = Agent.PolicyGradAgent(inputSize, outputSize,
                                           hiddenSize, learningRate)

    def compute_reward(self, robot_pos, goal_pos):
        distance = ((robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2) ** 0.5
        return -distance

    def run_episode(self, maxSteps=1000, draw=False):
        rewards = []
        log_probs = []
        self.sim.reset()
        for _ in range(50):
            input, dist, running, collision = self.sim.step(draw=draw)

        step = 0
        while running and step < maxSteps:
            action, action_prob = self.agent.select_action(input)
            # print(action_prob)
            input, dist, running, collision = self.sim.step(action=action,draw=draw)

            # print(input)

            # Compute reward based on progress towards goal
            reward = -abs(dist)
            if (collision):
                reward = -2
            rewards.append(reward)
            log_probs.append(action_prob)
            print(f"Reward: {reward}")
            # print(f"theta: {theta}")
            step += 1

        # Update policy using accumulated rewards and log probabilities
        self.agent.update_policy(rewards, log_probs)


def main():
    inputSize = 184
    hiddenSize = 100
    outputSize = 5
    learningRate = 0.01
    maxSteps = 2000

    model = Model(inputSize, outputSize, hiddenSize, learningRate)

    checkpoint = torch.load("model.ckpt")
    # model.agent.policy_network.load_state_dict(checkpoint['net_stat_dict'])
    # model.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    
    model.agent.policy_network.load_state_dict(checkpoint)

    # Training Loop
    epoch = 0
    while True:
        if epoch % 10 == 0:
            model.run_episode(maxSteps=maxSteps, draw=True)
            torch.save({
                           'net_state_dict': model.agent.policy_network.state_dict(),
                            'optimizer_state_dict': model.agent.optimizer.state_dict(),
                            'epoch': epoch}, "model2.ckpt")
            pass
        else:
            model.run_episode()
        epoch += 1
