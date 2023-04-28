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
        states = []
        rewards = []
        log_probs = []
        actions = []

        self.sim.reset()
        for _ in range(50):
            input, dist, running, collision = self.sim.step(draw=draw)

        step = 0
        while running and step < maxSteps:
            states.append(input)
            mean, rand_action, action_prob = self.agent.select_action(input)

            x = torch.rand(1)
            action = mean if x < 0.75 else rand_action
            actions.append(action)

            input, dist, running, collision = self.sim.step(action=action,draw=draw)

            # print(action)
            # print(self.sim.robot.x)

            reward = dist
            # Compute reward based on progress towards goal
            # reward = 1-dist
            # if (collision):
                # reward = -2
            rewards.append(reward)
            # print(reward)
            log_probs.append(action_prob)
            # print(f"Reward: {reward}")
            # print(f"theta: {theta}")
            step += 1

        # Update policy using accumulated rewards and log probabilities
        self.agent.update_policy(states, rewards, log_probs, actions)


def main():
    inputSize = 3
    hiddenSize = 2
    outputSize = 2
    learningRate = 0.01
    maxSteps = 2000
    epoch = 0

    model = Model(inputSize, outputSize, hiddenSize, learningRate)

    # checkpoint = torch.load("DualNormDistModel3Inputs.ckpt")
    # model.agent.policy_network.load_state_dict(checkpoint['net_state_dict'])
    # model.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']

    # Training Loop
    while True:
        if epoch % 1 == 0:
            model.run_episode(maxSteps=maxSteps, draw=True)
            # torch.save({
            #            'net_state_dict': model.agent.policy_network.state_dict(),
            #             'optimizer_state_dict': model.agent.optimizer.state_dict(),
            #             'epoch': epoch}, "DualNormDistModel3Inputs.ckpt")
        else:
            model.run_episode(maxSteps=maxSteps)

        print("Epoch: ", epoch)
        epoch += 1
