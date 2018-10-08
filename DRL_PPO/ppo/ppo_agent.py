import ptan
import numpy as np

class AgentAct(ptan.agent.BaseAgent):

    def __init__(self, net, device="cpu"):
        self.policy_net = net
        self.device = device

    def __call__(self, states, agent_states):

        states= ptan.agent.float32_preprocessor(states).to(self.device)
        mu= self.policy_net(states)
        mu = mu.data.cpu().numpy()
        logstd = self.policy_net.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentTest():

    def __init__(self, net, env, trials=10, device="cpu"):
        self.trials=trials
        self.policy_net=net
        self.env=env
        self.device=device
        self.rewards_list=[]
        self.steps_list=[]

    def __call__(self):
        for _ in range(self.trials):
            steps = 0
            rewards = 0
            state = self.env.reset()
            while True:
                state = ptan.agent.float32_preprocessor([state]).to(self.device)
                mu = self.policy_net(state)[0]
                action = mu.squeeze(dim=0).data.cpu().numpy()
                action = np.clip(action, -1, 1)
                state, reward, done, _ = self.env.step(action)
                rewards += reward
                steps += 1
                if done:
                    break
            self.rewards_list.append(rewards)
            self.steps_list.append(steps)
        return np.mean(self.rewards_list), np.mean(self.steps_list)