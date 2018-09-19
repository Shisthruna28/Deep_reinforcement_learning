from DeepRL import *
import gym

class test_agent():
    def __init__(self, config):
        self.agent=Agent(config)
        self.config=config

    def test_play(self):
        self.agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        self.agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
        state = config.eval_env.reset()
        for t in range(self.config.test_interval):
            action = self.agent.act(state, add_noise=False)
            config.eval_env.render()
            state, reward, done, _ = config.eval_env.step(action)
            if done:
                break
        config.eval_env.close()

if __name__ == '__main__':
    config = Config()
    config.task_name='Pendulum-v0'
    config.eval_env = gym.make(config.task_name).unwrapped
    config.state_dim=config.eval_env.observation_space.shape[0]
    config.action_dim=config.eval_env.action_space.shape[0]
    test=test_agent(config)
    test.test_play()
