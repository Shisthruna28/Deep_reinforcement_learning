from DeepRL import *
import gym
class train_agent():
    def __init__(self, config):
        self.agent=Agent(config)
        self.config=config

    def train_network(self):
        scores_deque = deque(maxlen=self.config.print_interval)
        scores = []
        for i_episode in range(1, self.config.train_episode):
            state = self.config.eval_env.reset()
            self.agent.reset()
            score = 0
            for t in range(self.config.episode_length):
                action = self.agent.act(state)
                next_state, reward, done, _ = config.eval_env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            torch.save(self.agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(self.agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if i_episode % self.config.print_interval == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    def random_network(self):
        scores_deque = deque(maxlen=self.config.print_interval)
        for i_episode in range(1, self.config.train_episode):
            state = self.config.eval_env.reset()
            score = 0
            for t in range(self.config.episode_length):
                action = self.config.eval_env.action_space.sample()
                next_state, reward, done, _ = self.config.eval_env.step(action)
                state = next_state
                score += reward
                if done: break
            scores_deque.append(score)
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

if __name__ == '__main__':
    config = Config()
    config.task_name='Pendulum-v0'
    config.eval_env = gym.make(config.task_name).unwrapped
    config.state_dim=config.eval_env.observation_space.shape[0]
    config.action_dim=config.eval_env.action_space.shape[0]
    train=train_agent(config)
    train.train_network()
