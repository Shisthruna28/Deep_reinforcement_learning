from ..network import *
from ..component import *

import torchvision
import torch.optim as optim

class Agent():
    def __init__(self, config):
        self.config = config
        # Actor Network
        self.actor_local = Actor(config.state_dim, config.action_dim, config.random_seed).to(self.config.device)
        self.actor_target = Actor(config.state_dim, config.action_dim, config.random_seed).to(self.config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)
        # Critic Network
        self.critic_local = Critic(config.state_dim, config.action_dim, config.random_seed).to(self.config.device)
        self.critic_target = Critic(config.state_dim, config.action_dim, config.random_seed).to(self.config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_dedcay)
        # Noise process
        self.noise = OUNoise(config.action_dim, config.random_seed)
        # Replay memory
        self.memory = ReplayBuffer(config)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.config.batch_size:
            experiences = self.memory.sample()
            self.fit(experiences, self.config.discount_factor)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def fit(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        #Crtic update as in DQN
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #Actor Update
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() # As impemented in OpenAI baselines
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Target Update
        self.soft_update(self.critic_local, self.critic_target,self. config.target_network_update_freq)
        self.soft_update(self.actor_local, self.actor_target, self.config.target_network_update_freq)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
