#!/usr/bin/env python3
from ppo import *
import os
import ptan
import time
import gym
import roboschool
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class PPO():

    def __init__(self, config):
        self.config = config
        # Actor Network
        self.actor_net = Actor(config.state_dim, config.action_dim).to(self.config.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=config.lr_actor)
        # Critic Network
        self.critic_net = Critic(config.state_dim, config.action_dim).to(self.config.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=config.lr_critic)
        # Agent for generating actions
        self.agent = AgentAct(self.actor_net, device=config.device)
        # Agent for testing
        self.agent_test = AgentTest(self.actor_net, config.test_env, device=config.device)
        # Experience Generator
        self.exp_source = ptan.experience.ExperienceSource(config.eval_env, self.agent, steps_count=1)
        # Writer
        self.writer=SummaryWriter(comment="-ppo_" + args.name)
        self.device=config.device

    def test(self,step_idx,best_reward):
        ts = time.time()
        rewards, steps = self.agent_test()
        print('----------------------------------------------------------------------------------')
        print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
        print('----------------------------------------------------------------------------------')
        #logging
        self.writer.add_scalar("test_reward", rewards, step_idx)
        self.writer.add_scalar("test_steps", steps, step_idx)

        if best_reward is None or best_reward < rewards:
            if best_reward is not None:
                # Saving the model
                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                fname = os.path.join(self.config.save_path, name) #####
                torch.save(self.actor_net.state_dict(), fname)

            best_reward = rewards
        return best_reward

    def train(self):
        trajectory = []
        best_reward = None
        mean_reward = 0.0
        steps = 0
        writer = SummaryWriter(comment="-ppo_" + args.name)
        with ptan.common.utils.RewardTracker(writer) as tracker:
            for step_idx, experience in enumerate(self.exp_source):

                trajectory.append(experience)

                rewards_steps = self.exp_source.pop_rewards_steps()
                # Logging steps done in each episode
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    self.writer.add_scalar("episode_length_steps", steps[0], step_idx)
                    mean_reward = tracker.reward(rewards[0], step_idx)

                # Testing the agent and saving the better model
                if step_idx % self.config.test_iters == 0:
                    best_reward = self.test(step_idx,best_reward)

                if step_idx > self.config.thshld_frme:
                    print( 'rewards after %d frames: %.3f'% (step_idx, mean_reward))
                    break
                if mean_reward and  mean_reward > self.config.thrshld_rewards:
                    print('rewards after %d frames: %.3f with %d steps' % (step_idx, mean_reward,steps[0]))
                    break

                if len(trajectory) < self.config.traj_size:
                    continue

                states = [t[0].state for t in trajectory]
                actions = [t[0].action for t in trajectory]
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)

                # Calculate and normalize advantages
                advantage_val, value_ref = calc_adv_ref(trajectory, self.critic_net, states, device=self.device)
                advantage_val = (advantage_val - torch.mean(advantage_val)) / torch.std(advantage_val)

                #calculate pi_old
                mu_v = self.actor_net(states)
                old_logprob_v = calc_logprob(mu_v, self.actor_net.logstd, actions)

                # drop last entry from the trajectory
                trajectory = trajectory[:-1]
                old_logprob_v = old_logprob_v[:-1].detach()

                sum_loss_value = 0.0
                sum_loss_policy = 0.0
                count_steps = 0

                for epoch in range(self.config.ppo_epochs):
                    for batch_ofs in range(0, len(trajectory), self.config.ppo_batch_size):
                        states_v = states[batch_ofs:batch_ofs + self.config.ppo_batch_size]
                        actions_v = actions[batch_ofs:batch_ofs + self.config.ppo_batch_size]
                        batch_adv_v = advantage_val[batch_ofs:batch_ofs + self.config.ppo_batch_size].unsqueeze(-1)
                        batch_ref_v = value_ref[batch_ofs:batch_ofs + self.config.ppo_batch_size]
                        batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + self.config.ppo_batch_size]

                        # critic training
                        self.critic_optimizer.zero_grad()
                        value_v = self.critic_net (states_v)
                        loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                        loss_value_v.backward()
                        self.critic_optimizer.step()

                        # actor training
                        self.actor_optimizer.zero_grad()
                        mu_v = self.actor_net(states_v)
                        # calculate pi
                        logprob_pi_v = calc_logprob(mu_v, self.actor_net.logstd, actions_v)
                        # calculate r_thetha
                        ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                        surrogate_1 = batch_adv_v * ratio_v
                        surrogate_2 = batch_adv_v * torch.clamp(ratio_v, 1.0 - self.config.ppo_eps, 1.0 + self.config.ppo_eps)
                        loss_policy_v = -torch.min(surrogate_1, surrogate_2).mean()
                        loss_policy_v.backward()
                        self.actor_optimizer.step()

                        sum_loss_value += loss_value_v.item()
                        sum_loss_policy += loss_policy_v.item()
                        count_steps += 1

                trajectory.clear()
                writer.add_scalar("advantage", advantage_val.mean().item(), step_idx)
                writer.add_scalar("values", value_ref.mean().item(), step_idx)
                writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
                writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)



if __name__ == "__main__":
    ENV_ID = "RoboschoolHalfCheetah-v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default='train', help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    config = Config()
    config.save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path, exist_ok=True)
    config.task_name = args.env
    config.eval_env = gym.make(config.task_name).unwrapped
    config.test_env = gym.make(config.task_name)
    config.state_dim = config.eval_env.observation_space.shape[0]
    config.action_dim = config.eval_env.action_space.shape[0]
    train_agent = PPO(config)
    train_agent.train()
