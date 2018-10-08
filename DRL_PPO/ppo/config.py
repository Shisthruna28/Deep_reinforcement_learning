import torch

class Config:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr_actor = 1e-4
        self.lr_critic = 1e-3

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_eps = 0.2

        self.traj_size = 2049
        self.ppo_epochs = 10
        self.ppo_batch_size = 64
        self.test_iters = 1000

        self.thshld_frme = 10000000
        self.thrshld_rewards = 3000
        self.thrshld_epi_steps = 1000


        self.state_dim = None
        self.action_dim = None
        self.task_name = None
        self.eval_env = None
        self.test_env=None

        self.save_path =None

