import torch

class Config:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount_factor = 0.99
        self.buffer_size = int(1e5)
        self.target_network_update_freq = 1e-3
        self.weight_dedcay = 0
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.batch_size = 128
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.random_seed=2
        self.train_episode=600
        self.episode_length=300
        self.print_interval=100
        self.test_interval=300
        self.state_dim = None
        self.action_dim = None
        self.task_name = None
        self.eval_env = None
