import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_unit=64):
        super(Actor, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(state_size, hidden_unit),
            nn.Tanh(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.Tanh(),
            nn.Linear(hidden_unit, action_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        return self.mu(x)

class Critic(nn.Module):
    def __init__(self, state_size, hidden_unit=64):
        super(Critic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, 1),
        )

    def forward(self, x):
            return self.value(x)