from .network_utils import *

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_unit=(400,300),gate=F.relu,gate_action=torch.tanh):
        super(Actor, self).__init__()
        dims=(state_size, ) + hidden_unit
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.gate_action = gate_action
        self.layers=nn.ModuleList([nn.Linear(dim_in,dim_out) for dim_in, dim_out in zip(dims[:-1],dims[1:])])
        self.action_layer= nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.action_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.gate_action(self.action_layer(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_unit=(400,300),gate=F.relu):
        super(Critic, self).__init__()
        dims=(state_size, ) + hidden_unit
        self.gate = gate
        self.seed = torch.manual_seed(seed)
        self.layers=nn.ModuleList([nn.Linear(dim_in,dim_out) for dim_in, dim_out in zip(dims[:-2],dims[1:-1])])
        self.state_action_layer = nn.Linear(dims[-2]+action_size, dims[-1])
        self.value_layer = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:layer.weight.data.uniform_(*hidden_init(layer))
        self.state_action_layer.weight.data.uniform_(*hidden_init(self.state_action_layer))
        self.value_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, action):
        for layer in self.layers: x = self.gate(layer(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.state_action_layer(x))
        return self.value_layer(x)
