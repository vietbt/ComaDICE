import torch
import torch.nn as nn
import torch.jit as jit


class VNet(jit.ScriptModule):

    def __init__(self, ob_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, 1)

    @jit.script_method
    def forward(self, obs):
        x = self.f1.forward(obs).relu()
        h = self.f2.forward(x).relu()
        v = self.f3.forward(h)
        return v


class QNet(jit.ScriptModule):

    def __init__(self, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + ac_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, 1)

    @jit.script_method
    def forward(self, obs):
        x = self.f1.forward(obs).relu()
        h = self.f2.forward(x).relu()
        q = self.f3.forward(h)
        return q


class Actor(jit.ScriptModule):

    def __init__(self, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, ac_dim)
        self.sigma = nn.Linear(h_dim, ac_dim)
        self.mean_min = -9.0
        self.mean_max = 9.0
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    @jit.script_method
    def forward(self, obs):
        x = self.f1.forward(obs).relu()
        h = self.f2.forward(x).relu()
        mean = self.mu.forward(h).clamp(self.mean_min, self.mean_max)
        logstd = self.sigma.forward(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(logstd)
        return mean, std


class MixNet(jit.ScriptModule):

    def __init__(self, st_dim, n_agents, h_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.f_v = nn.Linear(st_dim * n_agents, h_dim)
        self.w_v = nn.Linear(h_dim, n_agents)
        self.b_v = nn.Linear(h_dim, 1)

    @jit.script_method
    def forward(self, states):
        states = states.flatten(-2, -1)
        x = self.f_v.forward(states).relu()
        w = self.w_v.forward(x).reshape(-1, self.n_agents, 1)
        b = self.b_v.forward(x).reshape(-1, 1, 1)
        return torch.abs(w), b
    

class MADice(jit.ScriptModule):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.st_dim = st_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_agents = n_agents
        self.h_dim = h_dim

        self.q_mix_model = MixNet(st_dim, n_agents)
        self.q = QNet(ob_dim, ac_dim, n_agents, h_dim)
        self.v = VNet(ob_dim, n_agents, h_dim)
        self.actor = Actor(ob_dim, ac_dim, n_agents, h_dim)