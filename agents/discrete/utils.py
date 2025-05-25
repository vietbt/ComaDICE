import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):

    def __init__(self, ob_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, 1)

    def forward(self, obs):
        x = self.f1.forward(obs).relu()
        h = self.f2.forward(x).relu()
        v = self.f3.forward(h)
        return v


class QNet(nn.Module):

    def __init__(self, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, ac_dim)

    def forward(self, obs_with_id):
        x = self.f1.forward(obs_with_id).relu()
        h = self.f2.forward(x).relu()
        q = self.f3.forward(h)
        return q


class Actor(nn.Module):

    def __init__(self, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, ac_dim)

    def forward(self, obs):
        x = self.f1.forward(obs).relu()
        h = self.f2.forward(x).relu()
        logits = self.f3.forward(h)
        return logits


class MixNet(nn.Module):

    def __init__(self, st_dim, n_agents, h_dim=64):
        super().__init__()
        self.f_v = nn.Linear(st_dim * n_agents, h_dim)
        self.w_v = nn.Linear(h_dim, n_agents)
        self.b_v = nn.Linear(h_dim, 1)

    def forward(self, states):
        batch_size, context_length, n_agents, st_dim = states.shape
        states = states.reshape(-1, st_dim * n_agents)
        x = self.f_v.forward(states).relu()
        w = self.w_v.forward(x).reshape(batch_size, context_length, n_agents, 1)
        b = self.b_v.forward(x).reshape(batch_size, context_length, 1, 1)
        return torch.abs(w), b


class VAE(nn.Module):

    def __init__(self, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.f1 = nn.Linear(ob_dim + n_agents, h_dim)
        self.f2 = nn.Linear(h_dim, h_dim)
        self.f3 = nn.Linear(h_dim, ac_dim)

    def forward(self, obs_with_id):
        x = self.f1.forward(obs_with_id).relu()
        h = self.f2.forward(x).relu()
        q = self.f3.forward(h)
        return q
