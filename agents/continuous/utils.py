import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class BCQ_Actor(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(BCQ_Actor, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_action)

        self.max_action = 1
        self.phi = 0.05
    
    def forward(self, obs, action, one_hot_id):
        a = F.relu(self.fc1(torch.cat((obs, action, one_hot_id), dim=-1)))
        a = F.relu(self.fc2(a))
        a = self.phi * self.max_action * torch.tanh(self.fc3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class Actor(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Actor, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution

    def get_log_density(self, x, y):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=-1)
        return logp_pi

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action

    def get_deterministic_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)
        return mu

class Double_Critic(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Double_Critic, self).__init__()
        self.device = device

        # Q1 architecture
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def q1(self, obs, action, agent_id):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device)   
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q1 = self.fc3(x)
        return q1

    def q2(self, obs, action, agent_id):
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        q2 = self.fc6(x)
        return q2

class V_critic(nn.Module):

    def __init__(self, num_state, num_agent, num_hidden, device):
        super(V_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v
    
    def v(self, obs, agent_id):
        x = torch.cat([obs, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class Q_critic(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Q_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q

    def q(self, obs, action, agent_id):
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(VAE, self).__init__()
        self.device = device
        self.latent_dim = num_agent * 2
        self.num_state = num_state
        self.hyper_hidden_dim = num_hidden
        self.num_action = num_action

        self.e1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.e2 = nn.Linear(num_hidden, num_hidden)
        self.mean = nn.Linear(num_hidden, self.latent_dim)
        self.log_std = nn.Linear(num_hidden, self.latent_dim)
        
        self.d1 = nn.Linear(num_state + self.latent_dim + num_agent, num_hidden)
        self.d2 = nn.Linear(num_hidden, num_hidden)
        self.d3 = nn.Linear(num_hidden, num_action)


    def forward(self, obs, action, one_hot_id):
        z = F.relu(self.e1(torch.cat((obs, action, one_hot_id), dim=-1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(obs, z, one_hot_id)

        return u, mean, std
    
    def decode(self, obs, one_hot_id, z=None):
        if z is None:
            z = torch.randn((obs.shape[0], obs.shape[1], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat((obs, z, one_hot_id), dim=-1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))
    
class MixNet(nn.Module):

    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(MixNet, self).__init__()
        self.device = device
        self.state_shape = num_state * num_agent
        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden
        self.num_action = num_action

        self.f_v = nn.Linear(self.state_shape, num_hidden)
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)


    def forward(self, states):
        batch_size = states.size(0)
        states = torch.cat([states[:, j, :] for j in range(self.n_agents)], dim=-1)
        states = states.reshape(-1, self.state_shape)
        x = self.f_v(states)
        w = self.w_v(F.relu(x)).reshape(batch_size, self.n_agents, 1)
        b = self.b_v(F.relu(x)).reshape(batch_size, 1, 1)
        return torch.abs(w), b


class Scalar(nn.Module):

    def __init__(self, init_value, device):
        super().__init__()
        self.device = device
        self.init_value = init_value
        self.constant = nn.Parameter(
            torch.tensor(self.init_value, requires_grad=True)
        )

    def forward(self):
        return self.constant