import torch
import torch.nn as nn
from agents.discrete.utils import Actor
from torch.distributions import Categorical


class Agent(nn.Module):

    def __init__(self, ob_dim, ac_dim, n_agents):
        super().__init__()
        self.actor = Actor(ob_dim, ac_dim, n_agents)


class BC(object):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, lr, device):
        self.grad_norm_clip = 1.0
        self.device = device

        self.model = Agent(ob_dim, ac_dim, n_agents).to(device)
        self.actor_param = list(self.model.actor.parameters())
        self.actor = self.model.actor

        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=lr)

    def update(self, states, obs, rewards, next_states, next_obs, actions, avails, n_agents):
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        agent_ids = torch.eye(n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        obs = torch.cat((obs, agent_ids), -1)

        logits = self.model.actor.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = - log_probs.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()
        
        return {"actor_loss": actor_loss.item()}