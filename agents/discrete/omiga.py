import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from agents.discrete.utils import MixNet, QNet, VNet, Actor


class Agent(nn.Module):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents):
        super().__init__()
        self.q_mix_model = MixNet(st_dim, n_agents)
        self.v_mix_model = MixNet(st_dim, n_agents)

        self.q = QNet(ob_dim, ac_dim, n_agents)
        self.v = VNet(ob_dim, n_agents)
        self.actor = Actor(ob_dim, ac_dim, n_agents)


class OMIGA(object):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, lr, device):
        self.alpha = 10.0
        self.gamma = 0.99
        self.tau = 0.005
        self.grad_norm_clip = 1.0
        self.device = device

        self.model = Agent(st_dim, ob_dim, ac_dim, n_agents).to(device)
        self.v_param = list(self.model.v.parameters())
        self.q_param = list(self.model.q.parameters()) + list(self.model.q_mix_model.parameters())
        self.actor_param = list(self.model.actor.parameters())
        self.actor = self.model.actor

        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_param, lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_param, lr=lr)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.train(False)
    
    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, states, obs, rewards, next_states, next_obs, actions, avails, n_agents):
        states = states.to(self.device)
        obs = obs.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_obs = next_obs.to(self.device)
        actions = actions.to(self.device)
        avails = avails.to(self.device)

        rewards = rewards[:, :, 0, :]
        dones = (states==next_states).min(-1)[0].unsqueeze(-1).min(2)[0].float()
        agent_ids = torch.eye(n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        obs = torch.cat((obs, agent_ids), -1)
        next_obs = torch.cat((next_obs, agent_ids), -1)

        q_eval = torch.stack([self.model.q.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
        current_q = q_eval.gather(-1, actions)
        w_q, b_q = self.model.q_mix_model.forward(states)
        q_total = (w_q * current_q).sum(-2) + b_q.squeeze(-1)
        
        with torch.no_grad():
            v_next = torch.stack([self.target_model.v.forward(next_obs[:, :, j, :]) for j in range(n_agents)], 2)
            w_next, b_next = self.target_model.q_mix_model.forward(next_states)
            v_next_total = (w_next * v_next).sum(-2) + b_next.squeeze(-1)
            expected_q_total = rewards + self.gamma * (1 - dones) * v_next_total
        
        q_loss = F.mse_loss(q_total, expected_q_total)

        with torch.no_grad():
            target_q = torch.stack([self.target_model.q.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
            target_q = target_q.gather(-1, actions)
            target_w_q, target_b_q = self.target_model.q_mix_model.forward(states)

        v = torch.stack([self.model.v.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
        z = 1 / self.alpha * (target_w_q * target_q - target_w_q * v)
        z = torch.clamp(z, min=-10.0, max=10.0)

        with torch.no_grad():
            max_z = torch.max(z).clamp_min(-1.0)
            exp_a = torch.exp(z).squeeze(-1)

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * target_w_q * v / self.alpha
        v_loss = v_loss.mean()
        
        logits = torch.stack([self.model.actor.forward(obs[:, :, j, :]) for j in range(n_agents)], dim=2)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_param, self.grad_norm_clip)
        self.q_optimizer.step()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param, self.grad_norm_clip)
        self.v_optimizer.step()
        
        self.soft_update_target()
        
        return {"q_loss": q_loss.item(), "v_loss": v_loss.item(), "actor_loss": actor_loss.item()}