import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.continuous.utils import BCQ_Actor, Double_Critic, MixNet, VAE


class BCQ(object):

    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._gamma = config['gamma']
        self._tau = config['tau']
        self._hidden_sizes = config['hidden_sizes']
        self._mix_hidden_sizes = config['mix_hidden_sizes']
        self._vae_hidden_sizes = config['vae_hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._grad_norm_clip = config['grad_norm_clip']

        self._num_agent = num_agent
        self._device = config['device']
        self._eval_env = eval_env

        self._iteration = 0
        self._optimizers = dict()

        # vae-network
        self._vae_network = VAE(observation_spec, action_spec, num_agent, self._vae_hidden_sizes, self._device).to(self._device)
        self._optimizers['vae'] = torch.optim.Adam(self._vae_network.parameters(), self._lr)

        # Double q-network
        self._q_network = Double_Critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(observation_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)

        # policy-network
        self._policy_network = BCQ_Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._policy_target_network = copy.deepcopy(self._policy_network)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)

    def policy_loss(self, o, one_hot_agent_id, result={}):
        sampled_actions = self._vae_network.decode(o, one_hot_agent_id)
        perturbed_actions = self._policy_network(o, sampled_actions, one_hot_agent_id)

        q_values = self._q_network.q1(o, perturbed_actions, one_hot_agent_id)
        policy_loss = -q_values.mean()
        result.update({
            'policy_loss': policy_loss,
        })
        return result
    
    def vae_policy(self, o, a, one_hot_agent_id, result={}):
        recon, mean, std = self._vae_network(o, a, one_hot_agent_id)
        recon_loss = F.mse_loss(recon, a)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        result.update({
            'vae_loss': vae_loss,
        })
        return result

    def q_loss(self, o_next, o, s, s_next, mask, r, one_hot_agent_id, a, result={}):
        with torch.no_grad():
            next_o = torch.repeat_interleave(o_next, 10, 0)
            one_hot_id = torch.eye(self._num_agent).expand(next_o.shape[0], -1, -1).to(self._device)

            decode_a = self._vae_network.decode(next_o, one_hot_id)
            target_action =  self._policy_target_network(next_o, decode_a, one_hot_id)

            q1_next_target_values = self._q_target_network.q1(next_o, target_action, one_hot_id)
            q2_next_target_values = self._q_target_network.q2(next_o, target_action, one_hot_id)
            q_next_target_values = 0.75 * torch.min(q1_next_target_values, q2_next_target_values) + (1. - 0.75) * torch.max(q1_next_target_values, q2_next_target_values)
            q_next_target_values = torch.stack([q_next_target_values[:, j, :].reshape(o.shape[0], -1).max(1)[0].reshape(-1, 1) for j in range(self._num_agent)], dim=1)

            w_next_target, b_next_target = self._mix_target_network(s_next)
            q_next_target_total = (w_next_target * q_next_target_values).sum(dim=-2) + b_next_target.squeeze(dim=-1)
            expected_q_total = r + self._gamma * mask * q_next_target_total

        q1_values = self._q_network.q1(o, a, one_hot_agent_id)
        q2_values = self._q_network.q2(o, a, one_hot_agent_id)
        w, b = self._mix_network(s)
        q1_total = (w * q1_values).sum(dim=-2) + b.squeeze(dim=-1)
        q2_total = (w * q2_values).sum(dim=-2) + b.squeeze(dim=-1)
        q_loss = F.mse_loss(q1_total, expected_q_total) + F.mse_loss(q2_total, expected_q_total)
        
        result.update({
            'q_loss': q_loss,
        })
        return result

    
    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        # vae loss
        loss_result = self.vae_policy(o, a, one_hot_agent_id, result={})
        # q loss
        loss_result = self.q_loss(o_next, o, s, s_next, mask, r, one_hot_agent_id, a, result=loss_result)
        # policy loss
        loss_result = self.policy_loss(o, one_hot_agent_id, result=loss_result)

        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers['policy'].step()
        
        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers['q'].step()

        self._optimizers['vae'].zero_grad()
        loss_result['vae_loss'].backward()
        nn.utils.clip_grad_norm_(self._vae_network.parameters(), self._grad_norm_clip)
        self._optimizers['vae'].step()

        # soft update
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._policy_network.parameters(), self._policy_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        self._iteration += 1

        return loss_result

    def step(self, o):

        o = torch.from_numpy(o).to(self._device)
        state = o.repeat(100, 1, 1)
        one_hot_agent_id = torch.eye(self._num_agent).expand(state.shape[0], -1, -1).to(self._device)
        sampled_actions = self._vae_network.decode(state, one_hot_agent_id)
        perturbed_actions =  self._policy_network(state, sampled_actions, one_hot_agent_id)
        q_values = self._q_network.q1(state, perturbed_actions, one_hot_agent_id)

        ind = q_values[:, 0, :].argmax(0)
        a_one_agent = perturbed_actions[:, 0, :]
        a_one_agent = a_one_agent[ind].reshape(1,1,-1)
        if self._num_agent > 1:
            for k in range(1, self._num_agent):
                ind_other = q_values[:, k, :].argmax(0)
                a_other_agent = perturbed_actions[:, k, :]
                a_other_agent = a_other_agent[ind_other].reshape(1,1,-1)
                a_one_agent = torch.cat([a_one_agent, a_other_agent], dim=1)

        return a_one_agent.detach().cpu()

    def save(self, filedir):
        modeldir = os.path.join(filedir, 'model')
        os.makedirs(modeldir)
        torch.save(self._vae_network.state_dict(), os.path.join(modeldir, 'vae_network.pth'))
        torch.save(self._policy_network.state_dict(), os.path.join(modeldir, 'policy_network.pth'))
        torch.save(self._q_network.state_dict(), os.path.join(modeldir, 'q_network.pth'))
        torch.save(self._mix_network.state_dict(), os.path.join(modeldir, 'mix_network.pth'))

    def load(self, filedir):

        modeldir = os.path.join(filedir, 'model')
        self._vae_network.load_state_dict(torch.load(os.path.join(modeldir, 'vae_network.pth')))
        self._optimizers['vae'] = torch.optim.Adam(self._vae_network.parameters(), self._lr)

        self._policy_network.load_state_dict(torch.load(os.path.join(modeldir, 'policy_network.pth')))
        self._policy_target_network = copy.deepcopy(self._policy_network)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
    
        self._q_network.load_state_dict(torch.load(os.path.join(modeldir, 'q_network.pth')))
        self._q_target_network = copy.deepcopy(self._q_network)

        self._mix_network.load_state_dict(torch.load(os.path.join(modeldir, 'mix_network.pth')))
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)