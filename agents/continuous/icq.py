import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.continuous.utils import Actor, Double_Critic, MixNet


class ICQ(object):

    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._alpha = 10
        self._beta = 100
        self._lambda = 0.8
        self._gamma = config['gamma']
        self._tau = config['tau']
        self._hidden_sizes = config['hidden_sizes']
        self._mix_hidden_sizes = config['mix_hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._grad_norm_clip = config['grad_norm_clip']

        self._num_agent = num_agent
        self._device = config['device']
        self._eval_env = eval_env

        self._iteration = 0
        self._optimizers = dict()

        # q-network and mix-network
        self._q_network = Double_Critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(observation_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)

        # policy-network
        self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
    
    def q_loss(self, s_next, a_next, o_with_a_id, o_next_with_a_next_id, r, mask, s, result={}):
        
        with torch.no_grad():
            q1_next_target_values, q2_next_target_values = self._q_target_network(o_next_with_a_next_id)
            q_next_target_values = torch.min(q1_next_target_values, q2_next_target_values)

            w_target_next, b_target_next = self._mix_target_network(s_next)
            q_next_target_total = (w_target_next * q_next_target_values).sum(dim=-2) + b_target_next.squeeze(dim=-1)
            weights = F.softmax(q_next_target_total / self._alpha, dim=0)

            backup = r + self._gamma * mask * q_next_target_total * weights

        q1_values, q2_values = self._q_network(o_with_a_id)
        q_values = torch.min(q1_values, q2_values)
        w, b = self._mix_network(s)
        q_total = (w * q_values).sum(dim=-2) + b.squeeze(dim=-1)
        
        index = torch.zeros_like(q_total).to(self._device)

        for i in range(1, index.shape[0]):
            if mask[i] == 0:
                index[i] = 0
            else:
                index[i] = index[i-1] + 1
        
        decay = torch.ones_like(q_total) * self._lambda * self._gamma
        decay = torch.pow(decay, index)

        q_loss = ((decay * (q_total - backup.detach())) ** 2).mean()

        result.update({
            'q_loss': q_loss,
        })

        return result

    def policy_loss(self, o_with_id, o_with_a_id, a, s, result={}):
        w, _ = self._mix_network(s)
        q1_values, q2_values = self._q_network(o_with_a_id)
        q_values = torch.min(q1_values, q2_values)
        
        weights = F.softmax(w * q_values / self._beta, dim=0)

        log_prob = self._policy_network.get_log_density(o_with_id, a)
        policy_loss = (-log_prob.unsqueeze(-1) * weights.detach()).mean()

        result.update({
            'policy_loss': policy_loss,
        })
        return result


    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((o, a, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
        o_next_with_a_next_id = torch.cat((o_next, a_next, one_hot_agent_id), dim=-1)

        # q_loss
        loss_result = self.q_loss(s_next, a_next, o_with_a_id, o_next_with_a_next_id, r, mask, s, result={})
        # policy_loss
        loss_result = self.policy_loss(o_with_id, o_with_a_id, a, s, result=loss_result)

        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers['policy'].step()
        
        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers['q'].step()

        # soft update
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        self._iteration += 1

        return loss_result

    def step(self, o):
        o = torch.from_numpy(o).to(self._device)
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        action = self._policy_network.get_deterministic_action(o_with_id)

        return action.detach().cpu()

    def save(self, filedir):
        modeldir = os.path.join(filedir, 'model')
        os.makedirs(modeldir)

        torch.save(self._policy_network.state_dict(), os.path.join(modeldir, 'policy_network.pth'))
        torch.save(self._q_network.state_dict(), os.path.join(modeldir, 'q_network.pth'))
        torch.save(self._mix_network.state_dict(), os.path.join(modeldir, 'mix_network.pth'))

    def load(self, filedir):

        modeldir = os.path.join(filedir, 'model')
        self._policy_network.load_state_dict(torch.load(os.path.join(modeldir, 'policy_network.pth')))
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
    
        self._q_network.load_state_dict(torch.load(os.path.join(modeldir, 'q_network.pth')))
        self._q_target_network = copy.deepcopy(self._q_network)

        self._mix_network.load_state_dict(torch.load(os.path.join(modeldir, 'mix_network.pth')))
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)