import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.continuous.utils import Actor, Q_critic, MixNet


class CQL(object):

    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._bc_iteration = -1
        self._cql_n_action = 10
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

        self._q_network = Q_critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(observation_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)

        # policy-network
        self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)

    def policy_loss(self, o_with_id, o_with_a_id, a, result={}):
        policy_log_prob = self._policy_network.get_log_density(o_with_id, a)
        if self._iteration < self._bc_iteration:
            policy_loss = - (policy_log_prob).mean()
        else:
            q_values = self._q_network(o_with_a_id)
            policy_loss = (- policy_log_prob.unsqueeze(-1) * q_values).sum(dim=-2).mean()

        result.update({
            'policy_loss': policy_loss,
        })
        return result
    
    def q_loss(self, o_with_a_id, o_next_with_id, o_next, o, one_hot_agent_id, s_next, r, mask, s, result={}):
        """
        Q function Loss
        """
        next_new_actions= self._policy_network.get_action(o_next_with_id)
        q_next_target_values = self._q_target_network.q(o_next, next_new_actions, one_hot_agent_id)
        w_next_target, b_next_target = self._mix_target_network(s_next) 
        q_next_target_total = (w_next_target * q_next_target_values).sum(dim=-2) + b_next_target.squeeze(dim=-1)
        expected_q_total = r + self._gamma * mask * q_next_target_total

        q_values = self._q_network(o_with_a_id)
        w, b = self._mix_network(s)
        q_total = (w * q_values).sum(dim=-2) + b.squeeze(dim=-1)
        q_td_loss = F.mse_loss(q_total, expected_q_total.detach())
        """
        CQL Loss
        """
        random_actions_tensor = next_new_actions.new_empty((next_new_actions.shape[0] * self._cql_n_action, 
                                                            next_new_actions.shape[1], 
                                                            next_new_actions.shape[2]), 
                                                            requires_grad=False).uniform_(-1, 1).to(self._device)
        curr_actions_tensor = self.cql_get_policy_actions(o)
        new_curr_actions_tensor = self.cql_get_policy_actions(o_next)
        curr_actions_tensor = curr_actions_tensor.detach()
        new_curr_actions_tensor = new_curr_actions_tensor.detach()
        q_rand = self.cql_get_q_values(o, random_actions_tensor)
        q_curr_actions = self.cql_get_q_values(o, curr_actions_tensor)
        q_next_actions = self.cql_get_q_values(o, new_curr_actions_tensor)

        cql_s = torch.repeat_interleave(s, self._cql_n_action, 0)
        cql_w, cql_b = self._mix_network(cql_s)
        cql_w = cql_w.view(s.shape[0], self._cql_n_action, self._num_agent, 1)
        cql_b = cql_b.view(s.shape[0], self._cql_n_action, 1, 1)

        cat_q_total = torch.cat(
            [cql_w * q_rand + cql_b, cql_w * q_curr_actions + cql_b, cql_w * q_next_actions + cql_b], 1)

        min_qf_loss = torch.logsumexp(cat_q_total, dim=1).sum(dim=-2).mean()
        qf_loss = min_qf_loss - q_total.mean()

        q_loss = q_td_loss + qf_loss

        result.update({
            'q_loss': q_loss,
        })
        return result

    def cql_get_policy_actions(self, obs):
        obs_temp = torch.repeat_interleave(obs, self._cql_n_action, 0)
        one_hot_id = torch.eye(self._num_agent).expand(obs_temp.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((obs_temp, one_hot_id), dim=-1)
        actions = self._policy_network.get_action(o_with_id)

        return actions

    def cql_get_q_values(self, obs, action):
        action_shape = action.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = torch.repeat_interleave(obs, num_repeat, 0)
        one_hot_id = torch.eye(self._num_agent).expand(obs_temp.shape[0], -1, -1).to(self._device)
        o_with_a_id = torch.cat((obs_temp, action, one_hot_id), dim=-1)
        q_values = self._q_network(o_with_a_id)
        return q_values.view(obs_shape, self._cql_n_action, self._num_agent, 1)

    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        # Shared network values
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((o, a, one_hot_agent_id), dim=-1)

        # policy loss
        loss_result = self.policy_loss(o_with_id, o_with_a_id, a, result={})
        # q loss
        loss_result = self.q_loss(o_with_a_id, o_next_with_id, o_next, o, one_hot_agent_id, s_next, r, mask, s, result=loss_result)

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
