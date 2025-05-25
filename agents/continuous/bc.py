import os
import torch
from agents.continuous.utils import Actor


class BC(object):

    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
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

        # policy-network
        self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
    
    def policy_loss(self, o_with_id, a, result={}):
        log_probs = self._policy_network.get_log_density(o_with_id, a)
        policy_loss = - torch.mean(log_probs)

        result.update({
            'policy_loss': policy_loss,
        })
        return result
    
    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        # Shared network values
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)

        # policy_loss
        loss_result = self.policy_loss(o_with_id, a, result={})
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

    def load(self, filedir):
        modeldir = os.path.join(filedir, 'model')
        self._policy_network.load_state_dict(torch.load(os.path.join(modeldir, 'policy_network.pth')))
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
