import torch
import numpy as np
from torch.utils.data import Dataset
from runner.utils import load_smac_data, load_mamujoco_data


class SMACDataset(Dataset):

    def __init__(self, env_name, mode, data, context_length=1):
        self.env_name = env_name
        self.mode = mode
        
        self.st_dim = data["states"].shape[-1]
        self.ob_dim = data["obs"].shape[-1]
        self.ac_dim = data["avails"].shape[-1]
        self.n_agents = data["obs"].shape[-2]

        states, obs, actions, done_ids, rewards, next_states, next_obs, avails, is_inits = load_smac_data(data, self.n_agents)

        self.context_length = context_length
        self.states = states
        self.obs = obs
        self.next_states = next_states
        self.next_obs = next_obs
        self.actions = actions
        self.avails = avails
        self.done_ids = done_ids
        self.rewards = rewards
        self.is_inits = is_inits
    
    @classmethod
    def load(cls, env_name, mode, context_length=1):
        data = np.load(f"offline_data/{env_name}_{mode}.npz", allow_pickle=True)
        data = {k: data[k] for k in data.keys()}
        return cls(env_name, mode, data, context_length)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, id):
        context_length = self.context_length
        done_id = id + context_length
        for i in self.done_ids[:, 0].tolist():
            if i > id:
                done_id = min(int(i), done_id)
                break
        id = done_id - context_length
        states = torch.tensor(self.states[id:done_id], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[id:done_id], dtype=torch.float32)
        obs = torch.tensor(self.obs[id:done_id], dtype=torch.float32)
        next_obs = torch.tensor(self.next_obs[id:done_id], dtype=torch.float32)
        actions = torch.tensor(self.actions[id:done_id], dtype=torch.int64)
        avails = torch.tensor(self.avails[id:done_id], dtype=torch.int64)
        rewards = torch.tensor(self.rewards[id:done_id], dtype=torch.float32).unsqueeze(-1)
        is_inits = torch.tensor(self.is_inits[id:done_id], dtype=torch.bool).unsqueeze(-1)
        return states, obs, rewards, next_states, next_obs, actions, avails, is_inits


class MaMujocoDataset:

    def __init__(self, env_name, mode, data, action_scale=4.0, action_bias=0.0):
        self.env_name = env_name
        self.mode = mode
        self.action_bias = action_bias
        self.action_scale = action_scale
        
        self.st_dim = data["states"].shape[-1]
        self.ob_dim = data["obs"].shape[-1]
        self.ac_dim = data["actions"].shape[-1]
        self.n_agents = data["obs"].shape[-2]

        states, obs, actions, rewards, masks, next_states, next_obs, is_inits = load_mamujoco_data(data)

        self.states = states
        self.obs = obs
        self.actions = (actions - action_bias) / action_scale
        self.rewards = rewards
        self.masks = masks
        self.next_states = next_states
        self.next_obs = next_obs
        self.is_inits = is_inits

    @classmethod
    def load(cls, env_name, mode, action_scale=4.0, action_bias=0.0):
        if env_name == "Ant-v2":
            agent_conf = "2x4"
        elif env_name == "HalfCheetah-v2":
            agent_conf = "6x1"
        elif env_name == "Hopper-v2":
            agent_conf = "3x1"
        else:
            raise NotImplementedError
        data = np.load(f"offline_data/{env_name}-{agent_conf}-{mode}.npz", allow_pickle=True)
        data = {k: data[k] for k in data.keys()}

        return cls(env_name, mode, data, action_scale, action_bias)

    def __len__(self):
        return len(self.states)
    
    def sample(self, batch_size, device="cuda", use_next_action=False):
        ids = np.random.randint(0, self.obs.shape[0], size=batch_size)
        states = torch.FloatTensor(self.states[ids]).to(device)
        obs = torch.FloatTensor(self.obs[ids]).to(device)
        actions = torch.FloatTensor(self.actions[ids]).to(device)
        rewards = torch.FloatTensor(self.rewards[ids]).to(device)
        masks = torch.FloatTensor(self.masks[ids]).to(device)
        next_states = torch.FloatTensor(self.next_states[ids]).to(device)
        next_obs = torch.FloatTensor(self.next_obs[ids]).to(device)
        is_inits = torch.BoolTensor(self.is_inits[ids]).to(device)
        if use_next_action:
            next_ids = ids + 1
            next_ids[next_ids >= self.obs.shape[0]] = 0
            next_actions = torch.FloatTensor(self.actions[next_ids]).to(device)
            return states, obs, actions, rewards, masks, next_states, next_obs, next_actions
        return states, obs, actions, rewards, masks, next_states, next_obs, is_inits