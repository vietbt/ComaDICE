import torch
import numpy as np
from torch.distributions import Categorical


class RolloutWorkerDiscrete:

    def __init__(self, model, n_agents, device="cuda"):
        self.model = model
        self.n_agents = n_agents
        self.device = device
    
    def sample(self, obs, avails, deterministic=False):
        with torch.no_grad():
            obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).squeeze(0)
            avails = torch.tensor(np.array(avails), dtype=torch.float32, device=self.device).squeeze(0)
            inputs = torch.cat([obs, torch.eye(self.n_agents, device=self.device)], -1)
            logits = self.model.forward(inputs) + avails.log()
            logits = logits - logits.logsumexp(-1, True)
            actions = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
            actions = actions.cpu().numpy()
        return actions

    def rollout(self, env, num_episodes=32):
        self.model.eval()
        T_rewards, T_wins = [], 0
        for _ in range(num_episodes):
            reward_sum = 0
            obs, _, avails = env.reset()
            while True:
                actions = self.sample(obs, avails, deterministic=True)
                obs, _, rewards, dones, infos, avails = env.step(actions)
                reward_sum += np.mean(rewards)
                if np.all(dones):
                    T_rewards.append(reward_sum)
                    if infos[0]["won"]:
                        T_wins += 1
                    break
        avg_return = np.mean(T_rewards)
        avg_win_rate = T_wins / num_episodes
        self.model.train()
        return avg_return, avg_win_rate
    

class RolloutWorkerContinuous:

    def __init__(self, model, n_agents, action_scale=1, action_bias=0, device="cuda"):
        self.model = model
        self.n_agents = n_agents
        self.device = device
        self.action_scale = action_scale
        self.action_bias = action_bias
    
    def sample(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            one_hot_agent_id = torch.eye(self.n_agents).expand(obs.shape[0], -1, -1).to(self.device)
            o_with_id = torch.cat((obs, one_hot_agent_id), dim=-1)
            pretanh_actions, _ = self.model.forward(o_with_id)
            actions = torch.tanh(pretanh_actions)
            actions = self.action_scale * actions + self.action_bias
            actions = actions.cpu().numpy()
        return actions

    def rollout(self, env, num_episodes=32):
        self.model.eval()
        T_rewards = []
        for _ in range(num_episodes):
            reward_sum = 0
            obs, _, _ = env.reset()
            while True:
                actions = self.sample(obs)
                obs, _, rewards, dones, _, _ = env.step(actions)
                reward_sum += np.mean(rewards)
                if np.all(dones):
                    T_rewards.append(reward_sum)
                    break
        avg_return = np.mean(T_rewards)
        self.model.train()
        return avg_return