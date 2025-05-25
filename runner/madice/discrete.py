import copy
import torch
import torch.nn as nn
from agents.discrete import MADice
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from runner.evaluate import RolloutWorkerDiscrete
from torch.utils.data.dataloader import DataLoader
from concurrent.futures import ThreadPoolExecutor as Pool


class Trainer:

    def __init__(self, model: MADice, logdir, offline_data, n_agents, alpha, lr, f_divergence, seed, device="cuda"):
        self.model = model
        self.device = device
        self._lamb_v = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        self._lamb_e = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

        self.v_param = list(self.model.v.parameters())
        self.q_param = list(self.model.q.parameters()) + list(self.model.q_mix_model.parameters())
        self.actor_param = list(self.model.actor.parameters())

        self.lr = lr
        self.gamma = 0.99
        self.tau = 0.005
        self.grad_norm_clip = 1.0
        self.global_step = 0
        
        self._lamb_scale = 1.0
        self._alpha = alpha
        self._f_divergence = f_divergence
        self._use_w_tot = True
        self._seed = seed

        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=self.lr)
        self.v_optimizer = torch.optim.Adam(self.v_param, lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_param, lr=self.lr)
        self._optim_lamb_v = torch.optim.Adam([self._lamb_v], lr=self.lr)
        self._optim_lamb_e = torch.optim.Adam([self._lamb_e], lr=self.lr)

        self.target_model = copy.deepcopy(model).eval()
        self.target_model.load_state_dict(model.state_dict())

        self.writer = SummaryWriter(logdir)
        self.task_name = logdir

        if any(name in offline_data.env_name for name in ["protoss", "terran", "zerg"]):
            from envs.smacv2.env import SMACWrapper
            self.ENV_CLS = SMACWrapper
        else:
            from envs.smacv1.env import SMACWrapper
            self.ENV_CLS = SMACWrapper

        self.n_agents = n_agents
        self.offline_data = offline_data
        self.data_loader = DataLoader(offline_data, shuffle=True, pin_memory=True, batch_size=128, num_workers=0, drop_last=True)

    def _r_fn(self, x):
        if self._f_divergence == "kl":
            return torch.exp(x - 1)
        elif self._f_divergence == "chisquare":
            return torch.clamp_min(x + 1, 0)
        elif self._f_divergence == "soft_chisquare":
            _x = x.clamp_max(0)
            return torch.where(x < 0, torch.exp(_x), x + 1)
    
    def _g_fn(self, x):
        if self._f_divergence == "kl":
            return (x - 1) * torch.exp(x - 1)
        elif self._f_divergence == "chisquare":
            return 0.5 * x ** 2
        elif self._f_divergence == "soft_chisquare":
            _x = x.clamp_max(0)
            return torch.where(x < 0, torch.exp(_x) * (_x - 1) + 1, 0.5 * x ** 2)
    
    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, states, obs, rewards, next_states, next_obs, actions, avails, is_inits, n_agents):
        self.model.train()
        self.global_step += 1

        states = states.to(self.device)
        obs = obs.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_obs = next_obs.to(self.device)
        actions = actions.to(self.device)
        avails = avails.to(self.device)
        is_inits = is_inits.to(self.device)

        rewards = rewards[:, :, 0, :]
        dones = (states==next_states).min(-1)[0].unsqueeze(-1).min(2)[0].float()
        agent_ids = torch.eye(n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        obs = torch.cat((obs, agent_ids), -1)
        next_obs = torch.cat((next_obs, agent_ids), -1)

        q_values = torch.stack([self.model.q.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
        q_values = q_values.gather(-1, actions)
        mw_q, mb_q = self.model.q_mix_model.forward(states)
        q_values = (mw_q * q_values).sum(-2) + mb_q.squeeze(-1)
        
        with torch.no_grad():
            next_v_values = torch.stack([self.target_model.v.forward(next_obs[:, :, j, :]) for j in range(n_agents)], 2)
            mw_next, mb_next = self.target_model.q_mix_model.forward(next_states)
            next_v_values = (mw_next * next_v_values).sum(-2) + mb_next.squeeze(-1)
            expected_q_values = rewards + self.gamma * (1 - dones) * next_v_values

        q_loss = F.mse_loss(q_values, expected_q_values)

        assert not torch.isnan(q_loss).any()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_param, self.grad_norm_clip)
        self.q_optimizer.step()
        
        with torch.no_grad():
            q_values = torch.stack([self.model.q.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
            q_values = q_values.gather(-1, actions)
            mw_q, mb_q = self.model.q_mix_model.forward(states)

        v_values = torch.stack([self.model.v.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
        e_v = mw_q * (q_values - v_values)

        preactivation_v = (e_v - self._lamb_scale * self._lamb_v) / self._alpha
        w_v = self._r_fn(preactivation_v)
        f_w_v = self._g_fn(preactivation_v).detach()

        e_v = e_v.detach()
        init_values = v_values[is_inits]
        v_loss0 = (1 - self.gamma) * torch.mean(init_values) if len(init_values) > 0 else torch.tensor(0.0)
        v_loss1 = torch.mean(- self._alpha * f_w_v)
        v_loss2 = torch.mean(w_v * (e_v - self._lamb_v))
        v_loss3 = self._lamb_v
        v_loss = v_loss0 + v_loss1 + v_loss2 + v_loss3

        assert not torch.isnan(v_loss).any()
        self.v_optimizer.zero_grad()
        self._optim_lamb_v.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param, self.grad_norm_clip)
        self.v_optimizer.step()
        self._optim_lamb_v.step()

        with torch.no_grad():
            target_q_values = torch.stack([self.target_model.q.forward(obs[:, :, j, :]) for j in range(n_agents)], 2)
            target_q_values = target_q_values.gather(-1, actions)
            target_w_q, target_b_q = self.target_model.q_mix_model.forward(states)
            e_values = target_w_q * (target_q_values - v_values)
            preactivation_e = (e_values - self._lamb_scale * self._lamb_e) / self._alpha
            w_e = self._r_fn(preactivation_e)
            f_w_e = self._g_fn(preactivation_e)
            exp_a = w_e.clamp_max(2).squeeze(-1)
            if self._use_w_tot:
                exp_a = exp_a.sum(-1, True)
        
        lamb_e_loss = torch.mean(- self._alpha * f_w_e + w_e * (e_v - self._lamb_scale * self._lamb_e) + self._lamb_e)

        assert not torch.isnan(lamb_e_loss).any()
        self._optim_lamb_e.zero_grad()
        lamb_e_loss.backward()
        self._optim_lamb_e.step()

        logits = torch.stack([self.model.actor.forward(obs[:, :, j, :]) for j in range(n_agents)], dim=2)
        logits = logits + avails.log()
        logits = logits - logits.logsumexp(-1, True)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        assert not torch.isnan(actor_loss).any()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.soft_update_target()

        losses = {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
        }
        
        return {k: round(v, 4) for k, v in losses.items()}

    def eval(self, actor, step, n_episodes=32):
        env = self.ENV_CLS(self.offline_data.env_name, self._seed)
        rollout_worker = RolloutWorkerDiscrete(actor, self.n_agents, self.device)
        avg_return, avg_win_rate = rollout_worker.rollout(env, n_episodes)
        env.close()
        self.writer.add_scalar("game/return", avg_return, step)
        self.writer.add_scalar("game/winrate", avg_win_rate, step)
        print(f"Env: {self.task_name} - step: {step} - return: {avg_return:.3f} - winrate: {avg_win_rate:.3f}")
        return avg_return

    def train(self, n_epochs, n_evals=100):
        log_interval = n_epochs * len(self.data_loader) // n_evals
        with Pool(4) as p:
            tasks = []
            while True:
                params = self.model.state_dict()
                for data in self.data_loader:
                    try:
                        losses = self.update(*data, self.n_agents)
                    except:
                        self.model.load_state_dict(params)
                    step = self.global_step // log_interval
                    if self.global_step % log_interval == 0:
                        # print(f"Step: {step} - {losses}")
                        tasks.append(p.submit(self.eval, copy.deepcopy(self.model.actor), step))
                        if step >= n_evals:
                            for task in tasks:
                                task.result()
                            return