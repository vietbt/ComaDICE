from agents.discrete import OMIGA
from torch.utils.tensorboard import SummaryWriter
from runner.evaluate import RolloutWorkerDiscrete
from torch.utils.data.dataloader import DataLoader


class Trainer:

    def __init__(self, logdir, dataset, n_agents, lr, seed, device):
        self.agent = OMIGA(dataset.st_dim, dataset.ob_dim, dataset.ac_dim, n_agents, lr, device)

        self.lr = lr
        self.alpha = 10.0
        self.gamma = 0.99
        self.tau = 0.005
        self.grad_norm_clip = 1.0
        self.global_step = 0
        self.seed = seed
        self.n_agents = n_agents
        self.device = device
        
        self.writer = SummaryWriter(logdir)
        self.task_name = logdir

        self.offline_data = dataset
        self.data_loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=0, drop_last=True)
        
        if any(name in dataset.env_name for name in ["protoss", "terran", "zerg"]):
            from envs.smacv2.env import Config
            from envs.smacv2.smac_env import SMACEnv
            self.eval_env = SMACEnv(Config(dataset.env_name, seed))
        else:
            from envs.smacv1.env import Config
            from envs.smacv1.smac_env import SMACEnv
            self.eval_env = SMACEnv(Config(dataset.env_name, seed))
        self.rollout_worker = RolloutWorkerDiscrete(self.agent.actor, self.n_agents, self.device)
    
    def eval(self, step, n_episodes=32):
        avg_return, avg_win_rate = self.rollout_worker.rollout(self.eval_env, n_episodes)
        self.writer.add_scalar("game/return", avg_return, step)
        self.writer.add_scalar("game/winrate", avg_win_rate, step)
        print(f"Env: {self.task_name} - step: {step} - return: {avg_return:.3f} - winrate: {avg_win_rate:.3f}")
        return avg_return
    
    def train(self, n_epochs, n_evals=100):
        log_interval = n_epochs * len(self.data_loader) // n_evals
        while True:
            for states, obs, rewards, next_states, next_obs, actions, avails, _ in self.data_loader:
                losses = self.agent.update(states, obs, rewards, next_states, next_obs, actions, avails, self.n_agents)
                self.global_step += 1
                step = self.global_step // log_interval
                if self.global_step % log_interval == 0:
                    # print(f"Step: {step} - {losses}")
                    self.eval(step)
                    if step >= n_evals:
                        return