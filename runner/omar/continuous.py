from runner.utils import evaluate
from agents.continuous import OMAR
from envs.mamujoco.env import MaMujocoWrapper
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, logdir, dataset, n_agents, lr, seed, device):
        config = {
            'gamma': 0.99,
            'tau': 0.005,
            'hidden_sizes': 256,
            'mix_hidden_sizes': 64,
            'batch_size': 128,
            'lr': lr,
            'grad_norm_clip': 1.0,
            'device': device
        }

        st_dim = dataset.st_dim
        ac_dim = dataset.ac_dim
        n_agents = dataset.n_agents

        self.eval_env = MaMujocoWrapper(dataset.env_name, seed)
        self.agent = OMAR(st_dim, ac_dim, n_agents, self.eval_env, config)
        self.writer = SummaryWriter(logdir)
        self.offline_data = dataset
        self.global_step = 0
        self.device = device
        self.task_name = logdir

    def eval(self, step, n_episodes=32):
        avg_return = evaluate(self.agent, self.eval_env, num_evaluation=n_episodes)
        self.writer.add_scalar("game/return", avg_return, step)
        print(f"Env: {self.task_name} - step: {step} - return: {avg_return:.3f}")
        return avg_return
    
    def train(self, n_epochs, n_evals=100, batch_size=128):
        log_interval = n_epochs * len(self.offline_data) // batch_size // n_evals
        while True:
            states, obs, actions, rewards, masks, next_states, next_obs, _ = self.offline_data.sample(batch_size, self.device)
            losses = self.agent.train_step(obs, states, actions, rewards, masks, next_states, next_obs, None)
            self.global_step += 1
            step = self.global_step // log_interval
            if self.global_step % log_interval == 0:
                # losses = {k: round(v.item(), 3) for k, v in losses.items()}
                # print(f"Step: {step} - {losses}")
                self.eval(step)
                if step >= n_evals:
                    return